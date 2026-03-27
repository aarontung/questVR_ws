#!/usr/bin/env python3
import rospy
import tf2_ros
import rospkg
import geometry_msgs.msg
from std_msgs.msg import String
from tf.transformations import quaternion_from_matrix
from tf.transformations import quaternion_from_euler, quaternion_from_matrix, euler_from_quaternion

import casadi
import meshcat.geometry as mg
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.visualize import MeshcatVisualizer

import os
import json
import numpy as np
import math

from tools import MATHTOOLS
from piper_control import PIPER

from geometry_msgs.msg import PoseStamped

def matrix_to_xyzrpy(matrix):
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]
    roll = math.atan2(matrix[2, 1], matrix[2, 2])
    pitch = math.asin(-matrix[2, 0])
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    return [x, y, z, roll, pitch, yaw]


def create_transformation_matrix(x, y, z, roll, pitch, yaw):
    transformation_matrix = np.eye(4)
    A = np.cos(yaw)
    B = np.sin(yaw)
    C = np.cos(pitch)
    D = np.sin(pitch)
    E = np.cos(roll)
    F = np.sin(roll)
    DE = D * E
    DF = D * F
    transformation_matrix[0, 0] = A * C
    transformation_matrix[0, 1] = A * DF - B * E
    transformation_matrix[0, 2] = B * F + A * DE
    transformation_matrix[0, 3] = x
    transformation_matrix[1, 0] = B * C
    transformation_matrix[1, 1] = A * E + B * DF
    transformation_matrix[1, 2] = B * DE - A * F
    transformation_matrix[1, 3] = y
    transformation_matrix[2, 0] = -D
    transformation_matrix[2, 1] = C * F
    transformation_matrix[2, 2] = C * E
    transformation_matrix[2, 3] = z
    transformation_matrix[3, 0] = 0
    transformation_matrix[3, 1] = 0
    transformation_matrix[3, 2] = 0
    transformation_matrix[3, 3] = 1
    return transformation_matrix

def calc_pose_incre(base_pose, pose_data):
    begin_matrix = create_transformation_matrix(base_pose[0], base_pose[1], base_pose[2],
                                                base_pose[3], base_pose[4], base_pose[5])
    zero_matrix = create_transformation_matrix(0.19, 0.0, 0.2, 0, 0, 0)
    end_matrix = create_transformation_matrix(pose_data[0], pose_data[1], pose_data[2],
                                            pose_data[3], pose_data[4], pose_data[5])
    result_matrix = np.dot(zero_matrix, np.dot(np.linalg.inv(begin_matrix), end_matrix))
    xyzrpy = matrix_to_xyzrpy(result_matrix)
    return xyzrpy

class Arm_IK:
    def __init__(self):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        rospack = rospkg.RosPack()
        package_path = rospack.get_path('piper_description') 

        urdf_path = os.path.join(package_path, 'urdf', 'piper_description.urdf')
        urdf_path = self._prepare_runtime_urdf(urdf_path)
        
        self.robot = pin.RobotWrapper.BuildFromURDF(
            urdf_path,
            package_dirs=package_path
        )

        self.mixed_jointsToLockIDs = ["joint7",
                                      "joint8"
                                      ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0] * self.robot.model.nq),
        )

        self.first_matrix = create_transformation_matrix(0, 0, 0, 0, -1.57, 0)
        self.second_matrix = create_transformation_matrix(0.13, 0.0, 0.0, 0, 0, 0)  #第六轴到末端夹爪坐标的变换矩阵
        self.last_matrix = np.dot(self.first_matrix, self.second_matrix)
        q = quaternion_from_matrix(self.last_matrix)
        self.reduced_robot.model.addFrame(
            pin.Frame('ee',
                      self.reduced_robot.model.getJointId('joint6'),
                      pin.SE3(
                          # pin.Quaternion(1, 0, 0, 0),
                          pin.Quaternion(q[3], q[0], q[1], q[2]),
                          np.array([self.last_matrix[0, 3], self.last_matrix[1, 3], self.last_matrix[2, 3]]),  # -y
                      ),
                      pin.FrameType.OP_FRAME)
        )

        self.geom_model = pin.buildGeomFromUrdf(self.robot.model, urdf_path, pin.GeometryType.COLLISION)
        for i in range(4, 9):
            for j in range(0, 3):
                self.geom_model.addCollisionPair(pin.CollisionPair(i, j))
        self.geometry_data = pin.GeometryData(self.geom_model)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.history_data = np.zeros(self.reduced_robot.model.nq)

        # # Initialize the Meshcat visualizer  for visualization
        self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
        meshcat_open = os.getenv('MESHCAT_OPEN', '0').strip().lower() in ('1', 'true', 'yes', 'on')
        self.vis.initViewer(open=meshcat_open)
        self.vis.loadViewerModel("pinocchio")
        self.vis.displayFrames(True, frame_ids=[113, 114], axis_length=0.15, axis_width=5)
        self.vis.display(pin.neutral(self.reduced_robot.model))

        # Enable the display of end effector target frames with short axis lengths and greater width.
        frame_viz_names = ['ee_target']
        FRAME_AXIS_POSITIONS = (
            np.array([[0, 0, 0], [1, 0, 0],
                      [0, 0, 0], [0, 1, 0],
                      [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
        )
        FRAME_AXIS_COLORS = (
            np.array([[1, 0, 0], [1, 0.6, 0],
                      [0, 1, 0], [0.6, 1, 0],
                      [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
        )
        axis_length = 0.1
        axis_width = 10
        for frame_viz_name in frame_viz_names:
            self.vis.viewer[frame_viz_name].set_object(
                mg.LineSegments(
                    mg.PointsGeometry(
                        position=axis_length * FRAME_AXIS_POSITIONS,
                        color=FRAME_AXIS_COLORS,
                    ),
                    mg.LineBasicMaterial(
                        linewidth=axis_width,
                        vertexColors=True,
                    ),
                )
            )

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # # Get the hand joint ID and define the error function
        self.gripper_id = self.reduced_robot.model.getFrameId("ee")
        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.gripper_id].inverse() * cpin.SE3(self.cTf)
                    ).vector,
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        # self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf = self.opti.parameter(4, 4)

        # self.totalcost = casadi.sumsqr(self.error(self.var_q, self.param_tf))
        # self.regularization = casadi.sumsqr(self.var_q)

        error_vec = self.error(self.var_q, self.param_tf)
        pos_error = error_vec[:3]  
        ori_error = error_vec[3:]  
        weight_position = 1.0  
        weight_orientation = 0.1  
        self.totalcost = casadi.sumsqr(weight_position * pos_error) + casadi.sumsqr(weight_orientation * ori_error)
        self.regularization = casadi.sumsqr(self.var_q)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        # print("self.reduced_robot.model.lowerPositionLimit:", self.reduced_robot.model.lowerPositionLimit)
        # print("self.reduced_robot.model.upperPositionLimit:", self.reduced_robot.model.upperPositionLimit)
        self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization)
        # self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization + 0.1 * self.smooth_cost) # for smooth

        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 50,
                'tol': 1e-4
            },
            'print_time': False
        }
        self.opti.solver("ipopt", opts)
        self.enable_collision_check = str(os.getenv('TELEOP_CHECK_COLLISION', '0')).strip().lower() in ('1', 'true', 'yes', 'on')

    @staticmethod
    def _prepare_runtime_urdf(urdf_path):
        ws_root = os.getenv('QUESTVR_WS', os.getenv('PWD', '/workspace/questVR_ws'))
        try:
            with open(urdf_path, 'r', encoding='utf-8') as f:
                text = f.read()
            patched = text.replace('/home/agilex/questVR_ws', ws_root)
            patched = patched.replace('/home/<your name>/questVR_ws', ws_root)
            if patched != text:
                runtime_path = '/tmp/piper_description.single.runtime.urdf'
                with open(runtime_path, 'w', encoding='utf-8') as f:
                    f.write(patched)
                return runtime_path
        except Exception:
            pass
        return urdf_path

    def ik_fun(self, target_pose, gripper=0, motorstate=None, motorV=None):
        gripper = np.array([gripper/2.0, -gripper/2.0])
        if motorstate is not None:
            self.init_data = motorstate
        self.opti.set_initial(self.var_q, self.init_data)

        self.vis.viewer['ee_target'].set_transform(target_pose)     # for visualization

        self.opti.set_value(self.param_tf, target_pose)
        # self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            # sol = self.opti.solve()
            sol = self.opti.solve_limited()
            sol_q = self.opti.value(self.var_q)

            if self.init_data is not None:
                max_diff = max(abs(self.history_data - sol_q))
                # print("max_diff:", max_diff)
                self.init_data = sol_q
                if max_diff > 30.0/180.0*3.1415:
                    # print("Excessive changes in joint angle:", max_diff)
                    self.init_data = np.zeros(self.reduced_robot.model.nq)
            else:
                self.init_data = sol_q
            self.history_data = sol_q

            self.vis.display(sol_q)  # for visualization

            if motorV is not None:
                v = motorV * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            tau_ff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v,
                              np.zeros(self.reduced_robot.model.nv))

            if self.enable_collision_check:
                is_collision = self.check_self_collision(sol_q, gripper)
            else:
                is_collision = False
            dist = self.get_dist(sol_q, target_pose[:3, 3])
            # print("dist:", dist)
            return sol_q, tau_ff, is_collision

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")
            # sol_q = self.opti.debug.value(self.var_q)   # return original value
            return None, '', False

    def check_self_collision(self, q, gripper=np.array([0, 0])):
        pin.forwardKinematics(self.robot.model, self.robot.data, np.concatenate([q, gripper], axis=0))
        pin.updateGeometryPlacements(self.robot.model, self.robot.data, self.geom_model, self.geometry_data)
        collision = pin.computeCollisions(self.geom_model, self.geometry_data, False)
        # print("collision:", collision)
        return collision

    def get_dist(self, q, xyz):
        # print("q:", q)
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, np.concatenate([q], axis=0))
        dist = math.sqrt(pow((xyz[0] - self.reduced_robot.data.oMi[6].translation[0]), 2) + pow((xyz[1] - self.reduced_robot.data.oMi[6].translation[1]), 2) + pow((xyz[2] - self.reduced_robot.data.oMi[6].translation[2]), 2))
        return dist

    def get_pose(self, q):
        index = 6
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, np.concatenate([q], axis=0))
        end_pose = create_transformation_matrix(self.reduced_robot.data.oMi[index].translation[0], self.reduced_robot.data.oMi[index].translation[1], self.reduced_robot.data.oMi[index].translation[2],
                                                math.atan2(self.reduced_robot.data.oMi[index].rotation[2, 1], self.reduced_robot.data.oMi[index].rotation[2, 2]),
                                                math.asin(-self.reduced_robot.data.oMi[index].rotation[2, 0]),
                                                math.atan2(self.reduced_robot.data.oMi[index].rotation[1, 0], self.reduced_robot.data.oMi[index].rotation[0, 0]))
        end_pose = np.dot(end_pose, self.last_matrix)
        return matrix_to_xyzrpy(end_pose)


class VR:
    def __init__(self):
        self.piper_control = PIPER()
        self.tools = MATHTOOLS()
        self.inverse_solution = Arm_IK()
        self.piper_control.init_pose()

        self.latest_buttons = {'A': False, 'B': False, 'rightTrig': 0.0}
        self.last_buttons_time = rospy.Time(0)
        self.debug_print_hz = float(os.getenv('TELEOP_PRINT_HZ', '0') or 0.0)
        self.debug_print_period = (1.0 / self.debug_print_hz) if self.debug_print_hz > 0.0 else 0.0
        self.last_debug_print_time = rospy.Time(0)
        self.deadband_pos_m = float(os.getenv('TELEOP_DEADBAND_POS_M', '0.002') or 0.002)
        self.deadband_rot_rad = float(os.getenv('TELEOP_DEADBAND_ROT_RAD', '0.03') or 0.03)
        self.max_step_m = float(os.getenv('TELEOP_MAX_STEP_M', '0.01') or 0.01)
        self.rot_gain = float(os.getenv('TELEOP_ROT_GAIN', '1.0') or 1.0)
        self.joint_deadband_rad = float(os.getenv('TELEOP_JOINT_DEADBAND_RAD', '0.003') or 0.003)
        self.last_sent_q = None
        
        # 延时0.5秒，确保 OculusReader 初始化完成   
        import time
        time.sleep(0.5)

        self.target_pose = [0.19, 0.0, 0.2, 0, 0, 0]
        self.last_vr_pose = None
        self.latest_vr_pose = None
        self.control_rate_hz = float(os.getenv('TELEOP_CONTROL_HZ', '60') or 60.0)
        # 订阅回调
        rospy.Subscriber('/right_handle_pose', PoseStamped, self.handle_pose_callback, queue_size=1)
        rospy.Subscriber('/oculus_buttons', String, self.buttons_callback, queue_size=1)
        if self.control_rate_hz > 0.0:
            self.control_timer = rospy.Timer(rospy.Duration(1.0 / self.control_rate_hz), self.control_loop)

    def buttons_callback(self, msg):
        try:
            payload = json.loads(msg.data)
            self.latest_buttons['A'] = bool(payload.get('A', False))
            self.latest_buttons['B'] = bool(payload.get('B', False))
            self.latest_buttons['rightTrig'] = float(payload.get('rightTrig', 0.0))
            self.last_buttons_time = rospy.Time.now()
        except Exception as exc:
            rospy.logwarn_throttle(2.0, '[teleop_single] invalid /oculus_buttons payload: %s', exc)

    def _get_buttons(self):
        return dict(self.latest_buttons)
        
    def get_ik_solution(self, x,y,z,roll,pitch,yaw,gripper,b):
        
        q = quaternion_from_euler(roll, pitch, yaw)
        target = pin.SE3(
            pin.Quaternion(q[3], q[0], q[1], q[2]),
            np.array([x, y, z]),
        )
        sol_q, tau_ff, is_collision = self.inverse_solution.ik_fun(target.homogeneous,0)
        # print("result:", sol_q)
        if sol_q is None:
            return

        if is_collision:
            rospy.logwarn_throttle(1.0, "[teleop_single] self-collision detected, command skipped")
            return

        if b:
            if self.last_sent_q is not None:
                if max(abs(sol_q - self.last_sent_q)) < self.joint_deadband_rad:
                    return
            self.piper_control.joint_control_piper(
                sol_q[0], sol_q[1], sol_q[2], sol_q[3], sol_q[4], sol_q[5], gripper
            )
            self.last_sent_q = np.array(sol_q)

    def _integrate_relative_pose(self, prev_vr_pose, curr_vr_pose):
        dpos = math.sqrt(
            (curr_vr_pose[0] - prev_vr_pose[0]) ** 2 +
            (curr_vr_pose[1] - prev_vr_pose[1]) ** 2 +
            (curr_vr_pose[2] - prev_vr_pose[2]) ** 2
        )
        q_prev = quaternion_from_euler(prev_vr_pose[3], prev_vr_pose[4], prev_vr_pose[5])
        q_curr = quaternion_from_euler(curr_vr_pose[3], curr_vr_pose[4], curr_vr_pose[5])
        dot = abs(
            q_prev[0] * q_curr[0] +
            q_prev[1] * q_curr[1] +
            q_prev[2] * q_curr[2] +
            q_prev[3] * q_curr[3]
        )
        dot = min(1.0, max(-1.0, dot))
        drot = 2.0 * math.acos(dot)
        if dpos < self.deadband_pos_m and drot < self.deadband_rot_rad:
            return
        # Decouple translation from rotation to avoid axis coupling
        # (e.g. moving X causing Z drift due to frame composition).
        dx = (curr_vr_pose[0] - prev_vr_pose[0])
        dy = (curr_vr_pose[1] - prev_vr_pose[1])
        dz = (curr_vr_pose[2] - prev_vr_pose[2])
        if self.max_step_m > 0.0:
            dx = float(np.clip(dx, -self.max_step_m, self.max_step_m))
            dy = float(np.clip(dy, -self.max_step_m, self.max_step_m))
            dz = float(np.clip(dz, -self.max_step_m, self.max_step_m))
        self.target_pose[0] += dx
        self.target_pose[1] += dy
        self.target_pose[2] += dz

        q_target = quaternion_from_euler(self.target_pose[3], self.target_pose[4], self.target_pose[5])
        q_prev_inv = [-q_prev[0], -q_prev[1], -q_prev[2], q_prev[3]]

        def _qmul(q1, q2):
            x1, y1, z1, w1 = q1
            x2, y2, z2, w2 = q2
            return [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ]

        q_delta = _qmul(q_prev_inv, q_curr)
        if self.rot_gain != 1.0:
            qw = min(1.0, max(-1.0, q_delta[3]))
            angle = 2.0 * math.acos(qw)
            sin_half = math.sqrt(max(0.0, 1.0 - qw * qw))
            if sin_half > 1e-6:
                axis = [q_delta[0] / sin_half, q_delta[1] / sin_half, q_delta[2] / sin_half]
                scaled = angle * self.rot_gain
                half = 0.5 * scaled
                s = math.sin(half)
                q_delta = [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)]
        q_new = _qmul(q_target, q_delta)
        self.target_pose[3], self.target_pose[4], self.target_pose[5] = euler_from_quaternion(q_new)

    def handle_pose_callback(self, msg):
        self.latest_vr_pose = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            *euler_from_quaternion([
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w
            ])
        ]

    def control_loop(self, _event):
        if self.latest_vr_pose is None:
            return
        buttons = self._get_buttons()
        buttons_age = (rospy.Time.now() - self.last_buttons_time).to_sec()
        if buttons_age > 1.0:
            rospy.logwarn_throttle(2.0, '[teleop_single] waiting /oculus_buttons (age=%.2fs)', buttons_age)

        RR = list(self.latest_vr_pose)
        a_pressed = bool(buttons.get('A', False))
        b_pressed = bool(buttons.get('B', False))
        right_trig = float(buttons.get('rightTrig', 0.0))
        if a_pressed:
            # 按下A键后，机械臂回到初始点位并且记录 右 坐标原点
            self.piper_control.init_pose()
            self.target_pose = [0.19, 0.0, 0.2, 0, 0, 0]
            self.last_vr_pose = RR
        else:
            if not b_pressed:
                self.last_vr_pose = RR
            elif self.last_vr_pose is None:
                self.last_vr_pose = RR
            else:
                self._integrate_relative_pose(self.last_vr_pose, RR)
                self.last_vr_pose = RR

        if self.debug_print_period > 0.0:
            now = rospy.Time.now()
            if (now - self.last_debug_print_time).to_sec() >= self.debug_print_period:
                rospy.loginfo(
                    '[teleop_single][cmd_pose] xyz=(%.3f, %.3f, %.3f) rpy=(%.3f, %.3f, %.3f) B=%s rightTrig=%.3f',
                    self.target_pose[0], self.target_pose[1], self.target_pose[2],
                    self.target_pose[3], self.target_pose[4], self.target_pose[5],
                    b_pressed, right_trig
                )
                self.last_debug_print_time = now
        
        r_gripper_value = right_trig * 0.07
        # 按下B键后，开始遥操作（未按下时不跑IK，降低延迟）
        if b_pressed:
            self.get_ik_solution(
                self.target_pose[0], self.target_pose[1], self.target_pose[2],
                self.target_pose[3], self.target_pose[4], self.target_pose[5],
                r_gripper_value, b_pressed
            )
            

if __name__ == '__main__':
    rospy.init_node('teleop_single_piper_node')
    vr = VR()
    rospy.spin()
    

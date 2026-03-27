#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-questvr-gazebo:latest}"
WS_HOST="${WS_HOST:-/home/user/questVR_ws}"

# Host side: allow local root to use X11 (required for Gazebo GUI in container).
xhost +local:root >/dev/null 2>&1 || true

docker run --rm -it \
  --network host \
  -e DISPLAY="${DISPLAY:-:0}" \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "${WS_HOST}":/workspace/questVR_ws \
  "${IMAGE}" \
  bash -lc "set -e; \
    source /opt/ros/noetic/setup.bash; \
    [ -e /workspace/questVR_ws/src/piper_gazebo ] || ln -s /workspace/questVR_ws/piper_ros/src/piper_sim/piper_gazebo /workspace/questVR_ws/src/piper_gazebo; \
    mkdir -p /workspace/questVR_ws/src/Piper_ros/src/piper_description/urdf; \
    cp -f /workspace/questVR_ws/piper_ros/src/piper_description/urdf/piper_description_gazebo.xacro /workspace/questVR_ws/src/Piper_ros/src/piper_description/urdf/piper_description_gazebo.xacro; \
    cp -f /workspace/questVR_ws/piper_ros/src/piper_description/urdf/piper_no_gripper_description_gazebo.xacro /workspace/questVR_ws/src/Piper_ros/src/piper_description/urdf/piper_no_gripper_description_gazebo.xacro; \
    catkin_make -C /workspace/questVR_ws; \
    source /workspace/questVR_ws/devel/setup.bash; \
    roslaunch /workspace/questVR_ws/src/piper_gazebo/launch/piper_with_gripper/piper_gazebo.launch"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from http import HTTPStatus
import json
import os
import ssl
import threading
import time

import geometry_msgs.msg
import numpy as np
import rospy
import tf2_ros
import websockets
from std_msgs.msg import String
from tf.transformations import quaternion_from_matrix


WEBXR_SENDER_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>WebXR Pose Publisher</title>
  <style>
    body { font-family: sans-serif; margin: 24px; line-height: 1.4; }
    button { font-size: 18px; padding: 10px 18px; }
    pre { background: #111; color: #eee; padding: 12px; border-radius: 8px; overflow:auto; }
  </style>
</head>
<body>
  <h2>WebXR Pose Publisher</h2>
  <p id="status">status: idle</p>
  <button id="startBtn">Enter XR (Passthrough)</button>
  <pre id="log">waiting...</pre>
  <script>
    const statusEl = document.getElementById("status");
    const logEl = document.getElementById("log");
    const btn = document.getElementById("startBtn");
    let xrSession = null;
    let refSpace = null;
    let ws = null;
    let sendCounter = 0;
    let logCounter = 0;
    let seq = 0;

    function clamp(v) { return Math.max(0, Math.min(1, v || 0)); }
    function buttonPressed(gp, idx) { return !!(gp && gp.buttons && gp.buttons[idx] && gp.buttons[idx].pressed); }
    function buttonValue(gp, idx) { return gp && gp.buttons && gp.buttons[idx] ? gp.buttons[idx].value || 0 : 0; }
    function wsUrl() {
      const proto = window.location.protocol === "https:" ? "wss" : "ws";
      return proto + "://" + window.location.host + "/ws";
    }

    function pickAxes(gp) {
      if (!gp || !gp.axes || gp.axes.length < 2) return [0, 0];
      const axes = Array.from(gp.axes).map((v) => Number(v) || 0);
      if (gp.mapping === "xr-standard" && axes.length >= 4) return [axes[2], axes[3]];
      return [axes[0], axes[1]];
    }

    function stateFromGamepad(gp, hand) {
      const axes = pickAxes(gp);
      return {
        aButton: hand === "right" ? buttonPressed(gp, 4) : buttonPressed(gp, 4),
        bButton: hand === "right" ? buttonPressed(gp, 5) : buttonPressed(gp, 5),
        thumbstick: buttonPressed(gp, 3),
        squeeze: buttonPressed(gp, 1),
        trigger: buttonPressed(gp, 0),
        thumbstickValue: axes,
        squeezeValue: clamp(buttonValue(gp, 1)),
        triggerValue: clamp(buttonValue(gp, 0)),
      };
    }

    function connectSocket() {
      ws = new WebSocket(wsUrl());
      return new Promise((resolve, reject) => {
        ws.addEventListener("open", () => { statusEl.textContent = "status: websocket connected"; resolve(); }, { once: true });
        ws.addEventListener("error", (e) => { statusEl.textContent = "status: websocket error"; reject(e); }, { once: true });
        ws.addEventListener("close", () => { statusEl.textContent = "status: websocket closed"; });
      });
    }

    function sendPayload(payload) {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      try {
        ws.send(JSON.stringify(payload));
      } catch (e) {
        statusEl.textContent = "status: send failed " + e;
      }
    }

    function onXRFrame(t, frame) {
      const session = frame.session;
      const payload = {};
      for (const input of session.inputSources) {
        const hand = input.handedness;
        if (!hand || (hand !== "left" && hand !== "right")) continue;
        const space = input.gripSpace || input.targetRaySpace;
        if (space) {
          const pose = frame.getPose(space, refSpace);
          if (pose && pose.transform && pose.transform.matrix) {
            payload[hand] = Array.from(pose.transform.matrix);
          }
        }
        payload[hand + "State"] = stateFromGamepad(input.gamepad, hand);
      }
      payload._meta = {
        seq: seq++,
        clientTsMs: Date.now(),
      };
      sendCounter += 1;
      sendPayload(payload);
      logCounter += 1;
      if (logCounter % 10 === 0) {
        logEl.textContent = JSON.stringify(payload);
      }
      session.requestAnimationFrame(onXRFrame);
    }

    btn.onclick = async () => {
      if (!navigator.xr) {
        statusEl.textContent = "status: WebXR not available";
        return;
      }
      try {
        statusEl.textContent = "status: connecting websocket";
        await connectSocket();
        const params = new URLSearchParams(window.location.search);
        const wantPassthrough = params.get("passthrough") !== "0";
        let sessionMode = "immersive-vr";
        if (wantPassthrough) {
          try {
            const arSupported = await navigator.xr.isSessionSupported("immersive-ar");
            if (arSupported) sessionMode = "immersive-ar";
          } catch (_) {}
        }
        xrSession = await navigator.xr.requestSession(sessionMode, {
          optionalFeatures: ["local-floor", "bounded-floor", "hand-tracking"]
        });
        const canvas = document.createElement("canvas");
        const gl = canvas.getContext("webgl", { xrCompatible: true });
        await gl.makeXRCompatible();
        xrSession.updateRenderState({ baseLayer: new XRWebGLLayer(xrSession, gl) });
        try {
          refSpace = await xrSession.requestReferenceSpace("local-floor");
        } catch (_) {
          refSpace = await xrSession.requestReferenceSpace("local");
        }
        statusEl.textContent = "status: running (" + sessionMode + ")";
        xrSession.requestAnimationFrame(onXRFrame);
      } catch (e) {
        statusEl.textContent = "status: failed " + e;
      }
    };
  </script>
</body>
</html>
"""


def default_buttons():
    return {
        'A': False,
        'B': False,
        'X': False,
        'Y': False,
        'RTr': False,
        'LTr': False,
        'rightTrig': (0.0,),
        'leftTrig': (0.0,),
        'rightJS': (0.0, 0.0),
        'leftJS': (0.0, 0.0),
    }


class WebXRDataSource:
    def __init__(self, host, port, cert=None, key=None, debug=False):
        self.host = host
        self.port = int(port)
        self.cert = cert
        self.key = key
        self.debug = bool(debug)
        self._lock = threading.Lock()
        self._loop = None
        self._stop_event = None
        self.thread = None
        self.transforms = {}
        self.buttons = default_buttons()
        self.rx_count = 0
        self.last_rx_wall = 0.0
        self.last_seq = -1
        self.last_client_ts_ms = 0.0
        self.init_error = None

    @staticmethod
    def _matrix_from_webxr(values):
        arr = np.asarray(values, dtype=float)
        if arr.size != 16:
            return None
        return arr.reshape((4, 4), order='F')

    @staticmethod
    def _vec2(value):
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return (0.0, 0.0)
        return (float(value[0]), float(value[1]))

    @staticmethod
    def _vec1(value):
        if value is None:
            return (0.0,)
        return (float(value),)

    def _parse_buttons(self, payload):
        out = default_buttons()
        r_state = payload.get('rightState') if isinstance(payload, dict) else None
        l_state = payload.get('leftState') if isinstance(payload, dict) else None
        if isinstance(r_state, dict):
            out['A'] = bool(r_state.get('aButton', False))
            out['B'] = bool(r_state.get('bButton', False))
            out['RTr'] = bool(r_state.get('trigger', False))
            out['rightJS'] = self._vec2(r_state.get('thumbstickValue'))
            out['rightTrig'] = self._vec1(r_state.get('triggerValue'))
        if isinstance(l_state, dict):
            out['X'] = bool(l_state.get('aButton', False))
            out['Y'] = bool(l_state.get('bButton', False))
            out['LTr'] = bool(l_state.get('trigger', False))
            out['leftJS'] = self._vec2(l_state.get('thumbstickValue'))
            out['leftTrig'] = self._vec1(l_state.get('triggerValue'))
        return out

    def _update(self, payload):
        now = time.time()
        transforms = dict(self.transforms)
        if isinstance(payload, dict):
            if 'right' in payload:
                right = self._matrix_from_webxr(payload['right'])
                if right is not None:
                    transforms['r'] = right
            if 'left' in payload:
                left = self._matrix_from_webxr(payload['left'])
                if left is not None:
                    transforms['l'] = left
        buttons = self._parse_buttons(payload if isinstance(payload, dict) else {})
        seq = -1
        client_ts_ms = 0.0
        if isinstance(payload, dict):
            meta = payload.get('_meta')
            if isinstance(meta, dict):
                try:
                    seq = int(meta.get('seq', -1))
                except Exception:
                    seq = -1
                try:
                    client_ts_ms = float(meta.get('clientTsMs', 0.0))
                except Exception:
                    client_ts_ms = 0.0
        with self._lock:
            self.transforms = transforms
            self.buttons = buttons
            self.rx_count += 1
            self.last_rx_wall = now
            self.last_seq = seq
            self.last_client_ts_ms = client_ts_ms

    async def _process_http_request(self, path, request_headers):
        del request_headers
        if path == '/ws':
            return None
        if path in ('/', '/index.html'):
            body = WEBXR_SENDER_HTML.encode('utf-8')
            return (
                HTTPStatus.OK,
                [('Content-Type', 'text/html; charset=utf-8'), ('Content-Length', str(len(body)))],
                body,
            )
        body = b'Not Found\n'
        return (
            HTTPStatus.NOT_FOUND,
            [('Content-Type', 'text/plain; charset=utf-8'), ('Content-Length', str(len(body)))],
            body,
        )

    async def _ws_handler(self, websocket, path):
        if path != '/ws':
            await websocket.close(code=1008, reason='invalid path')
            return
        if self.debug:
            rospy.loginfo('[webxr_pub_pose] websocket connected: %s', str(websocket.remote_address))
        try:
            async for message in websocket:
                try:
                    payload = json.loads(message)
                except Exception:
                    continue
                self._update(payload)
        finally:
            if self.debug:
                rospy.loginfo('[webxr_pub_pose] websocket closed: %s', str(websocket.remote_address))

    def _run(self):
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._stop_event = asyncio.Event()
            ssl_context = None
            if self.cert and self.key:
                ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ssl_context.load_cert_chain(certfile=self.cert, keyfile=self.key)
                rospy.loginfo('[webxr_pub_pose] HTTPS + WSS at https://%s:%d/', self.host, self.port)
            else:
                rospy.logwarn('[webxr_pub_pose] HTTP + WS at http://%s:%d/ (WebXR usually needs HTTPS)', self.host, self.port)

            async def _start_server():
                return await websockets.serve(
                    self._ws_handler,
                    self.host,
                    self.port,
                    process_request=self._process_http_request,
                    ssl=ssl_context,
                )

            server = self._loop.run_until_complete(_start_server())
            self._loop.run_until_complete(self._stop_event.wait())
            server.close()
            self._loop.run_until_complete(server.wait_closed())
            self._loop.close()
        except Exception as exc:
            self.init_error = RuntimeError('Failed to start webxr server: {}'.format(exc))

    def start(self):
        self.init_error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        time.sleep(0.1)
        if self.init_error is not None:
            raise self.init_error

    def stop(self):
        if self._loop is not None and self._stop_event is not None:
            try:
                if not self._loop.is_closed():
                    self._loop.call_soon_threadsafe(self._stop_event.set)
            except RuntimeError:
                pass
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def get(self):
        with self._lock:
            return dict(self.transforms), dict(self.buttons)

    def get_debug(self):
        with self._lock:
            return {
                'rx_count': int(self.rx_count),
                'last_rx_wall': float(self.last_rx_wall),
                'last_seq': int(self.last_seq),
                'last_client_ts_ms': float(self.last_client_ts_ms),
            }


class WebXRPosePublisher:
    def __init__(self):
        rospy.init_node('webxr_pub_pose_node')
        self.right_handle_pose_pub = rospy.Publisher('right_handle_pose', geometry_msgs.msg.PoseStamped, queue_size=1)
        self.left_handle_pose_pub = rospy.Publisher('left_handle_pose', geometry_msgs.msg.PoseStamped, queue_size=1)
        self.buttons_pub = rospy.Publisher('oculus_buttons', String, queue_size=1)
        self.br = tf2_ros.TransformBroadcaster()
        self.rate = rospy.Rate(100)
        self.debug_hz = float(os.getenv('WEBXR_PRINT_HZ', '0') or 0.0)
        self.debug_period = (1.0 / self.debug_hz) if self.debug_hz > 0.0 else 0.0
        self._last_debug_time = 0.0
        self._prev_rx_count = 0
        self._prev_rx_wall = 0.0

        host = os.getenv('WEBXR_HOST', '0.0.0.0')
        port = int(os.getenv('WEBXR_PORT', '8012'))
        cert = os.getenv('WEBXR_CERT')
        key = os.getenv('WEBXR_KEY')
        debug = str(os.getenv('WEBXR_DEBUG_EVENTS', '0')).lower() in ('1', 'true', 'yes', 'on')
        self.source = WebXRDataSource(host=host, port=port, cert=cert, key=key, debug=debug)
        self.source.start()

    def xyzrpy2mat(self, x, y, z, roll, pitch, yaw):
        mat = np.eye(4)
        a = np.cos(yaw)
        b = np.sin(yaw)
        c = np.cos(pitch)
        d = np.sin(pitch)
        e = np.cos(roll)
        f = np.sin(roll)
        de = d * e
        df = d * f
        mat[0, 0] = a * c
        mat[0, 1] = a * df - b * e
        mat[0, 2] = b * f + a * de
        mat[0, 3] = x
        mat[1, 0] = b * c
        mat[1, 1] = a * e + b * df
        mat[1, 2] = b * de - a * f
        mat[1, 3] = y
        mat[2, 0] = -d
        mat[2, 1] = c * f
        mat[2, 2] = c * e
        mat[2, 3] = z
        return mat

    def adjustment_matrix(self, transform):
        adj_mat = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        r_adj = self.xyzrpy2mat(0, 0, 0, -np.pi, 0, -np.pi / 2)
        transform = adj_mat @ transform
        transform = np.dot(transform, r_adj)
        return transform

    def _publish_transform(self, transform, child_name, pub):
        translation = transform[:3, 3]
        quat = quaternion_from_matrix(transform)

        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = 'vr_device'
        t.child_frame_id = child_name
        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        self.br.sendTransform(t)

        pose_msg = geometry_msgs.msg.PoseStamped()
        pose_msg.header.stamp = t.header.stamp
        pose_msg.header.frame_id = 'vr_device'
        pose_msg.pose.position.x = translation[0]
        pose_msg.pose.position.y = translation[1]
        pose_msg.pose.position.z = translation[2]
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        pub.publish(pose_msg)

    def _publish_buttons(self, buttons):
        payload = {
            'A': bool(buttons.get('A', False)),
            'B': bool(buttons.get('B', False)),
            'X': bool(buttons.get('X', False)),
            'Y': bool(buttons.get('Y', False)),
            'RTr': bool(buttons.get('RTr', False)),
            'LTr': bool(buttons.get('LTr', False)),
            'rightTrig': float((buttons.get('rightTrig', (0.0,)) or (0.0,))[0]),
            'leftTrig': float((buttons.get('leftTrig', (0.0,)) or (0.0,))[0]),
            'rightJS': list(buttons.get('rightJS', (0.0, 0.0))),
            'leftJS': list(buttons.get('leftJS', (0.0, 0.0))),
        }
        self.buttons_pub.publish(String(data=json.dumps(payload)))

    def run(self):
        try:
            while not rospy.is_shutdown():
                transforms, buttons = self.source.get()
                self._publish_buttons(buttons)
                if 'r' in transforms:
                    self._publish_transform(self.adjustment_matrix(transforms['r']), 'right_controller', self.right_handle_pose_pub)
                if 'l' in transforms:
                    self._publish_transform(self.adjustment_matrix(transforms['l']), 'left_controller', self.left_handle_pose_pub)

                if self.debug_period > 0.0:
                    now = time.time()
                    if now - self._last_debug_time >= self.debug_period:
                        dbg = self.source.get_debug()
                        age = (now - dbg['last_rx_wall']) if dbg['last_rx_wall'] > 0.0 else 1e9
                        if self._prev_rx_wall > 0.0 and dbg['last_rx_wall'] > self._prev_rx_wall:
                            dt = dbg['last_rx_wall'] - self._prev_rx_wall
                            dcount = max(0, dbg['rx_count'] - self._prev_rx_count)
                            rx_hz = (float(dcount) / dt) if dt > 1e-6 else 0.0
                        else:
                            rx_hz = 0.0
                        client_age_ms = (now * 1000.0 - dbg['last_client_ts_ms']) if dbg['last_client_ts_ms'] > 0.0 else -1.0
                        rospy.loginfo(
                            '[webxr_debug] rx_hz=%.1f age=%.3fs client_age=%.1fms seq=%d A=%s B=%s rt=%.3f hasR=%s hasL=%s',
                            rx_hz,
                            age,
                            client_age_ms,
                            dbg['last_seq'],
                            bool(buttons.get('A', False)),
                            bool(buttons.get('B', False)),
                            float((buttons.get('rightTrig', (0.0,)) or (0.0,))[0]),
                            ('r' in transforms),
                            ('l' in transforms),
                        )
                        self._prev_rx_count = dbg['rx_count']
                        self._prev_rx_wall = dbg['last_rx_wall']
                        self._last_debug_time = now
                self.rate.sleep()
        finally:
            self.source.stop()


if __name__ == '__main__':
    WebXRPosePublisher().run()

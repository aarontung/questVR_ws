#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-questvr-webxr:latest}"
WS_HOST="${WS_HOST:-/home/user/questVR_ws}"
CERT_DIR="${CERT_DIR:-/home/user/questVR_ws/certs}"

if [[ ! -f "${CERT_DIR}/fullchain.pem" || ! -f "${CERT_DIR}/privkey.pem" ]]; then
  echo "[err] cert files not found under ${CERT_DIR}"
  echo "[err] expected: ${CERT_DIR}/fullchain.pem and ${CERT_DIR}/privkey.pem"
  exit 1
fi

docker run --rm -it \
  --network host \
  --privileged \
  -v /dev/bus/usb:/dev/bus/usb \
  -e ROS_MASTER_URI=http://127.0.0.1:11311 \
  -e ROS_IP=127.0.0.1 \
  -e WEBXR_CERT=/certs/fullchain.pem \
  -e WEBXR_KEY=/certs/privkey.pem \
  -e OCULUS_READER_BACKEND=webxr \
  -e TELEOP_USE_OCULUS_READER=0 \
  -e TELEOP_PRINT_HZ=0 \
  -e TELEOP_CONTROL_HZ=90 \
  -e TELEOP_DEADBAND_POS_M=0.003 \
  -e TELEOP_DEADBAND_ROT_RAD=0.05 \
  -e TELEOP_MAX_STEP_M=0.01 \
  -e TELEOP_ROT_GAIN=1.8 \
  -e TELEOP_CHECK_COLLISION=0 \
  -e WEBXR_PRINT_HZ=1 \
  -v "${CERT_DIR}":/certs:ro \
  -v "${WS_HOST}":/workspace/questVR_ws \
  "${IMAGE}" \
  bash -lc "source /opt/ros/noetic/setup.bash && source /workspace/questVR_ws/devel/setup.bash && if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then source /opt/conda/etc/profile.d/conda.sh && conda activate vt || true; fi && cd /workspace/questVR_ws && roslaunch oculus_reader teleop_single_piper_webxr.launch run_piper:=false run_teleop:=true"

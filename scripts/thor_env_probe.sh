#!/usr/bin/env bash
set -euo pipefail

echo "== OS =="
source /etc/os-release
echo "${PRETTY_NAME}"
uname -m

echo "== NVIDIA =="
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
dpkg -l | grep -E 'nvidia-cuda|tensorrt|nvinfer' || true

echo "== ROS =="
command -v ros2 >/dev/null 2>&1 && ros2 --version || true
echo "ROS_DISTRO=${ROS_DISTRO:-}"

echo "== Python backend =="
python3 -m sam_backend.env_probe

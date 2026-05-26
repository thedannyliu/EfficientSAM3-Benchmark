from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import subprocess
import sys


def main() -> None:
    print(json.dumps(probe(), indent=2))


def probe() -> dict:
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "commands": {name: shutil.which(name) for name in ["nvidia-smi", "ros2", "sbatch", "srun"]},
        "python_modules": {
            name: importlib.util.find_spec(name) is not None
            for name in ["torch", "torchvision", "cv2", "numpy", "rclpy", "sensor_msgs", "onnx", "tensorrt"]
        },
        "gpu": nvidia_smi(),
    }


def nvidia_smi() -> list[str]:
    if shutil.which("nvidia-smi") is None:
        return []
    cmd = ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    try:
        result = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=10)
    except subprocess.SubprocessError:
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


if __name__ == "__main__":
    main()

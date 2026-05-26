# EfficientSAM3 / SAM3 Jetson Thor Benchmark

Portable benchmark and ROS 2 integration scaffold for developing SAM3 or
EfficientSAM3 backend code on PACE, then validating final deployment on Jetson
Thor.

## Current Feasibility Notes

PACE is useful for backend development and GPU benchmarking, but it is not a
drop-in Thor replica. This workspace probe found:

- PACE login node: RHEL 9.6, x86_64, Python 3.13 conda env.
- PACE GPU queues include `gpu-l40s`, `gpu-h100`, `gpu-h200`, and others.
- Active login env has CPU-only PyTorch and no ROS 2.
- PACE modules include Python 3.12, CUDA 12.6/13.0, and PyTorch modules.

Recommended split:

- PACE: backend API, image/video benchmark loops, Slurm GPU runs.
- Thor: JetPack CUDA/TensorRT validation, ROS 2 Jazzy build, camera pipeline,
  final latency.

Use Python 3.12 for the real environment because ROS 2 Jazzy on Ubuntu 24.04
and upstream SAM3/EfficientSAM3 are aligned there better than with Python 3.13.

## Layout

```text
sam_backend/                  # model-independent backend API and benchmark CLI
ros_ws/src/sam_benchmark_ros/ # ROS 2 wrapper nodes
scripts/                      # PACE/Thor helper scripts
configs/                      # environment-specific run configs
tests/                        # smoke tests that do not need model weights
```

## Quick Local Smoke Test

This validates the benchmark loop without requiring CUDA or checkpoints:

```bash
python3 -m unittest
python3 -m sam_backend.benchmark --backend null --synthetic-frames 8 --prompt person
```

## PACE GPU Benchmark Shape

```bash
module load python/3.12.5 cuda/12.6.1
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .

sbatch scripts/pace_l40s_benchmark.sbatch
```

Install SAM3 or EfficientSAM3 in that same Python 3.12 environment before using
the real backends.

For EfficientSAM3 component profiling on the local demo videos:

```bash
bash scripts/setup_pace_venv.sh
sbatch scripts/pace_l40s_profile_sam3.sbatch
sbatch scripts/pace_l40s_profile_efficientsam3.sbatch
```

The default prompt is `monitor`. Per-frame component timings and parameter
counts are written to `results/`; mask overlay demo videos are written to
`overlays/`.

## Thor ROS Shape

Use `docs/thor_setup.md` for the current Thor procedure. It covers the unified
Thor environment, EfficientSAM3 performance profiling, overlay validation, and
ROS 2 setup.

On Thor, build the ROS package after the backend package is installed:

```bash
python3 -m pip install -e .
cd ros_ws
colcon build --symlink-install
source install/setup.bash
ros2 run sam_benchmark_ros sam_backend_node --ros-args -p backend:=sam3
```

The ROS wrapper publishes result JSON, optional overlay images, per-frame CSV
records, aggregate summaries, and overlay MP4 demos through separate recorder
nodes. Use `docs/thor_setup.md` for the current command sequence.

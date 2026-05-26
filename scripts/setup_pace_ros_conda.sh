#!/usr/bin/env bash
set -euo pipefail

module load mamba/1.4.9

mamba env create -f environment-ros-jazzy.yml
conda run -n esam3-ros-jazzy python -m pip install -e .
conda run -n esam3-ros-jazzy bash -lc 'cd ros_ws && colcon build --symlink-install'

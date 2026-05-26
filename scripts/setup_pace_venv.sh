#!/usr/bin/env bash
set -euo pipefail

module load python/3.12.5 cuda/12.6.1

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .

mkdir -p external
if [ ! -d external/efficientsam3/.git ]; then
  git clone https://github.com/SimonZeng7108/efficientsam3.git external/efficientsam3
fi
python -m pip install -e external/efficientsam3

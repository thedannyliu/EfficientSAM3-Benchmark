#!/usr/bin/env bash
#SBATCH --job-name=sav-salient-prepare
#SBATCH --partition=cpu-small
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load python/3.12.5
source .venv/bin/activate

mkdir -p logs data/sa-v data/manifests overlays/sav/review

bash scripts/prepare_sav_salient_subset.sh

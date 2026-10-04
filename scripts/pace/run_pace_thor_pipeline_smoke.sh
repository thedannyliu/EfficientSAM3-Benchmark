#!/usr/bin/env bash
set -euo pipefail

module load python/3.12.5 cuda/12.6.1
source .venv/bin/activate

mkdir -p results/thor_pipeline_smoke overlays/thor_pipeline_smoke

if [ "$#" -gt 0 ]; then
  videos=("$@")
else
  videos=(videos/test1.mov videos/test2.mov)
fi

for video in "${videos[@]}"; do
  stem="$(basename "${video}")"
  stem="${stem%.*}"
  python -m sam_backend.thor_pipeline_smoke \
    --backend "${SAM_BACKEND:-null}" \
    --device "${SAM_DEVICE:-cpu}" \
    --prompt "${SAM_PROMPT:-monitor}" \
    --video "${video}" \
    --max-frames "${SAM_MAX_FRAMES:-5}" \
    --output-jsonl "results/thor_pipeline_smoke/${SAM_BACKEND:-null}-${stem}.jsonl" \
    --overlay-output "overlays/thor_pipeline_smoke/${SAM_BACKEND:-null}-${stem}.mp4"
done

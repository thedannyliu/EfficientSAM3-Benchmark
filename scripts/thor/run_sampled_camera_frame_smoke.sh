#!/usr/bin/env bash
set -euo pipefail

module load python/3.12.5 cuda/12.6.1
source .venv/bin/activate

source_input="${1:-videos/test1.mov}"
run_id="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
max_frames="${SAM_MAX_FRAMES:-1}"
device="${SAM_DEVICE:-cuda}"
prompt="${SAM_PROMPT:-monitor}"
point="${SAM_POINT:-0.5,0.5}"

result_root="results/camera_sample/${run_id}"
overlay_root="overlays/camera_sample/${run_id}"
mkdir -p "${result_root}" "${overlay_root}"

run_efficientsam3() {
  local model_id="efficientsam3_es3p1_weak_image_weak_text"
  python -m sam_backend.thor_pipeline_smoke \
    --backend efficientsam3 \
    --device "${device}" \
    --checkpoint-path checkpoints/stage1_sam3p1/efficient_sam3p1_efficientvit_s_mobileclip_s0_ctx16.pt \
    --external-repo external/efficientsam3 \
    --backbone-type efficientvit \
    --model-name b0 \
    --text-encoder-type MobileCLIP-S0 \
    --text-encoder-context-length 16 \
    --text-encoder-pos-embed-table-size 16 \
    --prompt "${prompt}" \
    --video "${source_input}" \
    --frame-id camera \
    --max-frames "${max_frames}" \
    --output-jsonl "${result_root}/${model_id}/result.jsonl" \
    --sample-frame-dir "${result_root}/${model_id}/sampled_frames" \
    --overlay-output "${overlay_root}/${model_id}/overlay.mp4" \
    --overlay-frame-dir "${overlay_root}/${model_id}/frames"
}

run_efficient_sam2() {
  local model_id="efficient_sam2p1_hiera_tiny"
  python -m sam_backend.thor_pipeline_smoke \
    --backend efficient-sam2 \
    --device "${device}" \
    --checkpoint-path checkpoints/efficient-sam2/sam2.1_hiera_tiny.pt \
    --model-config configs/sam2.1/sam2.1_hiera_t.yaml \
    --external-repo external/Efficient-SAM2 \
    --point "${point}" \
    --point-normalized \
    --video "${source_input}" \
    --frame-id camera \
    --max-frames "${max_frames}" \
    --output-jsonl "${result_root}/${model_id}/result.jsonl" \
    --sample-frame-dir "${result_root}/${model_id}/sampled_frames" \
    --overlay-output "${overlay_root}/${model_id}/overlay.mp4" \
    --overlay-frame-dir "${overlay_root}/${model_id}/frames"
}

run_efficientsam3
run_efficient_sam2

cat <<EOF
Sampled camera-frame smoke complete.
source=${source_input}
run_id=${run_id}
results=${result_root}
overlays=${overlay_root}
prompt=${prompt}
point=${point}
EOF

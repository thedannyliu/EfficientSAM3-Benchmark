# Video Benchmark

Full SA-V is too large for this repo budget. The official download page lists
train archives at about 8 GiB each and val/test archives at about 16 GiB each.
For GT evaluation, download the official val/test archive, extract only a fixed
3-video subset for tracking, and remove the archive after extraction:

```bash
scripts/data/check_storage_budget.sh 300 data checkpoints external
bash scripts/data/download_sav_valtest_subset.sh val 3
```

SAM2-family native video profiler example:

```bash
bash scripts/data/download_sam2_family_checkpoints.sh
sam-profile-sav-video \
  --backend sam2 \
  --model-id sam2p1_hiera_tiny \
  --external-repo external/sam2 \
  --model-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --checkpoint-path checkpoints/sam2/sam2.1_hiera_tiny.pt \
  --manifest data/manifests/sav_val_fixed3.jsonl \
  --eval-mode both \
  --csv-output results/sav/video/manual/sam2p1_hiera_tiny/frames.csv \
  --summary-output results/sav/video/manual/sam2p1_hiera_tiny/summary.json \
  --pred-root results/sav/video/manual/sam2p1_hiera_tiny/pred \
  --overlay-root overlays/sav/video/manual/sam2p1_hiera_tiny
```

The SA-V profiler uses each manifest row's initial point prompt, propagates with
the backend's native video predictor, reports component timings and parameter
or weight sizes, computes IoU on the official val/test PNG annotations, writes
SA-V-style prediction PNGs, and writes per-video overlay MP4s for visual review.
SA-V val/test has segmentation masks but no semantic category labels, so this
benchmark uses point prompts for video tracking unless a separate text-label
source is explicitly documented.

The default SA-V target selection is official-GT-first, not saliency-first. It
selects the largest annotated object in the first GT frame of each seeded video,
which can still be a visually minor object if that is what SA-V annotated. For
POC overlays with more important-looking targets, rerun extraction with:

```bash
bash scripts/data/prepare_sav_salient_subset.sh
```

That writes an independent manifest at
`data/manifests/sav_val_salient_fixed3.jsonl` and a review contact sheet at
`overlays/sav/review/sav_val_salient_fixed3/contact_sheet.png`, leaving the
original `sav_val_fixed3` manifest untouched.

To manually add text prompts for a fixed SA-V manifest:

```bash
sam-sav-text-prompts init \
  --manifest data/manifests/sav_val_fixed3.jsonl \
  --review-dir overlays/sav/review/current_fixed3 \
  --output configs/datasets/sav_val_fixed3_text_prompts.json

# Fill text_prompt and instance_hint in configs/datasets/sav_val_fixed3_text_prompts.json.

sam-sav-text-prompts apply \
  --manifest data/manifests/sav_val_fixed3.jsonl \
  --prompts configs/datasets/sav_val_fixed3_text_prompts.json \
  --output data/manifests/sav_val_fixed3_text.jsonl
```

The text prompt must describe the selected official object ID shown in the
review overlay, not a more obvious unannotated object elsewhere in the frame.
When the frame has multiple same-class objects, `text_prompt` alone is
ambiguous; use `instance_hint` to document the selected GT object and report
top-1 localization separately from GT-assisted best-instance diagnostics.
The resulting `_text.jsonl` manifest can be passed to `sam-profile-yoloe-edgetam`
without `--text-prompt`; it will use the per-row prompt.

On PACE L40S:

```bash
SAV_TEXT_MANIFEST=data/manifests/sav_val_salient_fixed3_text.jsonl \
DOWNLOAD_YOLOE_EDGETAM_MOBILESAM=1 \
sbatch scripts/pace/pace_l40s_yoloe_edgetam_sav_text.sbatch
```

On PACE L40S:

```bash
DOWNLOAD_SAM2_FAMILY_CHECKPOINTS=1 sbatch scripts/pace/pace_l40s_sav_video_sam2_family.sbatch
```

The SA-V Slurm job writes `results/sav/video/<run>/sav_video_suite_summary.csv`
with one row per video-capable model.

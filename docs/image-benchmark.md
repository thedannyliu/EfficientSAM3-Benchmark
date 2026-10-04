# Image Benchmark

Create the fixed 10-image COCO val2017 manifest, then run text and point
prompt profiling with IoU metrics:

```bash
bash scripts/data/prepare_coco_fixed_subset.sh 10

sam-profile-coco \
  --backend null \
  --device cpu \
  --manifest data/manifests/coco_val2017_fixed10.jsonl \
  --prompt-mode both \
  --eval-mode both \
  --csv-output results/coco/smoke/null_fixed10/profile.csv \
  --summary-output results/coco/smoke/null_fixed10/summary.json
```

For real SAM3/EfficientSAM3 runs, replace `--backend null` with the selected
backend and checkpoint/model flags. The manifest chooses 10 random images with
a fixed seed, uses the largest non-crowd COCO object as the text prompt, uses
that object mask centroid as the point prompt, and reports best-mask and merged
IoU against that selected annotation.

Profiling CSVs include per-run CUDA memory, component latency hooks, component
parameter counts, and component weight bytes. Component columns are shared for
readability, but each backend only fills the components present in its native
model tree: SAM3/EfficientSAM3 use image/text/transformer/geometry/segmentation
and optional interactive prompt/mask components; SAM2-family models use image
encoder, prompt encoder, mask decoder, and video memory components.

Use `--eval-mode both` for metrics plus overlays, `--eval-mode gt` for metrics
only, `--eval-mode overlay` for visual inspection only, or `--eval-mode profile`
for profiling without GT/overlay work.

The selected image IDs, annotation IDs, category prompts, and points are recorded
in `data/coco/coco_val2017_fixed10_selection.json`. The visually reviewed
fixed text and point prompts are tracked in
`configs/datasets/coco_val2017_fixed10_prompts.json` for reproduction on
other devices. The prompt/eval protocol is documented in
[dataset protocol](benchmark_dataset_protocol.md).

Current fixed text prompts are:

```text
cow, train, motorcycle, bird, person, bed, bicycle, zebra, elephant, sink
```

SAM2-family image runs are point-prompt only in this benchmark:

```bash
sam-profile-coco \
  --backend sam2 \
  --external-repo external/sam2 \
  --model-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --checkpoint-path checkpoints/sam2/sam2.1_hiera_tiny.pt \
  --manifest data/manifests/coco_val2017_fixed10.jsonl \
  --prompt-mode point \
  --eval-mode both \
  --csv-output results/coco/single/sam2_tiny_fixed10/profile.csv
```

Use `--backend efficient-sam2 --external-repo external/Efficient-SAM2` for the
Efficient-SAM2 fork, or `--backend efficienttam --external-repo external/EfficientTAM`
with an EfficientTAM config such as `configs/efficienttam/efficienttam_ti.yaml`.
Use `--backend mobilesam --external-repo external/MobileSAM --checkpoint-path
checkpoints/mobilesam/mobile_sam.pt --mobile-sam-model-type vit_t` for the
MobileSAM point-prompt image baseline.

To run the default fixed COCO model matrix:

```bash
bash scripts/data/download_sam3_checkpoint.sh
sam-run-coco-suite \
  --manifest data/manifests/coco_val2017_fixed10.jsonl \
  --device cuda \
  --skip-missing \
  --output-dir results/coco/suite/manual \
  --overlay-dir overlays/coco/suite/manual
```

Suite-level comparison is written to
`results/coco/suite/<run>/coco_suite_component_summary.csv`, with one row per
model/prompt mode.

On PACE L40S:

```bash
sbatch scripts/pace/pace_l40s_coco_suite.sbatch
```

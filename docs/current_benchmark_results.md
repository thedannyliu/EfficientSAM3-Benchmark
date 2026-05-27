# Current Benchmark Results

Last updated: 2026-05-27.

This file records the current completed benchmark outputs. Runtime artifacts
remain under ignored `results/` and `overlays/`; this tracked file is the
readable index.

## COCO Fixed10 Image Suite

Run:

```text
9206977 sam-coco-suite
```

Source files:

```text
results/coco/suite/9206977/coco_suite_summary.csv
results/coco/suite/9206977/coco_suite_component_summary.csv
overlays/coco/suite/9206977/
data/manifests/coco_val2017_fixed10.jsonl
data/manifests/coco_val2017_fixed10_selection.json
```

Fixed prompts:

```text
cow, train, motorcycle, bird, person, bed, bicycle, zebra, elephant, sink
```

Summary:

| Model | Prompt | ms/img | FPS | mIoU best | mIoU merged | CUDA peak MB | Params M | Weights MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sam3` | point | 943.68 | 1.06 | 0.881 | 0.882 | 5340.84 | 860.06 | 3280.85 |
| `sam3` | text | 478.22 | 2.09 | 0.872 | 0.689 | 5340.84 | 860.06 | 3280.85 |
| `es3p1_weak_image_weak_text` | point | 234.16 | 4.27 | 0.172 | 0.173 | 1310.05 | 112.89 | 430.63 |
| `es3p1_weak_image_weak_text` | text | 221.81 | 4.51 | 0.000 | 0.000 | 1310.05 | 112.89 | 430.63 |
| `es3p1_strong_image_weak_text` | point | 118.92 | 8.41 | 0.135 | 0.143 | 1397.50 | 127.44 | 486.16 |
| `es3p1_strong_image_weak_text` | text | 126.31 | 7.92 | 0.086 | 0.086 | 1397.50 | 127.44 | 486.16 |
| `es3_weak_image_strong_available_text` | point | 100.29 | 9.97 | 0.383 | 0.383 | 1431.74 | 133.87 | 510.69 |
| `es3_weak_image_strong_available_text` | text | 102.07 | 9.80 | 0.000 | 0.000 | 1431.74 | 133.87 | 510.69 |
| `es3_strong_image_strong_available_text` | point | 112.79 | 8.87 | 0.549 | 0.550 | 1517.74 | 148.43 | 566.22 |
| `es3_strong_image_strong_available_text` | text | 94.22 | 10.61 | 0.000 | 0.000 | 1517.74 | 148.43 | 566.22 |
| `sam2p1_hiera_tiny` | point | 145.69 | 6.86 | 0.443 | 0.443 | 597.84 | 38.96 | 148.63 |
| `efficient_sam2p1_hiera_tiny` | point | 60.95 | 16.41 | 0.443 | 0.443 | 597.84 | 38.96 | 148.63 |
| `efficienttam_ti` | point | 5671.48 | 0.18 | 0.420 | 0.420 | 367.04 | 17.87 | 68.15 |
| `efficienttam_s` | point | 5453.26 | 0.18 | 0.548 | 0.548 | 474.43 | 34.06 | 129.91 |

Notes:

- `mobilesam_vit_t` was skipped in this suite because its checkpoint had not
  been downloaded at submission time. Replacement job `9215147` is pending.
- EfficientSAM3 text rows with zero IoU indicate prompt localization failures
  under the current checkpoint/config combination, not missing GT.
- The complete component columns are in `coco_suite_component_summary.csv`,
  including image encoder, text encoder, prompt encoder, mask decoder,
  grounding, detector, memory, parameter, and weight-size fields.

## SA-V Fixed3 Video Tracking

Partial run:

```text
9205043 sav-sam2-family
```

Source files:

```text
results/sav/video/9205043/sav_video_suite_summary.csv
overlays/sav/video/9205043/sam2p1_hiera_tiny/
```

Summary:

| Model | Videos | Frames | GT frames | mean IoU | FPS | CUDA peak MB | Params M | Weights MB | Overlays |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sam2p1_hiera_tiny` | 3 | 291 | 75 | 0.680 | 21.05 | 6395.76 | 38.96 | 148.63 | 3 |

Notes:

- This run completed SAM2 outputs and overlays, then failed while starting the
  Efficient-SAM2 fork because `enable_MeP_info` was missing. The adapter has
  since been patched to initialize that field.
- Replacement job `9210795` is pending and should produce SAM2,
  Efficient-SAM2, EfficientTAM-Ti, and EfficientTAM-S rows.

## Pending Runs

Checked on 2026-05-27 with `squeue`; all listed jobs use `QOS=embers`.

| Job | Purpose | Status |
| --- | --- | --- |
| `9210795` | SA-V SAM2-family video suite | Pending on `embers` priority |
| `9210880` | sampled camera-frame smoke on PACE | Pending on `embers` priority |
| `9215147` | MobileSAM COCO fixed10 point baseline | Pending on `embers` priority |
| `9215800` | YOLOE-26M-seg + EdgeTAM recorded-video POC | Pending on `embers` priority |
| `9215801` | YOLOE-26M-seg + EdgeTAM SA-V text manifest | Pending on `afterok:9215800` |

Local validation:

```text
python -m unittest
python -m sam_backend.coco_suite --manifest data/manifests/coco_val2017_fixed10.jsonl --models sam3 efficient_sam2p1_hiera_tiny mobilesam_vit_t --dry-run --output-dir results/local_smoke/coco_suite_dry_run --overlay-dir overlays/local_smoke/coco_suite_dry_run
scripts/check_storage_budget.sh 300 data checkpoints external
bash scripts/check_pace_qos.sh
```

All four checks passed. The storage check reported 13.78 GiB total for
`data`, `checkpoints`, and `external`.

# Status and Plan

Last updated: 2026-05-26.

## Current Status

- Repo initialized and pushed to `origin/main`.
- `AGENTS.md`, `.venv/`, `external/`, `videos/`, `results/`, `overlays/`, `logs/`, and checkpoints are ignored.
- PACE `.venv` is built with Python 3.12.5, CUDA PyTorch 2.10.0, torchvision 0.25.0, and torchaudio 2.10.0.
- EfficientSAM3 is installed editable from `external/efficientsam3`.
- Hugging Face auth is available on PACE as `danny010324`.
- Local test videos are `videos/test1.mov` and `videos/test2.mov`.
- Default prompt is `monitor`.

## Implemented

- Shared backend API in `sam_backend/`.
- Video profiling CLI: `sam_backend/profile_video.py`.
- EfficientSAM3 variant runner: `sam_backend/variant_runner.py`.
- ROS-free Thor pipeline smoke runner: `sam_backend/thor_pipeline_smoke.py`.
- CSV profiling fields include total latency, image encoder time, text encoder time, grounding time, CUDA memory, mask count, scores, and parameter counts.
- Overlay demo video export writes masks/boxes to MP4.
- PACE setup and Slurm scripts:
  - `scripts/setup_pace_venv.sh`
  - `scripts/run_pace_thor_pipeline_smoke.sh`
  - `scripts/pace_l40s_profile_sam3.sbatch`
  - `scripts/pace_l40s_profile_efficientsam3.sbatch`
- English Thor benchmark and ROS guide: `docs/thor_setup.md`.
- Legacy pointer for the old Traditional Chinese path: `docs/thor_setup_zh_tw.md`.

PACE does not provide system ROS 2 on the login node. A separate experimental
conda/robostack environment is described by `environment-ros-jazzy.yml`; the
ROS-free smoke runner is the reliable PACE check for video-to-backend plumbing.

## PACE Benchmark Jobs

Both use `gpu-l40s`, account `gts-agarg35-ideas_l40s`, and QOS `embers`.

| Job ID | Job | Status |
| --- | --- | --- |
| `9183683` | EfficientSAM3 profiling | Failed after partial output; MobileCLIP-S1 variants needed 77-position text table |
| `9183707` | SAM3 profiling | Completed |

Monitor with:

```bash
squeue -j 9183683,9183707
tail -f logs/esam3-prof-9183683.out
tail -f logs/sam3-prof-9183707.out
```

SAM3 completed 30 frames per video on L40S:

| Video | Mean total latency |
| --- | --- |
| `videos/test1.mov` | 234.63 ms |
| `videos/test2.mov` | 168.33 ms |

EfficientSAM3 partial outputs from job `9183683`:

| Variant | Video | Mean total latency |
| --- | --- | --- |
| `es3p1_weak_image_weak_text` | `videos/test1.mov` | 182.86 ms |
| `es3p1_weak_image_weak_text` | `videos/test2.mov` | 68.54 ms |
| `es3p1_strong_image_weak_text` | `videos/test1.mov` | 73.56 ms |
| `es3p1_strong_image_weak_text` | `videos/test2.mov` | 73.26 ms |

## Expected Outputs

EfficientSAM3:

```text
results/efficientsam3_variants/<jobid>/*.csv
results/efficientsam3_variants/<jobid>/efficientsam3_variant_summary.csv
overlays/efficientsam3_variants/<jobid>/*.mp4
```

SAM3:

```text
results/sam3-<jobid>-test1.csv
results/sam3-<jobid>-test2.csv
overlays/sam3-<jobid>-test1.mp4
overlays/sam3-<jobid>-test2.mp4
```

## Plan

1. Wait for L40S jobs to run.
2. Inspect logs for checkpoint/model/API failures.
3. If successful, compare CSV summaries and overlay videos.
4. If model loading fails, patch backend loader against the exact upstream API error.
5. After PACE backend results are stable, copy the same backend to Thor.
6. On Thor, validate JetPack CUDA/TensorRT, ROS 2 Jazzy nodes, camera input, and final latency.

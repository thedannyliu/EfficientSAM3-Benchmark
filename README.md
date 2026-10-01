# SAM-family benchmark toolkit

Evaluate SAM3, EfficientSAM3, SAM2-family and MobileSAM backends with shared
image/video manifests, prompt protocols, quality metrics and latency reports.
Includes ROS 2 camera integration and PACE/Jetson Thor operating recipes.

## Run a CPU smoke

Use Python 3.12. The null backend needs no model weights or CUDA:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -m unittest discover -s tests
sam-benchmark --backend null --synthetic-frames 8 --prompt person
```

This checks the benchmark plumbing. For a real model, install its upstream
repository, provide a compatible checkpoint, and select its backend explicitly.
On Thor, install JetPack-compatible PyTorch first and use `requirements-thor.txt`.

## Choose a benchmark

| Task | Guide / entry point |
| --- | --- |
| Fixed-image COCO profiling | [Image benchmark](docs/image-benchmark.md), `sam-profile-coco` |
| Prompted SA-V video tracking | [Video benchmark](docs/video-benchmark.md), `sam-profile-sav-video` |
| Repeated model matrix | `sam-run-coco-suite`, `sam-run-saco-stream-suite` |
| Live camera | [Thor ROS guide](docs/thor_ros_camera_benchmark.md) |
| Offline Thor measurements | [Thor offline guide](docs/thor_offline_benchmark.md) |

Use the [dataset protocol](docs/benchmark_dataset_protocol.md) before comparing
models. Fixed small subsets are smokes, not full-dataset scores. SAM2-family
image backends use geometric prompts; SA-V does not supply semantic labels,
so text prompts need separately documented annotation.

## Layout

| Path | Responsibility |
| --- | --- |
| `sam_backend/` | Backend API, benchmark loops, metrics and public `sam-*` CLIs |
| `configs/`, `data/` | Run settings, prompt definitions and small manifests |
| `scripts/data/` | Dataset and checkpoint preparation |
| `scripts/setup/` | Environment and upstream installation |
| `scripts/pace/` | Slurm jobs and server-side probes |
| `scripts/thor/` | Device benchmark and camera workflows |
| `ros_ws/` | ROS nodes and launch files |
| `tests/` | Backend, protocol, metric and reporting regression checks |

[Script guide](scripts/README.md) · [Documentation index](docs/README.md)

## Reporting results

Report checkpoint/revision, selected images/videos, prompt and mask-selection
policy, hardware, precision, warmup and timing scope. Keep model latency,
end-to-end throughput and source-frame age separate. PACE measurements do not
establish Thor performance. Store raw traces and overlays under ignored
`results/` and `overlays/`; commit only small attributable summaries.

Upstream model implementations and checkpoints retain their own licenses.

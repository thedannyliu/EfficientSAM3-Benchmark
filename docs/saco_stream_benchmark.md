# SA-Co/VEval Stream Benchmark

This benchmark evaluates text/point-initialized video stream segmentation on a
fixed SA-Co/VEval-SAV subset and writes overlay MP4s for visual review.

Large assets should live under scratch:

```bash
export SAM_BENCH_SCRATCH=/storage/scratch1/9/eliu354/efficientsam3-benchmark
```

Download model assets and source repos:

```bash
bash scripts/download_saco_stream_assets.sh
```

Prepare a fixed 20-video SA-Co/VEval-SAV manifest:

```bash
sam-prepare-saco-veval-sav-subset \
  --annotation "$SAM_BENCH_SCRATCH/data/annotation/saco_veval_sav_val.json" \
  --media-root "$SAM_BENCH_SCRATCH/data/media/saco_sav/JPEGImages_24fps" \
  --count 20 \
  --output data/manifests/saco_veval_sav_fixed20.jsonl
```

Run the suite on PACE or Thor:

```bash
sam-run-saco-stream-suite \
  --manifest data/manifests/saco_veval_sav_fixed20.jsonl \
  --gt-annotation-file "$SAM_BENCH_SCRATCH/data/annotation/saco_veval_sav_val.json" \
  --scratch-root "$SAM_BENCH_SCRATCH" \
  --max-frames 120 \
  --output-dir results/saco_stream/fixed20 \
  --overlay-dir overlays/saco_stream/fixed20 \
  --skip-missing
```

Each successful model writes:

```text
results/saco_stream/fixed20/<model_id>/frames.csv
results/saco_stream/fixed20/<model_id>/frames_summary.csv
results/saco_stream/fixed20/<model_id>/saco_veval_preds.json
results/saco_stream/fixed20/<model_id>/saco_veval_eval_res.json
overlays/saco_stream/fixed20/<model_id>/<source_id>/overlay.mp4
```

For Thor recorded ROS stream timing, publish the fixed video at 30 FPS with
`video_stream_node`, run the selected backend, and use:

```bash
bash scripts/run_thor_ros_saco_stream_suite.sh data/manifests/saco_veval_sav_fixed20.jsonl <model_id>
```

This helper prints the recorder commands for `/sam/result_json` and
`/sam/overlay`; run them while the matching backend is active.

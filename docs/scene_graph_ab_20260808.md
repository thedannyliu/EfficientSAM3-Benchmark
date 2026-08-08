# Scene Graph A/B: Original SAM3.1 vs General Instinct SAM

Date: 2026-08-07/08

Platform: second NVIDIA Jetson AGX Thor (`magni`)

Status: completed development-branch evaluation; not approved for stable or production use

## Decision Summary

The delivered General Instinct (GI) runtime is not a drop-in acceleration for
the current Scene Graph workload. With the same Lifestyle Lab bag segment and
39 category prompts, GI completed only 66% as many frames per second of camera
source time as Original SAM3.1. Its median detector HTTP latency was 3.962 s,
versus 2.579 s for Original grounding. GI also produced fewer 3D detection
frames and a smaller final graph.

The result is not caused by exhausted Thor memory. At least 80.9 GiB of Linux
unified memory remained available during the GI run. GI was slower while using
less average GPU compute utilization, which points to workload mismatch,
serialization/synchronization, and lower kernel utilization rather than an
out-of-memory or thermal limit.

Do not replace the stable detector with this integration. The next useful
experiment is a small-prompt, long-lived tracking test that matches the GI
runtime's intended operating mode. Only optimize the container bridge if that
test first demonstrates a model-level advantage.

## Compared Systems

| Condition | Detector | Deployment | Threshold |
| --- | --- | --- | ---: |
| Original | `facebook/sam3.1`, snapshot `daa63191845a41281374e725f4c9e51c7a824460` | Existing Ether/Scene Graph container | 0.8 |
| GI | `instinctsam:thor-r39`, Hiera-L in768 TRT-FP16 plus detect in1152 TRT-FP16 | Ether/Scene Graph client plus GI HTTP runtime container | 0.5 |

The GI image archive SHA-256 is
`30b40a025a76e8a8e911a3c57320637260e9fc78b54fcc4b90b73c7982bb7e75`.
The loaded image ID is
`sha256:8fd009341104f6944441d4e6fccbcd9af2598fa03812ee7ae64488ac28906ecd`.
The evaluation-only runtime overlay SHA-256 is
`1dc8ef34fe61142d6825dcb73c35310bd6b30aec3123f2118cf7c6b6467fb11d`.

GI threshold 0.8 returned no masks in the preserved sensitivity run, so 0.5
uses the delivery's intended operating point. Confidence values are not
assumed to be calibrated between systems.

## Protocol

- Input: Lifestyle Lab D435 bag beginning at offset zero.
- Playback: 1x, approximately 180 wall-clock seconds.
- Camera source duration: 173.96 s Original and 173.99 s GI.
- Prompts: the same 39 Scene Graph categories in the same order.
- Sampling: 35 JPEG frames, every 5 s by camera header timestamp.
- Alignment: both mask dumps use the exact 35 Original-sampled JPEG files and
  source timestamps. Model completion time never determines the sample.
- Isolation: Original and GI were run separately, never concurrently.
- Architecture: Original uses the stable one-container deployment. GI uses the
  intended two-container deployment, so its measured bridge overhead is real.
- Telemetry: resource samples cover Linux `MemAvailable`, tegrastats RAM and
  power, Docker working set and CPU, NVIDIA process memory, GPU utilization,
  GPU power, and temperature.
- Quality: COCO-10 polygon annotations are the true ground truth. Lifestyle bag
  mask IoU uses Original as a teacher and is an agreement/regression metric,
  not accuracy against ground truth.

The Original and GI recorder source timestamps start at the same value,
`1781705462949240832` ns. Their final timestamps differ by only 33.35 ms because
the bag player stopped after the same wall-time window.

## Speed Results

| Metric | Original | GI | Interpretation |
| --- | ---: | ---: | --- |
| Completed full frames | 56 | 37 | GI completed 66.1% of Original's count |
| Processed frames / source-second | 0.322 | 0.213 | GI throughput was 34.0% lower |
| Grounding / HTTP mean | 2,551 ms | 3,888 ms | GI was 52.4% slower at the detector boundary |
| Grounding / HTTP p50 | 2,579 ms | 3,962 ms | GI was 53.6% slower |
| Grounding / HTTP p95 | 2,647 ms | 4,374 ms | GI had a substantially longer tail |
| Full-frame mean | 2,735 ms | 4,045 ms | GI was 47.9% slower end to end |
| Full-frame p50 | 2,633 ms | 4,009 ms | GI was 52.3% slower |
| Full-frame p95 | 3,276 ms | 4,642 ms | GI was 41.7% slower |

GI latency decomposes as follows over 38 detector calls:

| GI stage | Mean | p50 | p95 |
| --- | ---: | ---: | ---: |
| Runtime detect | 3,160 ms | 3,196 ms | 3,596 ms |
| Runtime process | 3,279 ms | 3,328 ms | 3,752 ms |
| Client HTTP total | 3,888 ms | 3,962 ms | 4,374 ms |
| Scene Graph full frame | 4,045 ms | 4,009 ms | 4,642 ms |

On mean values, GI's model detect path was already 23.9% slower than Original
grounding. Runtime processing added about 118 ms, and the remaining HTTP/JPEG/
NPZ boundary added about 609 ms before Scene Graph post-processing. Eliminating
HTTP alone therefore cannot make this 39-prompt path faster than Original.

## Quality Results

### COCO-10 polygon ground truth

Each system received the same ten images and one selected object text prompt
per image. `mean_total_ms` includes the same benchmark-client boundary for each
backend; for GI it also includes JPEG and HTTP.

| Metric | Original | GI | GI change |
| --- | ---: | ---: | ---: |
| Mean latency | 362.5 ms | 509.6 ms | 40.6% slower |
| Best-mask mIoU | 0.8740 | 0.7881 | -0.0859 |
| Merged-mask mIoU | 0.6906 | 0.6170 | -0.0736 |
| AP | 0.7766 | 0.7208 | -0.0558 |
| AP50 | 1.0000 | 0.9010 | -0.0990 |
| AP75 | 1.0000 | 0.9010 | -0.0990 |

This small fixed set is appropriate for integration regression testing, not a
claim about general model accuracy.

### Source-aligned Lifestyle bag agreement

| Metric | Result |
| --- | ---: |
| Fixed frames | 35 |
| Original non-empty frames | 16 |
| GI non-empty frames | 15 |
| Frames where both are non-empty | 13 |
| Original masks | 41 |
| GI masks | 41 |
| Original-to-GI instance mIoU | 0.2701 |
| Original instance recall at IoU 0.5 | 0.2927 |
| GI-to-Original instance mIoU | 0.2645 |
| Per-label union mIoU | 0.3060 |

For each Original instance, agreement selects the highest-IoU GI mask with the
same prompt label; a missing label scores zero. This explains why the aggregate
can be low even though isolated frames occasionally agree strongly. It must not
be labeled ground-truth mIoU.

## Scene Graph Output

| Metric | Original | GI |
| --- | ---: | ---: |
| Detection messages | 70 | 42 |
| Camera timestamps with 3D detections | 26 | 17 |
| Final graph nodes | 9 | 3 |
| Final graph edges | 0 | 1 |

Original final categories were two books, two drawers, two chairs, one kitchen
towel, one table, and one cup. GI produced one book, one phone, and one plush
toy. The category-multiset overlap was one node: precision 0.333, recall 0.111,
and F1 0.167 when Original is treated only as the comparison reference.

The smaller GI graph is consistent with its lower processing rate and fewer
3D-detection timestamps. It is not sufficient evidence by itself to determine
which graph is semantically correct.

## Hardware Utilization and Co-resident Headroom

| Metric | Original | GI |
| --- | ---: | ---: |
| Mean GPU utilization | 70.3% | 58.9% |
| GPU utilization p95 | 97.0% | 92.0% |
| Mean GPU power | 19.2 W | 16.3 W |
| Maximum GPU power | 31.9 W | 24.4 W |
| Mean system power | 58.8 W | 50.4 W |
| Maximum system power | 124.9 W | 90.5 W |
| Mean GPU temperature | 49.4 C | 47.8 C |
| Maximum GPU temperature | 56.0 C | 50.0 C |
| Mean Linux unified memory available | 90.0 GiB | 81.3 GiB |
| Minimum Linux unified memory available | 87.2 GiB | 80.9 GiB |
| Mean Docker working set | 8.09 GiB | 12.27 GiB |
| Maximum Docker working set | 10.81 GiB | 12.51 GiB |
| Mean NVIDIA process memory | 10.17 GiB | 8.79 GiB |
| Maximum NVIDIA process memory | 20.92 GiB | 9.13 GiB |
| Mean summed container CPU | 366.6% | 395.9% |

Thor has unified CPU/GPU memory. Linux available memory, Docker working set,
CUDA allocator values, and NVIDIA process memory describe different views and
must not be added together. The GI two-container deployment reduced available
host memory by about 8.75 GiB on average and used about 4.18 GiB more Docker
working set than Original, while still leaving at least 80.9 GiB available.

Memory and temperature therefore leave room for additional models in this
isolated run. Capacity planning must still repeat this telemetry with all models
co-resident: both pipelines already show burst GPU utilization above 90%, and
the current GI path is latency/compute-utilization limited before it is
memory-capacity limited.

## Why GI Is Slower Here

1. The GI runtime is designed around a small prompt set followed by tracking,
   while Scene Graph submits 39 object categories. Its reported
   `max_objects=24` also does not naturally cover this category set in one
   operating window.
2. Even before IPC, GI runtime detect averaged 3.160 s versus 2.551 s for the
   Original batched grounding path.
3. The two-container API exchanges JPEG input and compressed NumPy mask output,
   adding serialization, copies, synchronization, and approximately 609 ms of
   mean client-visible overhead beyond GI runtime processing.
4. GI's lower average GPU utilization despite higher latency suggests bubbles
   between CPU preprocessing, per-object work, serialization, and GPU kernels.
5. The thresholds differ because GI at 0.8 produced no masks. Lowering GI to
   0.5 restored detections but did not align confidence calibration or mask
   selection with Original.

## Artifacts and Reproduction

The complete run is stored outside Git at:

```text
/mnt/nas/danny/thor-scene-graph/run-artifacts/scene-graph-ab-20260808/
```

Important subdirectories:

- `original/` and `gi/`: recorder output, full 3D detection JSONL, final graph,
  logs, resource telemetry, and SHA-256 manifests.
- `quality/original-coco10/` and `quality/gi-coco10/`: true-GT profiles,
  summaries, and overlays.
- `quality/original-bag35/` and `quality/gi-bag35/`: exact source-aligned packed
  masks and timing profiles.
- `report/`: summary JSON, mask-agreement CSV, English generated report, and
  four PNG figures.
- `gi-threshold-0p8/`: preserved threshold sensitivity run.
- `gi-preflight-failed-import/`, `gi-preflight-failed-scipy/`, and
  `quality/gi-coco10-failed-geometry/`: failed preflights kept for provenance;
  they are excluded from formal metrics.

Regenerate the report on Thor with:

```bash
python3 -m sam_backend.scene_graph_ab_report \
  --root /mnt/nas/danny/thor-scene-graph/run-artifacts/scene-graph-ab-20260808 \
  --output-dir /mnt/nas/danny/thor-scene-graph/run-artifacts/scene-graph-ab-20260808/report
```

Benchmark repository commits through `f8f017b` provide resource sampling,
source-time recording, the GI HTTP adapter, and fixed-mask dumps. The isolated
Scene Graph integration uses commits `64e7e39` and `3b4a3a9`. The stable
baseline remains tagged `thor-d435-baseline-2026-08-07`; none of the GI bridge
changes were applied to it.

## Limitations and Next Gate

- COCO-10 is deliberately small and should be expanded before a model-quality
  decision.
- Lifestyle bag agreement has no human polygon ground truth.
- Original and GI thresholds are operational settings, not calibrated scores.
- The two deployments intentionally differ in container count.
- The disposable candidate Ether container required `scipy==1.15.3`; the
  candidate image/build instructions must record that dependency before the
  integration can be reproduced from a clean image.
- The supplied GI license permits evaluation/research use, not production use.

The next acceptance gate is a source-aligned benchmark with a small prompt set,
one initialization, and many tracking frames. Record initialization latency,
steady-state tracking latency, drift/IoU, dropped frames, GPU duty cycle, and
the same memory/power telemetry. If GI detect remains slower before HTTP in that
intended mode, stop integration rather than optimizing transport.

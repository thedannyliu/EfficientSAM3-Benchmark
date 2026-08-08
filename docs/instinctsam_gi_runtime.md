# General Instinct Runtime Integration on Jetson Thor

## Scope

This document records the isolated evaluation of General Instinct's
`instinctsam:thor-r39` runtime on the second Jetson AGX Thor. The goal is to
measure the delivered TensorRT runtime before using it as a candidate detector
for the Scene Graph pipeline.

The existing SAM3/SAM3.1 deployment remains the reproducible baseline. Runtime
integration work must stay on development branches until the compatibility,
mask-output, accuracy, and ROS interface gates below pass.

## Branch Isolation

- Benchmark repository: `dev/instinctsam-gi-runtime`
- Scene Graph repository: `dev/instinctsam-integration`
- Scene Graph baseline tag: `thor-d435-baseline-2026-08-07`
- Scene Graph baseline commit: `e9c7a14a24e3cfd6c7bbc9921fb3e7dec7a5c606`

Do not merge candidate-specific code, configuration, or generated results into
the stable branches during this evaluation.

## External Runtime Artifact

The delivery is downloaded from the public Google Drive folder supplied by the
runtime author:

```text
https://drive.google.com/drive/folders/1DyLOdRXWD_GT4s5jKT6AGkBbO5TKjS5c
```

The source folder contains the image archive, licenses, installation notes,
Compose configuration, and launch scripts. The archive and its accompanying
files are stored outside Git at:

```text
/mnt/nas/danny/thor-scene-graph/candidates/
  instinctsam-drive-1DyLOdRXWD_GT4s5jKT6AGkBbO5TKjS5c/
```

Expected image archive:

```text
instinctsam-thor-r39.tar.gz
SHA-256: 30b40a025a76e8a8e911a3c57320637260e9fc78b54fcc4b90b73c7982bb7e75
Docker tag after load: instinctsam:thor-r39
```

Never commit the archive, Docker image layers, checkpoints, TensorRT engines,
rosbags, overlays, or benchmark outputs. Never archive credentials or login
state with the runtime.

The supplied `run.sh` force-removes a container named `instinctsam`. Do not run
it as part of the integration workflow. Use an explicitly named evaluation
container after resolving the exact target with `docker inspect`.

## License Gate

Two supplied licenses apply:

- General Instinct grants use, reproduction, and modification for research,
  evaluation, testing, teaching, and other non-production uses. Commercial or
  production use requires a separate written license.
- Meta's SAM License continues to apply to bundled SAM materials and derived
  weights. Preserve the license and attribution files and do not reverse
  engineer the SAM materials.

This work is an evaluation only. Passing the technical gates below does not
authorize production deployment.

## Thor Compatibility Gate

Current Thor environment:

```text
Architecture: aarch64
L4T: R38.4
Driver: 580.00
CUDA reported by nvidia-smi: 13.0
Docker: 28.2.2
NVIDIA container runtime: available
```

The delivery author verified the image on R39 and states that R38.4 should be
compatible because the shipped CUDA components use CUDA 13.0. Treat R38.4 as
unverified until all of these checks pass on this Thor:

1. The supplied archive SHA-256 matches.
2. Docker loads an `arm64` image with the expected tag.
3. A temporary container can access the Thor GPU.
4. Python, PyTorch, CUDA, and TensorRT imports succeed.
5. The supplied TensorRT engines load, or rebuild successfully once into the
   dedicated evaluation cache volume.
6. One fixed image produces masks without a container restart or CUDA error.

Record the image ID, RepoDigest if present, host L4T/driver/CUDA versions, and
whether engines loaded or rebuilt.

## Integration Boundary

The benchmark repository uses this model-independent contract:

```text
RGB image + text prompt(s)
  -> backend.predict(...)
  -> masks, boxes, scores, latency metadata
```

The Scene Graph detector needs the following result for every configured text
prompt:

```text
combined boolean mask
instance boolean masks
instance confidence scores
```

The GI adapter must stop at this boundary. Keep the existing depth
deprojection, segmented point-cloud creation, 3D bounding boxes, overlay,
`/scene_graph/detections_3d` message format, and graph builder unchanged.

Prefer the runtime's supported API if it exposes raw masks. If the delivered
web application exposes only visualization output, add the smallest permitted
raw-mask endpoint around the GI application code; do not modify or decompile
Meta SAM internals.

## Fixed Evaluation Inputs

The fixed COCO subset and its embedded polygon ground truth are stored at:

```text
/mnt/nas/danny/thor-scene-graph/benchmark-inputs/coco-val2017-fixed10/
```

It contains ten images, the fixed selection and prompt files, the JSONL
manifest, and `SHA256SUMS`. This subset provides real GT IoU measurements.

The domain-specific input is the existing Lifestyle Lab bag:

```text
/home/magni/ether-onboard/mercury_20260617_141052/
  mercury_20260617_141052.mcap
```

It is approximately 28.8 GiB and contains 32,275 messages on
`/d435/color/image_raw_jpeg`. Select frames by fixed bag timestamp and save a
lightweight selection manifest. Do not commit or duplicate the full bag.

## Benchmark Protocol

Use identical images, text prompts, input resolution, confidence threshold,
warm-up count, and measured frame order for the original SAM3/SAM3.1 backend
and the GI runtime.

Report model/runtime measurements separately:

- cold startup and model/engine load time;
- warm detector-refresh latency and FPS;
- warm tracking-frame latency and FPS;
- prompt count and detected instance count;
- peak GPU/device memory when measurable;
- COCO best-mask and merged-mask IoU against GT;
- per-frame and mean mask IoU against original SAM3 on fixed Lifestyle Lab
  frames;
- no-detection rate and instance-count differences.

Teacher agreement against original SAM3 is a regression metric, not ground
truth accuracy. Label it `teacher mask agreement`, not `mIoU accuracy`.

The completed baseline display run logged 315 SAM3 grounding intervals for 39
configured categories:

```text
mean: 2.986295 s
min:  2.485458 s
max:  3.511565 s
rate derived from mean: 0.3349 FPS
```

That number includes a different workload from the delivery author's reported
tracking FPS. Do not compare them directly. Compare detector-refresh,
tracking-frame, and end-to-end Scene Graph timings independently.

## Result Storage

Generated results belong under ignored local directories and should be copied
to a dated NAS run directory after validation:

```text
results/instinctsam_gi_runtime/<run-id>/
overlays/instinctsam_gi_runtime/<run-id>/
/mnt/nas/danny/thor-scene-graph/run-artifacts/instinctsam-<run-id>/
```

Store commands, environment metadata, CSV/JSON summaries, logs, and checksums.
Keep large overlays and runtime artifacts out of Git.

## Current Status

- Baseline Scene Graph run is archived and its stable checkout remains unchanged.
- Benchmark and Scene Graph development branches are created and pushed.
- Runtime archive, fixed COCO inputs, and all formal results are on NAS with checksums.
- Both supplied TensorRT engines load and execute on Thor R38.4 / SM110.
- The evaluation-only overlay exposes source-aligned JPEG and packed-mask APIs.
- The Scene Graph bridge preserves the existing 3D detection message contract.
- Source-aligned Original/GI runs, COCO-10 GT evaluation, bag teacher agreement,
  and resource profiling are complete.

The measured GI deployment is slower and produces fewer Scene Graph detections
for the current 39-prompt workload. Detailed protocol, results, limitations,
artifact locations, and the next acceptance gate are recorded in
`docs/scene_graph_ab_20260808.md`.

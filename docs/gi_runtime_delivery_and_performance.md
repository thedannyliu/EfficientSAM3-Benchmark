# GI Thor Runtime Delivery, Integration, Weights, and Performance

Date: 2026-08-11

Status: development-branch evaluation; not approved for stable or production
use

This document answers three related questions:

1. What General Instinct (GI) delivered for Jetson Thor and how Scene Graph
   connects to it.
2. Which InstinctSAM weights are publicly available, and how those component
   weights differ from the delivered Thor runtime.
3. Why the original stateless GI detector was slower than Original SAM3.1 even
   though the delivery uses TensorRT, and where the later speedup comes from.

Detailed experiment protocols and raw result locations remain in
`docs/instinctsam_gi_runtime.md`, `docs/scene_graph_ab_20260808.md`, and
`docs/gi_tracking_experiments.md`.

## 1. What GI Delivered

The Thor delivery is an inference appliance, not a Python package or a single
trainable checkpoint. The shared Google Drive folder contained:

- `instinctsam-thor-r39.tar.gz`, a Docker image archive;
- Docker Compose and launch files;
- installation notes;
- the GI evaluation/research license and applicable SAM license material; and
- the application/runtime and TensorRT artifacts inside the image.

Recorded identity:

```text
Archive: instinctsam-thor-r39.tar.gz
SHA-256: 30b40a025a76e8a8e911a3c57320637260e9fc78b54fcc4b90b73c7982bb7e75
Docker tag: instinctsam:thor-r39
Loaded image ID: sha256:8fd009341104f6944441d4e6fccbcd9af2598fa03812ee7ae64488ac28906ecd
```

The image contains the userspace environment needed by GI, its application and
tracker, preprocessing and postprocessing, and two supplied TensorRT execution
paths used in this evaluation:

```text
tracking/backbone path: input 768, TensorRT FP16
detection path:         input 1152, TensorRT FP16
```

Both engines loaded and executed on the second Jetson AGX Thor running L4T
R38.4 / CUDA 13 / SM110. A TensorRT engine embeds parameters required for
inference, but it is a compiled execution plan rather than a normal PyTorch
`state_dict`. It is generally tied to TensorRT, CUDA, optimization profiles,
plugins, and target GPU compatibility. It should not be treated as a convenient
training checkpoint or proof that the full model can be rebuilt from public
files.

The delivery is licensed for research/evaluation. Commercial or production use
of the combined system requires separate authorization from GI, in addition to
compliance with the SAM license.

## 2. How Scene Graph Connects to the Delivered Model

The GI model was deliberately not installed into the Ether image. Two
containers run at the same time on one Thor:

| Container | Responsibility |
| --- | --- |
| `ether-onboard-gi-eval` | ROS 2, RGB/depth synchronization, pose/TF, the detector client, depth projection, 3D detections, graph construction, recording, and display |
| `instinctsam-t02-refresh30-headless` | GI model/runtime, TensorRT inference, prompt initialization, detection refreshes, temporal tracking, and packed-mask output |

They share the Thor kernel, unified CPU/GPU memory, and NVIDIA device. Docker
does not assign a separate physical GPU to each container. The GI container is
the model process; the Ether container runs the surrounding ROS and Scene Graph
pipeline. The Original SAM3.1 and GI formal conditions are run serially, not as
two competing detector models on the GPU.

The current evaluation containers communicate over Thor's host loopback:

```text
http://127.0.0.1:8767
```

The hot data path does not use NAS. It uses JPEG input and a compressed NumPy
mask response:

```text
D435 RGB/depth ROS topics
        |
        v
Ether detection_ros_node.py
        |
        | JPEG + HTTP on loopback
        v
GI TensorRT runtime container
        |
        | packed masks, labels, scores, lost flags, sequence metadata
        v
Ether mask-to-depth projection
        |
        v
/scene_graph/detections_3d
        |
        v
scene_graph_ros_node.py -> graph nodes/edges -> display
```

The evaluation-only application overlay exposes the minimum machine-readable
interface needed by Scene Graph:

| Endpoint | Purpose |
| --- | --- |
| `POST /reset` | Clear the current runtime/tracker session |
| `POST /prompt` | Install the configured text prompts |
| `POST /frame.jpg` | Submit one JPEG and return its `input_sequence` |
| `GET /masks.npz` | Return packed masks and metadata for a completed sequence |
| `GET /status.json` | Report readiness, runtime state, and timing metadata |

`masks.npz` contains packed boolean masks, their shape and bit order, labels,
scores, lost flags, runtime frame index, and `input_sequence`. The client waits
for the exact submitted sequence before accepting a result, filters lost
objects, and derives 2D boxes from the returned masks.

GI stops at the detector boundary:

```text
RGB image + text prompts
    -> instance masks + labels + scores
```

The following remain Scene Graph responsibilities and preserve the existing
ROS contract:

- synchronized aligned depth and camera intrinsics;
- mask-pixel depth deprojection;
- camera-to-map/world transformation through pose and TF;
- complete-cloud 3D bounding boxes;
- `/scene_graph/detections_3d` message construction;
- object association, graph nodes, spatial-relation edges, and display.

This boundary avoids mixing the GI TensorRT/Python environment with Ether's ROS
environment and permits either side to be rebuilt or rolled back independently.
Its cost is an extra JPEG/HTTP/NPZ serialization and synchronization boundary.

## 3. Public Weights Versus the Thor Runtime

GI currently publishes InstinctSAM component checkpoints through the gated but
publicly discoverable Hugging Face repository:

```text
https://huggingface.co/GM717/InstinctSAM-ViT-B
```

Access requires a Hugging Face login, acceptance of the repository conditions,
and sharing contact information. As of 2026-08-11, its model card lists:

```text
gitext_large_v4.pt
gitext_base_v3.pt
hiera_large_concept_trunk.pt
concept_vitb_trunk_step6000.pt
vit_base_stageA.pt
```

These are GI-trained text-tower and compressed vision-trunk components. They do
not include Meta-gated SAM3 detector heads, mask decoder, scoring/presence
heads, or a complete standalone SAM3 checkpoint. The intended assembly is:

```text
separately licensed Meta SAM3 checkpoint
    + GIText and/or compressed GI vision trunk
    + InstinctSAM assembly code
    = an assembled InstinctSAM PyTorch model
```

The public assembly/training repository is:

```text
https://github.com/william-Dic/InstinctSAM
```

This benchmark repository includes download and assembly support in
`scripts/download_instinctsam_compressed_checkpoints.sh`,
`scripts/download_instinctsam_vitb_checkpoint.sh`, and
`sam_backend/instinctsam.py`.

The public components must not be described as the exact delivered Thor model
without additional provenance. The Docker runtime also includes the remaining
SAM3 model pieces, tracker/application behavior, preprocessing, postprocessing,
TensorRT compilation settings, optimization profiles, and runtime integration.
No recorded evidence currently proves that the public `.pt` files plus a fully
public recipe reproduce the delivered engine bit-for-bit or output-for-output.

The correct distinction is:

| Artifact | Availability and use |
| --- | --- |
| GIText/Hiera-L/ViT-B component checkpoints | Available through a gated public Hugging Face repository under stated license conditions |
| Meta SAM3 full checkpoint and heads | Obtained separately under Meta's gated SAM license |
| Complete standalone GI PyTorch checkpoint matching the Thor runtime | Not supplied or verified as a single public artifact |
| Thor-optimized GI inference runtime | Supplied as the `instinctsam:thor-r39` Docker archive |
| Bit-for-bit public rebuild of the supplied TensorRT engines | Not established |

## 4. Original SAM3.1 Prompt Processing

The stable Scene Graph detector uses Original `facebook/sam3.1` and its
`_grounding_batched` path. The approximately 39 configured categories are not
implemented as 39 container round trips or 39 model reloads. For each accepted
camera image, the detector handles the category list in one in-process batched
grounding invocation, with internal grouping if required by its batch size.

Conceptually:

```text
one accepted RGB image
    -> in-process SAM3.1 image/text grounding
    -> one configured batch/list of category prompts
    -> masks, labels, boxes, and scores
```

Original still performs grounding again for each accepted image, but it avoids
per-prompt process boundaries and amortizes more work across batched prompts.
It also returns masks directly inside the Ether process, without JPEG encoding,
HTTP polling, or NPZ packing/unpacking.

## 5. Why TensorRT GI Detection Was Initially Slower

TensorRT can accelerate a particular model graph relative to that model's
unoptimized implementation. It does not guarantee that a different model,
resolution, prompt strategy, decoder, or complete application will beat
Original SAM3.1 end to end.

### 5.1 The model-level detection path was already slower

The formal 39-prompt Lifestyle Lab comparison measured:

| Metric | Original SAM3.1 | Stateless GI | GI change |
| --- | ---: | ---: | ---: |
| Runtime/grounding mean | 2,551 ms | 3,160 ms | 23.9% slower |
| Detector/HTTP mean | 2,551 ms | 3,888 ms | 52.4% slower |
| Detector/HTTP p50 | 2,579 ms | 3,962 ms | 53.6% slower |
| Full-frame mean | 2,735 ms | 4,045 ms | 47.9% slower |
| Full-frame p50 | 2,633 ms | 4,009 ms | 52.3% slower |
| Completed frames per source-second | 0.322 | 0.213 | 34.0% lower |

GI mean latency decomposed as:

```text
TensorRT detection                         3,160 ms
remaining GI runtime processing              119 ms
client-visible JPEG/HTTP/NPZ boundary         609 ms
Scene Graph work after the HTTP result        157 ms
---------------------------------------------------
complete frame                              4,045 ms
```

Removing HTTP alone could not win this condition because GI TensorRT detection
was already 609 ms slower than Original grounding before the bridge overhead.

### 5.2 The first adapter discarded GI's tracker advantage

The original drop-in adapter was intentionally stateless for comparison:

```text
for every accepted image:
    reset
    resend every prompt
    run full detection
    wait for masks
```

This converted every image into an expensive detection refresh and discarded
temporal memory. It matched an image-detector contract but did not match GI's
intended operating mode.

### 5.3 Detection is much heavier than a tracking update

The delivery uses a 1152-input detection path and a 768-input tracking/backbone
path. The detection input has 2.25 times as many pixels as the tracking input:

```text
1152^2 / 768^2 = 2.25
```

Detection also performs text grounding, query/object matching, scoring,
presence handling, mask decoding, and tracker initialization or refresh.
Tracking can reuse existing object state and temporal features.

### 5.4 Thirty-nine categories mismatch the intended GI workload

GI is designed around a small prompt set followed by many tracking frames. The
Scene Graph baseline instead supplied approximately 39 categories on every
accepted image. The runtime also reported `max_objects=24`, which does not
naturally cover 39 category/object slots in one operating window.

Original SAM3.1's in-process batched grounding is a better match for the
per-image, many-category workload than resetting and reinitializing GI for all
categories every frame.

### 5.5 TensorRT does not remove non-engine work

The two-container GI path also performs:

- JPEG encoding and copying;
- loopback HTTP request/response handling;
- synchronization and result polling;
- packed-mask snapshot construction;
- NPZ transfer and unpacking;
- label, score, lost-state, and object bookkeeping; and
- existing depth projection and Scene Graph postprocessing.

The 39-prompt run used only 58.9% mean GPU utilization for GI versus 70.3% for
Original, despite GI's higher latency. This indicates CPU/GPU synchronization,
serialization, per-object work, or smaller/fragmented GPU workloads left
execution bubbles. Memory exhaustion and thermal throttling were not the cause.

### 5.6 Smaller prompt sets did not make full GI detection faster

The controlled five-prompt experiment measured every-frame detection:

| Metric | Original O5 | GI G5-R1 |
| --- | ---: | ---: |
| Effective FPS | 1.948 | 1.104 |
| Per-frame p50 | 498.2 ms | 818.5 ms |
| 100-frame wall time | 51.34 s | 90.56 s |

GI every-frame detection remained about 64% slower at p50. On fixed COCO-10
images with one selected text prompt per image, Original averaged 362.5 ms and
GI averaged 509.6 ms, so GI was 40.6% slower at the tested operating points.

These tests show that the 39-prompt mismatch magnified the problem, but was not
the only cause. The complete GI image-detection path did not beat Original in
the tested configurations.

## 6. Where the GI Speedup Actually Comes From

The useful integration keeps one long-lived session:

```text
accepted frame 0:     reset + prompts + detection + tracker initialization
accepted frames 1-29: tracking only
accepted frame 30:    detection refresh
accepted frames 31-59: tracking only
accepted frame 60:    detection refresh
```

Scene Graph adds opt-in `instinctsam_stateful` behavior. It initializes once,
retains runtime state, and invalidates that state after an HTTP failure so the
next accepted frame cleanly resets and reinstalls prompts. The existing
single-worker busy-frame rejection provides backpressure and prevents an
unbounded queue of stale camera frames.

Measured five-prompt latency:

| Mode | p50 latency |
| --- | ---: |
| Original per-frame detection | 498.2 ms |
| GI per-frame detection | 818.5 ms |
| GI R30 tracking before headless mode | 196.3 ms |
| GI R30 headless tracking | 149.6 ms |
| GI integrated Scene Graph detector full-frame p50 | 177.1 ms |

Therefore:

```text
498.2 / 149.6 = 3.33x
498.2 / 177.1 = 2.81x
```

Those speedups compare Original per-frame detection with GI tracking frames.
They must not be reported as GI full detection being 2.8--3.3 times faster.
The accurate statement is:

> GI headless tracking frames were 3.33 times faster than Original five-prompt
> per-frame detection, while GI full detection remained slower than Original.

T02 headless mode removed duplicate runtime UI JPEG generation while preserving
masks, labels, IDs, lost flags, and sequence metadata bit-for-bit. T04 then
moved downstream-equivalent 1 cm voxel aggregation ahead of 3D JSON transport,
reducing serialized detection data by 94.92% and complete-publication p95 from
1,072.7 ms to 820.6 ms. Neither optimization makes GI detection itself faster;
they remove application and transport overhead around the stateful tracker.

## 7. Reporting Rules

Use these descriptions to avoid conflating unlike measurements:

- **GI full detection:** reset/grounding or a scheduled detection refresh;
- **GI tracking frame:** an update that reuses initialized prompts and tracker
  state;
- **runtime latency:** time reported inside the GI application;
- **HTTP/client latency:** runtime plus JPEG, synchronization, packed-mask, and
  loopback-transfer overhead;
- **Scene Graph full-frame latency:** detector/client plus depth projection,
  3D message construction, and publication;
- **teacher agreement:** mask comparison against Original SAM, not ground-truth
  accuracy; and
- **COCO mIoU:** ground-truth accuracy on the fixed COCO subset.

The recommended summary is:

> The supplied TensorRT runtime was not a faster drop-in image detector for the
> tested one-, five-, or 39-prompt workloads. The useful advantage appeared
> only after integrating GI as a long-lived detector/tracker, amortizing an
> expensive detection refresh over many faster tracking frames and removing
> avoidable visualization and 3D transport overhead.

# GI Tracking Integration Experiments

Platform: second NVIDIA Jetson AGX Thor (`magni`)

Branch: `dev/instinctsam-gi-runtime`

This is the experiment log for integrating the General Instinct (GI) runtime
with Scene Graph as a stateful detector/tracker. Every attempt must be described
here before execution, then updated with commands, checksums, measurements,
failures, and a decision. Failed attempts remain in the log and are never mixed
with formal metrics.

The stable Scene Graph checkout and the completed 39-prompt A/B results are not
modified by these experiments.

## Common Rules

- Use camera source timestamps or an immutable ordered frame manifest, never
  model-completion time, to align systems.
- Store code and lightweight manifests in Git. Store images, masks, telemetry,
  overlays, checkpoints, engines, and logs on NAS.
- Run Original and GI conditions separately so they do not contend for the GPU.
- Exclude model/container startup from warm latency, but report initialization
  and first-keyframe latency separately.
- Preserve the prompt list, threshold, refresh cadence, runtime overlay SHA-256,
  Docker image ID, repository commit, input SHA-256, and raw per-frame metrics.
- Label comparison with Original as `teacher agreement`, not ground-truth
  accuracy.
- Record Linux unified-memory headroom, Docker working set, NVIDIA process
  memory, CPU, GPU utilization, power, and temperature for every condition.
- Do not merge GI code into the stable Scene Graph branch until an experiment
  passes both speed and quality gates.

## Attempt Register

| ID | Question | Status | Decision |
| --- | --- | --- | --- |
| T01 | Does GI become useful when five prompts are initialized once and tracked, with grounding only every 30 frames? | Complete | Partial pass: tracking is useful, but 4.17x missed the 5x gate |
| T02 | How much latency can a mask-only/headless API remove without changing model output? | Complete | Pass: 23.8% lower tracking p50 with bitwise-identical outputs |
| T03 | Can Scene Graph preserve GI tracker state and consume live camera input without accumulating stale frames? | Complete | Partial pass: 2.72x throughput; complete-publication p95 missed by 72.7 ms |
| T04 | Can 1 cm voxel aggregation remove redundant 3D JSON points without changing graph geometry? | Designed; execution pending | Authorized by T03 publication tail |

## T01: Five-Prompt Keyframe Detection and Tracking

### Question and hypothesis

The completed 39-prompt A/B used the GI runtime as a stateless image detector.
The client reset the runtime and resent all prompts for every image. In addition,
the evaluation overlay forced a detection pass for every externally supplied
frame. That measured a valid drop-in detector configuration, but not the
delivery's intended initialize-then-track mode.

T01 tests whether GI has a useful model-level advantage when it receives one
prompt initialization and then processes a contiguous video sequence. The
hypothesis is that tracking-only frames will be much faster than repeated text
grounding while retaining acceptable mask agreement over a 30-frame interval.

### Fixed input

- Source bag: Lifestyle Lab D435 recording used by the completed Scene Graph A/B.
- Selection: 100 consecutive color JPEG messages starting near source offset
  140 s, where the earlier fixed-five-second samples contain stable table,
  keyboard, and book detections.
- The extractor will record the exact source timestamp for every image in an
  immutable JSONL manifest and create `SHA256SUMS`.
- Frame order and JPEG bytes must be identical for every condition.
- Expected duration is approximately 3.3 s at the recorded camera rate.

Fixed prompts, in order:

```text
keyboard
table
book
computer desk
stool
```

These prompts are selected before extraction from categories already observed
near the target source interval. They will not be changed after viewing T01
outputs.

### Conditions

| ID | Backend | State | Detection cadence | Threshold |
| --- | --- | --- | --- | ---: |
| O5 | Original SAM3.1 `_grounding_batched` | Independent image inference | Every frame | Existing Original operating point, 0.8 |
| G5-R1 | GI runtime | One prompt initialization; tracker state persists | Every frame | 0.5 |
| G5-R30 | GI runtime | One prompt initialization; tracker state persists | Frames 0, 30, 60, and 90 | 0.5 |

`G5-R1` controls for the effect of detection cadence while keeping the same
stateful client and runtime implementation as `G5-R30`. It is not the prior
stateless `/reset`-per-frame condition.

### Runtime change allowed for T01

Create a separate evaluation overlay variant that makes external-frame
detection follow `--detect-every` instead of unconditionally detecting every
external frame. The default delivery, completed A/B overlay, Docker image, and
stable Scene Graph code remain unchanged. Archive the variant and its SHA-256
on NAS.

The tracked benchmark client must:

1. reset once before the sequence;
2. set the five prompts once;
3. upload each JPEG in manifest order;
4. wait for the matching `input_sequence` mask snapshot;
5. save packed masks and per-frame runtime status;
6. never reset or resend prompts between sequence frames.

### Measurements

Speed:

- container/model startup, excluded from warm latency;
- first-frame initialization latency;
- refresh-frame and tracking-only client latency distributions;
- runtime `backbone_ms`, `tracker_ms`, `detect_ms`, and `process_ms`;
- sequence wall time and effective FPS.

Quality and temporal stability:

- per-label directed instance IoU against O5 teacher masks;
- teacher instance recall at IoU 0.5;
- G5-R30 agreement against G5-R1 to isolate tracking drift from model-family
  differences;
- mask/label counts, lost-object rate, and non-empty-frame rate;
- metrics by frames since refresh: 1-9, 10-19, and 20-29;
- overlays or a contact sheet at frame 0, before each refresh, and final frame.

Hardware:

- mean, p95, and maximum GPU utilization;
- Linux minimum `MemAvailable` and Docker/NVIDIA process memory;
- mean and maximum GPU/system power;
- mean and maximum GPU temperature;
- summed container CPU.

### Exploratory success gates

T01 passes the speed gate only if G5-R30 tracking-only p50 is at most 250 ms
and at least 5x faster than G5-R1 refresh-frame p50. It passes the stability
gate if its teacher recall at IoU 0.5 is no more than 0.10 below G5-R1 and its
G5-R1 agreement does not show a monotonic collapse across the three
frames-since-refresh buckets.

It must also complete all 100 frames without a runtime restart, remain below
80 C, and retain at least 32 GiB of Linux unified-memory headroom. These are
screening gates, not production requirements.

If T01 fails at the model/runtime level, stop before implementing shared-memory
transport or changing the ROS Scene Graph scheduler. If it passes, T02 may
measure transport optimization, followed by T03's asynchronous graph-aware
scheduler.

### Planned artifact root

```text
/mnt/nas/danny/thor-scene-graph/run-artifacts/gi-tracking-t01-20260808/
```

Planned layout:

```text
input/
original-refresh1/
gi-refresh1/
gi-refresh30/
report/
failed-attempts/
```

### Results

T01 completed all three 100-frame conditions. The immutable input starts at
`1781705602980560640` ns and ends at `1781705606316382720` ns, spanning
3.33582208 s. Mean frame period was 33.70 ms. The 100 JPEG hashes and the input
manifest/config/summary checksums passed.

Evaluation overlay:

```text
SHA-256: 935e8f0243454c166aad0ffe6d6601b41a214df8f70e3a2ed1e655df5398a409
```

It differs from the completed 39-prompt overlay only by allowing external
sequential input to honor `--detect-every`. Stateless reset-and-prompt clients
still trigger a detection for each frame.

#### Speed

| Metric | O5 | G5-R1 | G5-R30 |
| --- | ---: | ---: | ---: |
| Startup/model initialization | 12.06 s | 84.63 s | 87.65 s |
| Sequence wall time | 51.34 s | 90.56 s | 28.81 s |
| Effective sequence FPS | 1.948 | 1.104 | 3.471 |
| All-frame p50 | 498.2 ms | 818.5 ms | 197.2 ms |
| Mask observations | 329 | 597 | 470 |
| Non-empty frames | 95 | 100 | 100 |

G5-R30 contained exactly four measured detection frames: 0, 30, 60, and 90.
The remaining 96 frames had `detect_ms=0` and are the formal tracking-only set.

| G5-R30 frame type | Count | Mean client | p50 client | p95 client | p50 runtime process |
| --- | ---: | ---: | ---: | ---: | ---: |
| Refresh | 4 | 1,154.1 ms | 868.9 ms | 1,913.7 ms | 803.8 ms |
| Tracking only | 96 | 197.2 ms | 196.3 ms | 219.2 ms | 125.2 ms |

The first refresh was a cold sequence initialization at 2,088.8 ms. Later
refreshes were 921.1, 789.8, and 816.7 ms. Tracking-only p50 was 4.17x faster
than G5-R1 and 2.54x faster than Original. The approximately 71 ms difference
between tracking client p50 and runtime-process p50 motivates T02.

#### Teacher agreement and tracking stability

| Directed comparison | Instance mIoU | Recall at IoU 0.5 |
| --- | ---: | ---: |
| O5 to G5-R1 | 0.6014 | 0.6231 |
| O5 to G5-R30 | 0.5215 | 0.5380 |
| G5-R1 to G5-R30 | 0.7686 | 0.7873 |

G5-R30 teacher recall was 0.0851 below G5-R1, within the pre-registered 0.10
gate. G5-R1-to-G5-R30 agreement by tracking phase was:

| Frames since refresh | Instance mIoU | Recall at IoU 0.5 |
| --- | ---: | ---: |
| Refresh | 0.8483 | 0.8636 |
| 1-9 | 0.7767 | 0.7953 |
| 10-19 | 0.7588 | 0.7778 |
| 20-29 | 0.7590 | 0.7778 |

Agreement drops after initialization, then plateaus rather than continuing to
collapse through frame 29. None of the GI frames reported a lost object. These
numbers are teacher/self agreement on one short sequence, not ground-truth
accuracy.

#### Hardware

| Mean / limit metric | O5 | G5-R1 | G5-R30 |
| --- | ---: | ---: | ---: |
| Mean GPU utilization | 89.8% | 64.0% | 75.3% |
| GPU utilization p95 | 97.0% | 96.9% | 96.0% |
| Mean GPU power | 32.9 W | 22.6 W | 21.7 W |
| Maximum GPU temperature | 52 C | 49 C | 47 C |
| Minimum Linux `MemAvailable` | 87.18 GiB | 82.33 GiB | 82.28 GiB |
| Mean Docker working set | 5.57 GiB | 11.41 GiB | 11.44 GiB |
| Mean NVIDIA process memory | 6.02 GiB | 8.84 GiB | 8.78 GiB |

No condition approached the temperature or unified-memory gates. G5-R30 used
the same model memory as G5-R1; its improvement comes from scheduling, not a
smaller resident model.

#### Gate decision

| Gate | Result |
| --- | --- |
| Tracking p50 <= 250 ms | Pass |
| At least 5x faster than G5-R1 | Fail: 4.17x |
| Teacher recall decrease <= 0.10 | Pass: 0.0851 |
| No monotonic phase-bucket collapse | Pass |
| Complete 100 frames, below 80 C, at least 32 GiB available | Pass |

T01 is therefore a strict partial pass. It missed one deliberately aggressive
speed ratio, but it demonstrated a real model/runtime tracking advantage and
stable short-horizon output. T02 is authorized only as a bounded measurement of
the remaining client/runtime boundary; ROS Scene Graph scheduling remains
unchanged until T02 is evaluated.

Full generated report and figures:

```text
/mnt/nas/danny/thor-scene-graph/run-artifacts/gi-tracking-t01-20260808/report/
```

Preserved preflights excluded from all metrics:

- `extractor-shutdown-preflight`: the first ROS extractor reached 100 images
  but called shutdown inside its subscription callback and did not close the
  manifest. The fixed extractor uses `spin_once` and a done flag.
- `original-preflight-pythonpath-duration`: incorrect copied-package path and
  missing resource-sampler duration.
- `original-preflight-rclpy-pythonpath`: replacing `PYTHONPATH` hid ROS Python
  packages; the formal command prepends instead.
- `original-preflight-config-path`: standalone Detection construction lacked
  the launch-provided object config; the formal command sets the explicit fixed
  T01 config.

## T02: Mask-Only External API

### Question and hypothesis

G5-R30 tracking-only runtime-process p50 was 125.2 ms, while client p50 was
196.3 ms. The current UI-oriented render worker still constructs an overlay and
JPEG-encodes both raw and tracked frames before publishing the mask snapshot.
T02 tests whether disabling those unused UI products reduces the approximately
71 ms boundary without changing inference, tracker state, masks, IDs, or labels.

This is intentionally smaller than a shared-memory redesign. If mask-only mode
does not materially reduce latency, transport work will stop rather than adding
IPC complexity.

### Fixed input and control

- Reuse the exact T01 input manifest, 100 JPEGs, five prompts, threshold 0.5,
  and refresh cadence 30.
- Reuse `G5-R30` as the control; do not rerun or alter it.
- Run a fresh `G5-R30-H` container so tracker state is independent.
- Preserve the existing 768 tracking / 1152 detection TensorRT engines.

### Runtime change allowed for T02

Create another evaluation overlay variant with an explicit `--api-headless`
flag. When set, the render worker must still copy masks, labels, scores, lost
flags, frame index, and `input_sequence` into `/masks.npz`, but it skips UI
overlay drawing and raw/tracked JPEG encoding. Default behavior remains
unchanged. Archive the new overlay and checksum separately.

No model, threshold, prompt, tracking, memory stride, mask resize, HTTP client,
or Scene Graph code may change in T02.

### Measurements and gates

- Same per-frame latency, runtime status, mask, resource, and startup records as
  T01.
- Compare tracking-only client p50/p95 and the client-minus-process boundary.
- Compare every G5-R30-H mask to G5-R30 by label and source frame.
- Inspect refresh indices to confirm 0/30/60/90.

T02 passes only if tracking-only client p50 is at most 160 ms, at least 20%
lower than G5-R30's 196.3 ms, all masks/labels/lost flags are bitwise identical,
all 100 frames complete, and hardware gates remain satisfied.

Planned artifact directory:

```text
/mnt/nas/danny/thor-scene-graph/run-artifacts/gi-tracking-t01-20260808/gi-refresh30-headless/
```

### Results

T02 completed all 100 frames with refreshes at exactly frames 0, 30, 60, and
90. The evaluation overlay checksum was:

```text
SHA-256: c6685227317c6698e4cd56f2ba1ba28905cb756ba182d61fb3af96192d703efd
```

The variant is archived on NAS at:

```text
/mnt/nas/danny/thor-scene-graph/candidates/instinctsam-drive-1DyLOdRXWD_GT4s5jKT6AGkBbO5TKjS5c/runtime-overlay/live_tracking_sam3.t02_api_headless.py
```

#### Speed

| Metric | G5-R30 UI control | G5-R30-H headless | Change |
| --- | ---: | ---: | ---: |
| Startup/model initialization | 87.65 s | 87.65 s | No material change |
| Sequence wall time | 28.81 s | 27.02 s | -6.2% |
| Effective sequence FPS | 3.471 | 3.701 | +6.6% |
| Tracking-only client p50 | 196.3 ms | 149.6 ms | -23.8% |
| Tracking-only client p95 | 219.2 ms | 175.0 ms | -20.2% |
| Tracking-only runtime-process p50 | 125.2 ms | 129.4 ms | +3.3% |
| Client-minus-process boundary p50 | 71.1 ms | 20.3 ms | -71.5% |

The headless all-frame mean/p50/p95 were 190.3/152.7/181.2 ms. The four
refresh frames averaged 1,097.3 ms, with p50 810.4 ms and p95 1,828.0 ms. The
96 tracking-only frames averaged 152.5 ms. The model/runtime processing time
did not improve; the gain is specifically the removal of UI drawing and two
JPEG encodes from the API response boundary.

#### Output identity and hardware

All 100 source-aligned frames had exactly identical masks, labels, IDs, lost
flags, and scores compared with the UI control. Therefore T02 changes delivery
work only, not model output.

| Resource metric | G5-R30-H |
| --- | ---: |
| Mean / p95 GPU utilization | 61.0% / 95.0% |
| Mean / maximum GPU power | 22.27 W / 26.37 W |
| Mean / maximum system power | 50.14 W / 78.25 W |
| Mean / maximum GPU temperature | 43.58 C / 46 C |
| Minimum Linux `MemAvailable` | 82.51 GiB |
| Mean Docker working set | 11.46 GiB |
| Mean NVIDIA process memory | 8.77 GiB |

#### Gate decision

| Gate | Result |
| --- | --- |
| Tracking-only p50 <= 160 ms | Pass: 149.6 ms |
| At least 20% below G5-R30 | Pass: 23.8% |
| All outputs bitwise identical | Pass: 100/100 frames |
| Complete 100 frames and preserve cadence | Pass |
| Below 80 C and at least 32 GiB available | Pass |

T02 passes. Shared-memory transport is not justified yet because the bounded
headless change removed 71.5% of the observed client/runtime boundary while
preserving output exactly. T03 is authorized to integrate this mode into the
candidate Scene Graph pipeline.

Full generated report and figures:

```text
/mnt/nas/danny/thor-scene-graph/run-artifacts/gi-tracking-t01-20260808/report-t02/
```

## T03: Stateful GI Scheduling in Scene Graph

### Question and hypothesis

The current candidate Scene Graph HTTP backend resets GI and resends prompts
for every accepted color/depth pair. That discards tracker state and converts
every accepted frame into a slow grounding frame. The synchronized ROS callback
already rejects new frames while a worker is active, so it behaves as a
one-worker, latest-future-frame scheduler rather than building an unbounded
queue.

T03 tests the smallest integration change: initialize prompts once, retain GI
state across accepted frames, and let the existing busy-frame rejection provide
backpressure. The hypothesis is that this will increase completed 3D detection
updates, keep detections close to the live camera timestamp, and still update
the graph without altering projection, point-cloud, or graph code.

### Code boundary and safety

Only `~/scene-graph-instinctsam` on branch `dev/instinctsam-integration` may be
changed. The stable checkout under Ether, its image, and the completed A/B
artifacts remain untouched.

Add an opt-in `instinctsam_stateful` detector parameter, default `false`:

1. when disabled, preserve the existing reset-and-prompt-per-frame behavior;
2. when enabled, call reset and prompt once before the first accepted frame;
3. reuse the runtime session for later accepted frames;
4. invalidate the initialized state after an HTTP failure so the next accepted
   frame performs a clean reset and prompt;
5. do not change image timestamps, mask-to-depth projection, detection message
   construction, graph update logic, threshold, prompts, or category mapping.

The GI runtime uses the T02 `--api-headless` overlay and `--detect-every 30`.
Scene Graph itself also runs headless so visualization does not contaminate the
throughput measurement.

### Fixed live-playback input and conditions

- Source: the same Lifestyle Lab D435 bag and the same region beginning near
  camera offset 140 s as T01.
- Playback: 30 source seconds at 1x ROS bag rate, separately for each condition.
- Prompts, in fixed order: `keyboard`, `table`, `book`, `computer desk`, and
  `stool`.
- GI threshold: 0.5; refresh cadence: every 30 GI-accepted frames.
- Start each condition from stopped model containers and a fresh GI runtime.
- Record image source timestamps rather than completion time.

| ID | Scene Graph HTTP behavior | GI mode | Purpose |
| --- | --- | --- | --- |
| SG5-S | Reset and prompt for every accepted frame | API headless | Existing stateless control |
| SG5-T30 | Reset/prompt once, then retain state | API headless, refresh every 30 accepted frames | Candidate stateful integration |

The conditions run serially. Container/model startup is reported separately
and excluded from steady-state pipeline measurements.

#### T03b controlled pose amendment

T03a showed that starting Cartographer from the middle of this bag is not a
repeatable A/B input. Although both attempts had identical camera messages,
one produced 252 tracked poses and the other only 18 because the latter hit
hundreds of transform past-extrapolation failures. Comparing detector
throughput under those conditions would be invalid.

Before further execution, T03 is therefore split into two scopes:

- T03b is the controlled detector-to-graph scheduler experiment. A small ROS
  fixture subscribes to the color stream and publishes an identity
  `/tracked_pose` plus `map -> d435_color_optical_frame` transform with the
  exact camera header timestamp. Bag playback is restricted to the fixed
  color, aligned depth, and camera-info topics. This retains real images,
  recorded depth, synchronization, mask-to-3D projection, detection messages,
  and graph updates while removing Cartographer as an uncontrolled variable.
- A later deployment validation must start localization from bag offset zero
  and pre-roll to the measured interval. T03b passing does not replace that
  validation and makes no localization-performance claim.

The pose fixture, topic filter, timing, and code checksum are identical in
SG5-S and SG5-T30. All pre-registered T03 gates remain unchanged, except
"source-to-publication" is explicitly detector-to-graph pipeline latency under
the fixed pose fixture.

The detection node continues to receive the five-prompt T01 config. The Scene
Graph node receives the existing full `scene_objects.json`, because it also
contains graph-only fields such as `relation_colors`; this does not expand the
detector prompt list. The recorder stops itself after 42 wall seconds and must
write `recorder_summary.json` before a run can be accepted. This replaces
process-signal shutdown so both conditions have the same deterministic
measurement lifetime.

### Measurements

Pipeline behavior:

- source camera messages, accepted/started frames, completed detections, and
  busy-frame skips;
- completed update rate and source-frame coverage;
- source timestamp to detection publication latency, including p50/p95/max and
  linear latency slope over playback time;
- accepted-frame source timestamp gaps to verify fresh-frame sampling instead
  of queued sequential processing;
- `/scene_graph/detections_3d` message count, detections per message, and
  non-empty-message rate;
- graph node/object counts at the end of playback;
- GI refresh indices and runtime detect/tracker/process timing.

Hardware and capacity:

- GPU utilization, GPU/system power, and GPU temperature time series;
- Linux `MemAvailable`, Docker working set, NVIDIA process memory, and summed
  container CPU;
- peak resident model footprint and remaining unified-memory headroom.

Quality is a pipeline sanity check rather than a ground-truth claim: preserve
labels, require non-empty 3D detections and graph objects, inspect mask/3D
overlays at fixed source times if available, and report any lost-object or empty
collapse. T01 remains the source-aligned teacher-agreement measurement.

### Pre-registered gates

SG5-T30 passes this integration screen only if it:

1. completes at least twice as many 3D detection updates as SG5-S in the same
   30 source seconds;
2. has source-to-publication p50 below 500 ms and p95 below 1.0 s;
3. shows no accumulating stale queue: latency slope is at most 10 ms per source
   second and accepted source timestamps continue advancing throughout playback;
4. refreshes at accepted-frame indices 0, 30, 60, and so on, without resetting
   between them;
5. publishes at least one non-empty 3D detection message and produces at least
   one Scene Graph object node;
6. completes without restart, remains below 80 C, and retains at least 32 GiB
   of Linux unified-memory headroom.

If an instrumentation limitation prevents an exact metric, preserve the run as
a preflight, document the limitation, fix only the measurement, and rerun both
conditions. Do not reinterpret a failed gate after viewing results.

### Planned artifact directory

```text
/mnt/nas/danny/thor-scene-graph/run-artifacts/gi-scene-graph-t03-20260808/
```

### Results

T03b completed both controlled conditions with identical input: 866 camera
messages, 866 fixture poses, the same first/last source timestamps, and
29.25557376 source seconds. The fixture SHA-256 was
`f703f906a5dc8730220bc61ad7a64b81e478b8f76174e2f73e19a4c42caa0460`.
The T02 runtime overlay SHA-256 remained
`c6685227317c6698e4cd56f2ba1ba28905cb756ba182d61fb3af96192d703efd`.

#### Speed and scheduling

| Metric | SG5-S stateless | SG5-T30 stateful |
| --- | ---: | ---: |
| Runtime startup | 59.42 s | 59.34 s |
| Detector calls | 33 | 88 |
| Completed full frames | 32 | 87 |
| Completed frames / source-second | 1.094 | 2.974 |
| Full-frame p50 / p95 | 815.8 / 1,176.4 ms | 198.0 / 617.3 ms |
| HTTP p50 | 667.5 ms | 153.5 ms tracking-only |
| Tracking runtime-process p50 / p95 | n/a | 131.7 / 156.1 ms |
| Busy synchronized callbacks | 825 / 857 | 770 / 857 |

Stateful scheduling completed 2.71875x as many full frames. It initialized
prompts once and refreshed at exactly zero-based accepted indices 0, 30, and
60; the other 85 calls were tracking-only. The callback continued consuming
new source timestamps rather than queuing every camera frame.

#### Source-to-publication latency and graph output

| Metric | SG5-S stateless | SG5-T30 stateful |
| --- | ---: | ---: |
| Non-empty 3D source frames | 18 | 44 |
| 3D detection messages | 26 | 79 |
| Mean gap between non-empty source frames | 1.574 s | 0.448 s |
| Camera to first 3D p50 / p95 | 1,072.4 / 1,301.6 ms | 452.6 / 991.3 ms |
| Camera to last 3D p50 / p95 | 1,112.5 / 1,397.3 ms | 452.6 / 1,072.7 ms |
| Complete-publication latency slope | -22.0 ms/source-s | -32.6 ms/source-s |
| Final graph nodes / edges | 2 / 0 | 5 / 5 |

Stateful first-publication p95 passed 1.0 s, but the stricter complete-frame
measurement waits for the final per-object JSON message and missed by 72.7 ms.
The negative slope and advancing source timestamps show no accumulating stale
queue. The gap between first and last messages, plus the very large serialized
per-instance point arrays, motivates T04 rather than shared-memory mask work.

Stateful output contained 54 table, 15 book, and 10 keyboard 3D messages. Its
final graph contained two keyboards, two books, and one table. These are
pipeline sanity results under a fixed pose, not semantic accuracy; T01 remains
the source-aligned mask-agreement experiment.

#### Hardware and capacity

| Resource metric | SG5-S stateless | SG5-T30 stateful |
| --- | ---: | ---: |
| Mean / p95 GPU utilization | 48.1% / 91.8% | 33.7% / 96.0% |
| Mean GPU power | 15.61 W | 15.37 W |
| Mean system power | 40.24 W | 34.70 W |
| Maximum GPU temperature | 45 C | 47 C |
| Minimum Linux `MemAvailable` | 82.61 GiB | 82.63 GiB |
| Mean Docker working set | 11.11 GiB | 11.09 GiB |
| Mean NVIDIA process memory | 8.77 GiB | 8.73 GiB |

The resident-memory footprint is essentially unchanged by scheduling. Thor
retained more than 82 GiB of unified-memory headroom, but both modes still had
GPU bursts above 90%; co-resident model planning must consider peak compute,
not only memory capacity.

#### Gate decision

| Gate | Result |
| --- | --- |
| At least 2x completed full frames | Pass: 2.71875x |
| Complete-publication p50 below 500 ms | Pass: 452.6 ms |
| Complete-publication p95 below 1.0 s | Fail: 1,072.7 ms |
| Latency slope at most 10 ms/source-s | Pass: -32.6 ms/source-s |
| Refresh cadence exact | Pass: 0, 30, 60 |
| Non-empty 3D output and graph | Pass: 44 frames, 5 nodes |
| No pipeline traceback | Pass |
| Below 80 C and at least 32 GiB available | Pass |

T03 is a strict partial pass: eight of nine gates passed. Stateful GI is the
correct integration mode and substantially improves throughput, but the 3D
publication tail remains above the pre-registered limit. The runtime reports a
post-playback external-frame idle timeout after the player stops; it occurs
after all measured calls and is not a playback restart.

Full generated report and figures:

```text
/mnt/nas/danny/thor-scene-graph/run-artifacts/gi-scene-graph-t03-20260808/report-t03b/
```

Preserved preflights excluded from all T03 metrics:

- `sg5-stateless-missing-urdf`: the first control runner did not pass
  `ETHER_ROBOT_URDF` into a freshly started Ether container. Localization
  exited before bag playback, so `/tracked_pose` never appeared and the
  detector remained in its startup wait. The recorder captured 866 camera
  messages but zero detections. The GI runtime subsequently timed out waiting
  for an external frame. This run is invalid, not a zero-throughput control.
  The corrected runner passes the known local robot URDF, pbstream, and map
  paths explicitly.
- `sg5-stateless-missing-tf-static`: localization started with the corrected
  paths, but mid-bag playback skipped the transient `/tf_static` samples near
  offset zero. Cartographer repeatedly reported that `livox_frame` did not
  exist and never published `/tracked_pose`; the detector again remained in
  its startup wait. The corrected runner now starts all subscribers, replays
  only `/tf_static` once at 1000x without `/clock`, and only then starts the
  measured 1x interval. This preload is outside timing and identical for both
  conditions.
- `sg5-stateless-localization` and `sg5-stateful-localization`: the first pair
  with static transforms present still had non-equivalent localization. Both
  recorded 866 camera messages over 29.26 source seconds, but the control
  produced 252 pose messages and 252 synchronized callbacks while stateful
  produced only 18. The stateful localization log contained 906 transform
  past-extrapolation warnings and its recorder did not close cleanly. Its 14 GI
  calls did demonstrate the intended session behavior (one initialization and
  13 tracking calls), but neither run is used for throughput or gate results.
  T03b uses the pre-registered controlled-pose amendment above.
- `sg5-stateless-pose-fixture-v1` and
  `sg5-stateful-r30-pose-fixture-v1`: the pose fixture gave both attempts 866
  camera and 866 pose messages, confirming controlled synchronization. The
  stateful detector completed 92 calls with one initialization and four
  refreshes, versus 32 stateless calls. However, the minimal detector prompt
  config was also passed to the graph node; when stateful output created an
  edge, graph visualization raised `KeyError: relation_colors`. Its recorder
  also failed to close after a process signal. The corrected protocol keeps the
  five-prompt detector config, gives the graph its complete existing config,
  and uses fixed-duration recorder shutdown. Per the instrumentation rule,
  both conditions will be rerun and the v1 numbers are excluded from gates.
- `state-test-unittest-path`: the first isolated state test passed an absolute
  file path to `python -m unittest`, which was interpreted as a module name;
  no tests executed.
- `state-test-ros-environment`: two subsequent state-test invocations had not
  sourced the container's actual ROS Humble installation (the first assumed
  Jazzy); no tests executed. With `/opt/ros/humble/setup.bash` and the workspace
  sourced, all three state tests passed: one stateful initialization, per-frame
  stateless initialization, and reinitialization after an injected HTTP
  failure.

- `report-matplotlib-keyword`: the first report generation wrote its JSON and
  Markdown, then Thor's older Matplotlib rejected the newer `tick_labels`
  keyword. The compatible rerun uses `labels`; this changed no measurements.

## T04: Voxel-Compressed 3D Detection Transport

### Question and hypothesis

Each non-empty instance currently serializes every masked depth pixel as nested
JSON floats. Scene Graph immediately compresses those points into 1 cm voxels
in `GeometryBuilder`, so the transport performs substantial redundant JSON
formatting, copying, DDS serialization, parsing, and NumPy allocation.

T04 tests whether performing the same 1 cm aggregation before publication can
bring complete source-to-last-3D p95 below 1.0 s while preserving the exact
geometry consumed by Scene Graph.

### Fixed control and allowed change

- Reuse the formal T03b `SG5-T30` run as the dense-point control.
- Run one fresh `SG5-T30-V01` condition with the same bag, pose fixture,
  prompts, threshold, refresh cadence, headless runtime, and resource sampling.
- Add opt-in `detection_node.instance_point_voxel_size`, default `0.0` so
  existing behavior is unchanged. T04 sets it to `0.01` m.
- Compute world-frame bounding-box position and size from the full point set.
  Only then group points using the same integer voxel index rule as downstream
  `GeometryBuilder` and transmit each voxel's mean point.
- Record raw point count, transmitted point count, JSON bytes, and publication
  time. No mask, confidence, association, graph, or runtime code may change.

### Geometry verification

Before the live run, replay every dense T03b stateful detection through the new
aggregation helper. For each message, compare the downstream 1 cm voxel cell
keys and per-cell mean against direct `GeometryBuilder`-equivalent compression
of the full points. Position and size fields must remain computed from the full
cloud. This is a transport/geometry test, not segmentation mIoU.

### Pre-registered gates

T04 passes only if:

1. complete source-to-last-3D p95 is below 1.0 s and at least 10% below T03b's
   1,072.7 ms;
2. total 3D JSON bytes and transmitted point count are each at least 50% lower;
3. offline voxel cell keys are identical for every T03b stateful detection and
   maximum per-cell mean error is at most `1e-5` m;
4. it completes at least 87 full frames, preserves exact R30 cadence, produces
   at least 40 non-empty 3D source frames and a non-empty graph, with no
   pipeline traceback;
5. it remains below 80 C and retains at least 32 GiB `MemAvailable`.

Planned artifact directory:

```text
/mnt/nas/danny/thor-scene-graph/run-artifacts/gi-scene-graph-t04-20260808/
```

### Results

Pending execution.

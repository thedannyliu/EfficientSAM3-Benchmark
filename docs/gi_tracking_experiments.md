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
| T04 | Can 1 cm voxel aggregation remove redundant 3D JSON points without changing graph geometry? | Complete | Partial pass: transport and latency gates passed; non-empty 3D frame gate failed |
| T05 | Can GI batch all text prompts in one grounding call and make every-frame five-prompt detection faster than Original SAM3.1? | Complete | Partial pass: batching cut p50 25.8%, but 607.3 ms remained slower than Original |
| T06 | Can an R1 detector-only GI path remove redundant tracker work and beat Original O5 end to end? | Complete | Partial pass: 459.8 ms beat Original, but teacher agreement dropped substantially |
| T07 | Can 768-input batched detector-only inference improve both T06 speed and teacher agreement? | Complete | Fail: 394.0 ms was faster, but both teacher metrics declined |

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

The offline geometry check and the live `SG5-T30-V01` condition completed. The
live run used the same 866 camera messages, 866 deterministic fixture poses,
29.25557376 source seconds, prompt order, threshold, R30 cadence, and T02
runtime overlay as the formal T03b dense control. The candidate Scene Graph
implementation is commit `dfc7085` (`Compress Scene Graph instance points by
voxel`). The stable checkout remained at `e9c7a14` and was not modified.

#### Offline geometry verification

The checker replayed all 79 non-empty dense T03b messages through the proposed
1 cm aggregation. It compared the integer cell keys and cell means obtained
from the transmitted aggregate points with direct downstream-equivalent
compression of the original full point clouds.

| Metric | Result |
| --- | ---: |
| Dense messages checked | 79 |
| Original points | 1,558,802 |
| Aggregated points | 81,660 |
| Point reduction | 94.76% |
| Identical cell-key sets | 79 / 79 |
| Maximum cell-mean error | 0.0 m |

Bounding-box position and size remain calculated from the full cloud before
aggregation. This establishes exact equivalence for the geometry data consumed
by `GeometryBuilder`; it does not establish segmentation accuracy.

#### Live latency and transport

| Metric | T03b dense control | T04 voxel01 | Change |
| --- | ---: | ---: | ---: |
| Completed full frames | 87 | 92 | +5.7% |
| Completed frames / source-second | 2.974 | 3.145 | +5.7% |
| Full-frame p50 / p95 | 198.0 / 617.3 ms | 177.1 / 690.6 ms | p50 -10.5%; p95 +11.9% |
| Tracking HTTP p50 / p95 | 153.5 / 178.1 ms | 150.2 / 174.6 ms | p50 -2.1%; p95 -1.9% |
| Camera to last 3D p50 / p95 | 452.6 / 1,072.7 ms | 457.9 / 820.6 ms | p95 -23.5% |
| Complete-publication latency slope | -32.6 ms/source-s | -11.2 ms/source-s | no stale queue |
| `detections.jsonl` bytes | 94,741,692 | 4,808,514 | -94.92% |
| Live raw / transmitted points | 1,558,802 / 1,558,802 | 1,487,952 / 78,358 | voxel run transmitted -94.73% |
| Detector publication p50 / p95 | not instrumented | 53.1 / 122.0 ms | diagnostic |

The condition made 93 detector calls and completed 92. It initialized once,
refreshed at accepted-frame indices 0, 30, 60, and 90, and used tracking for
the other 89 calls. The publication-tail target passed: p95 fell below 1 s and
was 23.5% lower than the dense control. The full detector p95 increased, but
that interval ends before the point-cloud JSON publication work; the end-to-end
metric targeted by T04 improved substantially.

#### Pipeline output and quality limitation

| Output metric | T03b dense control | T04 voxel01 |
| --- | ---: | ---: |
| Non-empty 3D source frames | 44 | 32 |
| 3D detection messages | 79 | 67 |
| Per-label messages | table 54, book 15, keyboard 10 | table 42, book 15, keyboard 10 |
| Final graph nodes / edges | 5 / 5 | 7 / 8 |
| Final graph categories | table 1, book 2, keyboard 2 | table 2, book 3, keyboard 2 |

The pre-registered requirement of at least 40 non-empty source frames failed.
The voxel transform occurs after mask inference and 3D projection, so it cannot
change which masks the GI runtime returns. Because the optimized run completes
faster under the latest-frame-wins callback, however, it accepts a different
subset of camera timestamps and therefore follows a different tracker path.
The unchanged book and keyboard counts, lower table count, and richer final
graph are consistent with sampling-path variation, but do not prove that the
quality difference is harmless. A fixed-frame replay or paired source-timestamp
run is required before merging this optimization.

#### Hardware and capacity

| Resource metric | T03b dense control | T04 voxel01 |
| --- | ---: | ---: |
| Mean / p95 GPU utilization | 33.7% / 96.0% | 33.8% / 95.2% |
| Mean GPU power | 15.37 W | 15.83 W |
| Mean system power | 34.70 W | 37.36 W |
| Maximum GPU temperature | 47 C | 45 C |
| Minimum Linux `MemAvailable` | 82.63 GiB | 82.71 GiB |
| Mean Docker working set | 11.09 GiB | 11.02 GiB |
| Mean NVIDIA process memory | 8.73 GiB | 8.70 GiB |
| Mean summed container CPU | 128.6% | 111.2% |

Voxel transport did not materially change resident model memory or peak GPU
pressure. It lowered mean container CPU while preserving more than 82 GiB of
unified-memory headroom. Additional onboard models must still be scheduled
around approximately 95% GPU-utilization bursts even though memory capacity is
comfortable.

#### Gate decision

| Gate | Result |
| --- | --- |
| Complete-publication p95 below 1.0 s | Pass: 820.6 ms |
| At least 10% p95 improvement | Pass: 23.5% |
| JSON bytes at least 50% lower | Pass: 94.92% |
| Transmitted points at least 50% lower | Pass: 94.73% |
| Exact offline cells and mean error at most `1e-5` m | Pass: all exact, 0.0 m |
| At least 87 completed full frames | Pass: 92 |
| Exact R30 cadence | Pass: 0, 30, 60, 90 |
| At least 40 non-empty 3D source frames | **Fail: 32** |
| Non-empty graph and no traceback | Pass: 7 nodes / 8 edges; no traceback |
| Below 80 C and at least 32 GiB available | Pass: 45 C; 82.71 GiB |

T04 is a strict partial pass: every recorded check except the non-empty-frame
gate passed, including all intended transport, geometry, latency, and resource
checks. The code stays opt-in and development-only until the non-empty-frame
difference is resolved with paired input frames. The result does establish that
dense per-pixel JSON transport is a major avoidable pipeline cost.

Raw condition, geometry check, generated report, and figures:

```text
/mnt/nas/danny/thor-scene-graph/run-artifacts/gi-scene-graph-t04-20260808/
```

## T05: Batched Multi-Text GI Grounding

### Question and hypothesis

The delivered GI application currently encodes each configured text label
separately and calls `det.forward_grounding` once per label. It shares the
high-resolution image features, but five labels still produce five serialized
grounding-head invocations. Original SAM3.1 instead maps all text IDs to one
image and performs one batched grounding call.

T05 tests whether the GI detector supports the same native multi-text shape.
The hypothesis is that replacing five Python-level grounding invocations with
one batched invocation will remove enough repeated decoder and synchronization
work for GI every-frame detection to beat the existing Original O5 operating
point. The TensorRT image-trunk engines remain unchanged because their batch is
one image; only the downstream text/grounding batch changes.

### Allowed change and isolation

- Copy the T02 headless Python overlay to a new T05 overlay on restricted NAS.
- Add an opt-in `batched` text-grounding mode while preserving `sequential` as
  the default and fallback.
- In batched mode, call `forward_text(labels)` once, construct a `FindStage`
  with `img_ids=zeros(N)` and `text_ids=arange(N)`, request `N` dummy prompts,
  and call `forward_grounding` once.
- Parse the leading prompt dimension independently, preserving the existing
  per-label mask-IoU NMS, cross-label NMS, thresholds, labels, scores, and
  `max_objects` behavior.
- Do not modify the stable Scene Graph checkout, Docker image layers, supplied
  TensorRT engines, GI checkpoints, or Meta SAM source.
- Keep the proprietary overlay and all runtime artifacts off GitHub. Git tracks
  only our experiment design, benchmark/report tooling, and lightweight tests.

This is source-level evaluation under the supplied non-production license. It
does not decompile or reverse engineer a TensorRT engine.

### Fixed input and conditions

Reuse T01's immutable 100-frame Lifestyle Lab input, frame order, JPEG bytes,
timestamps, and SHA-256 manifest. Reuse the fixed prompt order:

```text
keyboard
table
book
computer desk
stool
```

Use GI threshold `0.5`, detection cadence R1, tracking input 768, detection
input 1152, FP16 TensorRT trunks, headless API output, and `max_objects=24`.
Run conditions serially from fresh containers so they never contend for the
GPU.

| ID | Detector | Prompt execution | Purpose |
| --- | --- | --- | --- |
| O5 | Original SAM3.1 | One native five-text batch per independent image | Fixed target and teacher from T01 |
| G5-S-R1 | GI T02 | Five sequential grounding calls per detection frame | Existing GI control from T01 |
| G5-B-R1 | GI T05 | One five-text grounding call per detection frame | Batched candidate |

Before the formal run, execute bounded shape/parity checks:

1. one prompt, one fixed frame: sequential versus batched;
2. five prompts, one fixed frame: validate output tensor shapes and labels;
3. five prompts, ten frames: screen latency, stability, and memory;
4. proceed to all 100 frames only after the candidate completes without a
   traceback, NaN, CUDA error, or missing label dimension.

Failed preflights remain under `failed-attempts/` and are not mixed into formal
metrics.

### Measurements

Speed:

- prompt encoding time, reported separately from per-frame warm latency;
- image backbone and high-resolution detector-feature time;
- grounding-only time and complete runtime `detect_ms`;
- HTTP/client latency and 100-frame wall time/effective FPS;
- initialization and first-frame latency, but no container/model startup in
  warm distributions;
- p50, p90, p95, mean, standard deviation, minimum, and maximum.

Output and quality:

- N=1 masks, labels, scores, and counts against sequential GI;
- per-label directed instance IoU and recall at IoU 0.5 against G5-S-R1;
- teacher agreement against O5, explicitly not ground-truth accuracy;
- non-empty-frame rate, detections per label, lost-object rate, and NMS output
  count;
- representative overlays/contact sheets from identical source frames.

Hardware and capacity:

- GPU utilization distribution and active-process GPU memory;
- Linux `MemAvailable`, Docker working set, and summed container CPU;
- GPU/system power and GPU temperature;
- peak allocated and reserved CUDA memory where the runtime exposes them.

### Pre-registered gates

T05's primary speed gate is deliberately end-to-end at the detector boundary:

```text
G5-B-R1 client-visible p50 < Original O5 p50 (498.2 ms)
```

It must also improve client-visible p50 by at least 20% relative to G5-S-R1,
complete all 100 frames without a restart, and use exactly one grounding call
per five-prompt detection frame. Startup and prompt-cache construction do not
count toward warm per-frame latency.

The quality gate requires N=1 output parity within normal FP16 numerical
variation, no missing prompt dimension at N=5, G5-B-R1 teacher recall at IoU
0.5 no more than `0.02` below G5-S-R1, and mean directed teacher IoU no more
than `0.02` below G5-S-R1. It must remain below 80 C and retain at least 32 GiB
of Linux unified-memory headroom.

If the GI text tower or grounding head rejects `N > 1`, record the exact shape
failure and stop before changing model weights or rebuilding engines. If native
batching works but misses the speed gate, profile the batched call before
attempting resolution, query-count, precision, or engine changes; those would
be separate pre-registered attempts rather than silent changes to T05.

### Planned artifact root

```text
/mnt/nas/danny/thor-scene-graph/run-artifacts/gi-batched-grounding-t05-20260811/
```

Planned layout:

```text
input/                  # reference or manifest link to immutable T01 input
gi-sequential-r1/       # existing control metadata or rerun if required
gi-batched-r1/
report/
failed-attempts/
SHA256SUMS
```

### Results

The implementation and measurements used
overlay SHA-256
`fa872de0550d6227db800c1410cc98cea4935886d0700222667cf1e8da86da30`.
N=1 and N=5 both completed without a prompt-dimension, CUDA, or TensorRT
failure. The warmed N=5 single-frame check returned the same four semantic
instances expected on the first T01 frame: two `table`, one `keyboard`, and one
`book`.

The pre-registered ten-frame screening run then measured:

| Metric | G5-B-R1 preliminary value |
| --- | ---: |
| Frames | 10 |
| Runtime `detect_ms` mean / p50 | 424.8 / 423.8 ms |
| Runtime `process_ms` mean / p50 | 590.0 / 581.3 ms |
| Client-visible mean / p50 | 619.9 / 602.9 ms |

The batched detector calculation itself is already below Original O5's
498.2 ms client-visible p50, but the complete GI client path is not. Inspection
shows that R1 still computes the 768-input tracking backbone, propagates prior
tracker state, and consolidates detection masks into tracker state on every
frame. Those operations are useful for R30, but redundant for an independent
every-frame detector comparison. T05 therefore continues through formal parity
measurement; T06 separately tests removal of this identified non-grounding
work rather than silently expanding T05.

The formal 100-frame T05 condition completed with exactly one five-text
grounding call per accepted frame:

| Metric | GI sequential R1 | GI batched+tracker R1 | Change |
| --- | ---: | ---: | ---: |
| Client p50 | 818.5 ms | 607.3 ms | -25.8% |
| Client p95 | 852.8 ms | 642.5 ms | -24.7% |
| Runtime detect p50 | 579.3 ms | 420.1 ms | -27.5% |
| Mask observations | 597 | 597 | unchanged |
| Original-teacher mIoU | 0.6014 | 0.6012 | -0.0002 |
| Original-teacher recall at IoU 0.5 | 0.6231 | 0.6231 | unchanged |

The N=1 sequential and batched prediction archives were byte-identical. On
the N=5 first-frame parity check, labels and lost flags were identical, maximum
score difference was `0.0015922`, mean mask IoU was `0.9999782`, and only two
pixels differed across four masks.

T05 therefore proves that native prompt batching is correct and materially
faster, but it fails its primary end-to-end target because 607.3 ms is still
21.9% above Original O5's 498.2 ms p50. This result authorized T06 without
changing T05's gate after measurement.

Formal T05 artifacts and checksums:

```text
/mnt/nas/danny/thor-scene-graph/run-artifacts/gi-batched-grounding-t05-20260811/
```

## T06: R1 Detector-Only Fast Path

### Question and hypothesis

T05's ten-frame screen reduced five prompt grounding to one call and reached a
423.8 ms `detect_ms` p50, but client p50 remained 602.9 ms. T06 asks whether a
strict detector-only operating mode can remove the remaining redundant tracker
work and make the complete GI request faster than Original O5.

The hypothesis is that a detector invoked on every frame does not need to run
the 768-input tracking backbone, propagate old objects, initialize a tracker,
or re-anchor state. The 1152-input TensorRT detector features and one batched
grounding call are sufficient to return the current frame's masks, labels, and
scores.

### Allowed change and isolation

- Add an opt-in `--detector-only` mode to a new restricted T06 overlay derived
  from the checksum-addressed T05 overlay.
- Require `--text-grounding-mode batched` and `--detect-every 1` in this mode.
- Transform the input once and run only the 1152 detector trunk plus batched
  grounding; do not compute the 768 tracker trunk or create/propagate state.
- Apply the same detector threshold, per-label NMS, cross-label NMS,
  `max_objects` cap, and detector-mask-to-runtime-mask resampling used when T05
  seeds a new object.
- Publish deterministic frame-local IDs and `lost=false`. IDs are explicitly
  not temporal identities in detector-only mode.
- Preserve the existing packed-mask HTTP schema so Scene Graph's stateless
  detector client requires no change.
- Keep T02/T05 overlays, supplied engines, checkpoints, Docker image layers,
  and stable Scene Graph unchanged.

### Conditions and fixed input

Use the same immutable T01 input, five prompts, threshold, 1152 detection
resolution, FP16 engine, headless API, and serial execution as T05.

| ID | Mode | Purpose |
| --- | --- | --- |
| O5 | Original SAM3.1 independent per-frame batch | Fixed target and teacher |
| G5-B-R1 | T05 batched grounding with tracker path | Isolate removed tracker cost |
| G5-B-D1 | T06 batched detector-only path | Candidate complete request |

Run one-frame output checks first, then ten frames. Proceed to the formal
100-frame condition only if client-visible p50 is below 498.2 ms and the output
schema, labels, masks, scores, and frame-to-sequence association remain valid.

### Measurements and gates

Record the same latency, output/teacher-agreement, GPU, memory, CPU, power, and
temperature fields as T05. In addition, report the time removed by skipping
the tracker backbone/state path and verify every frame reports detector-only
mode with zero retained tracker states.

T06 passes only if:

1. 100-frame client-visible p50 is below Original O5's 498.2 ms and at least
   20% below G5-S-R1;
2. all frames complete without restart, stale sequence, traceback, or NaN;
3. teacher recall at IoU 0.5 and mean directed teacher IoU are each no more
   than `0.02` below the formal T05 batched condition;
4. the packed-mask API contract is unchanged and IDs are documented as
   frame-local;
5. maximum temperature stays below 80 C and Linux `MemAvailable` stays above
   32 GiB.

If the ten-frame speed screen misses 498.2 ms, stop and profile the 1152 trunk,
grounding decoder, mask head, and HTTP boundary. Resolution, query count,
precision, or TensorRT compilation changes require a new attempt and new
quality gates.

### Planned artifact root

```text
/mnt/nas/danny/thor-scene-graph/run-artifacts/gi-detector-only-t06-20260811/
```

### Results

T06 completed its ten-frame screen and formal 100-frame condition with overlay
SHA-256
`c63ff487120afe700ff603083aae8c4d4342e16baf6ed4527d614bfdd74a3c09`.
Every frame reported `backend=detector_only`, retained zero tracker states, and
preserved the packed-mask API schema.

#### Speed

| Metric | Original O5 | T05 batched+tracker | T06 detector-only |
| --- | ---: | ---: | ---: |
| Client mean | 506.4 ms | 617.5 ms | 460.5 ms |
| Client p50 | 498.2 ms | 607.3 ms | 459.8 ms |
| Client p95 | 506.4 ms | 642.5 ms | 473.9 ms |
| Runtime detect p50 | 497.9 ms | 420.1 ms | 435.0 ms |

T06 is 7.7% lower latency than Original by client p50 and 43.8% lower than the
original sequential GI R1 condition. The detector-only path removed 147.5 ms
from T05's client p50 by skipping the 768 tracking backbone, propagation, and
state consolidation. Its runtime detect p50 is slightly higher than T05
because T06 consistently uses the intended 1152 detector trunk; the tracker
path's effective detector resolution depends on its current backend state.

Benchmark NPZ writing is excluded from `total_ms`. The 101.7 s sequence wall
time includes writing large packed masks to NAS and is therefore not a detector
throughput metric.

#### Quality

| Directed comparison | Instance mIoU | Recall at IoU 0.5 | Mask observations |
| --- | ---: | ---: | ---: |
| Original to GI sequential | 0.6014 | 0.6231 | 597 GI |
| Original to T05 batched+tracker | 0.6012 | 0.6231 | 597 GI |
| Original to T06 detector-only | 0.3542 | 0.3647 | 310 GI |
| T05 batched+tracker to T06 detector-only | 0.4591 | 0.4858 | 310 target |

Batching itself preserved teacher agreement. T06's quality loss comes from
removing temporal object retention: the detector-only response contains only
objects detected on the current frame, while the R1 tracker path continues to
publish previously initialized objects. T06 also emitted no `computer desk` or
`stool`, as did the GI tracker conditions, and had fewer `table`, `book`, and
`keyboard` observations than the retained-state conditions.

#### Hardware and capacity

| Resource metric | Original | GI sequential | T05 batched+tracker | T06 detector-only |
| --- | ---: | ---: | ---: | ---: |
| Mean GPU utilization | 89.8% | 64.0% | 57.6% | 42.7% |
| P95 GPU utilization | 97.0% | 96.9% | 97.0% | 97.0% |
| Mean GPU power | 32.95 W | 22.59 W | 15.43 W | 14.80 W |
| Mean system power | 74.70 W | 51.44 W | 40.87 W | 39.76 W |
| Maximum GPU temperature | 52 C | 49 C | 47 C | 45 C |
| Minimum `MemAvailable` | 87.18 GiB | 82.33 GiB | 84.63 GiB | 86.85 GiB |
| Mean container working set | 5.57 GiB | 11.41 GiB | 10.32 GiB | 8.70 GiB |
| Mean NVIDIA process memory | 6.02 GiB | 8.84 GiB | 10.12 GiB | 7.29 GiB |
| Mean container CPU | 27.5% | 58.9% | 16.6% | 14.8% |

T06 preserved substantial memory and thermal headroom for other onboard
models, although every condition still reached approximately 97% GPU
utilization bursts.

#### Decision

T06 passes both speed gates, completion, API, resource, and zero-state gates,
but fails both pre-registered teacher-agreement gates. It is a strict partial
pass and must not replace the current stateful integration. T07 tests a single
configuration variable before considering model-level changes.

Formal report, plots, raw profiles, predictions, telemetry, and checksums:

```text
/mnt/nas/danny/thor-scene-graph/run-artifacts/gi-detector-only-t06-20260811/
```

## T07: 768-Input Batched Detector-Only Screen

### Question and hypothesis

T06 consistently uses the supplied 1152-input detection trunk and is faster
than Original, but its per-frame teacher agreement is much lower than the
retained-state GI conditions. The first-frame observations also showed that
the runtime's 768 and 1152 detector feature paths produce materially different
scores and masks on this sequence.

T07 asks whether using the existing 768-input TensorRT trunk for the same
batched detector-only path can improve teacher agreement while further lowering
latency. This is a configuration experiment, not a code change or engine
rebuild.

### Fixed conditions and allowed change

- Reuse the T06 overlay, five prompts, threshold `0.5`, independent R1 frames,
  API schema, input manifest, and hardware sampling.
- Change only detection input from 1152 to 768. Tracking remains disabled.
- Reuse the existing checksum-addressed 768 FP16 TensorRT engine.
- Run one warm-up frame, then the same first ten source frames.
- Compare those ten outputs against the corresponding Original masks and T06
  1152 outputs before deciding whether to run 100 frames.

### Screen gates

Proceed to 100 frames only if the warmed ten-frame condition:

1. completes with zero tracker states and no error;
2. keeps client p50 below Original's 498.2 ms;
3. improves or preserves both teacher mIoU and recall at IoU 0.5 relative to
   T06 on the same ten frames; and
4. does not increase GPU memory or temperature beyond T06's formal limits.

The formal quality target remains to close the gap to T05: teacher mIoU and
recall may be at most `0.02` below T05. If 768 does not materially improve
quality, stop rather than sweeping resolutions after observing results.

### Planned artifact root

```text
/mnt/nas/danny/thor-scene-graph/run-artifacts/gi-detector-only-t07-in768-20260811/
```

### Results

T07 used the unchanged T06 overlay and the existing 768 FP16 engine. A separate
warm-up frame was excluded, after which all ten fixed frames completed with
zero tracker states and no runtime error.

| Ten-frame metric | T06 1152 | T07 768 | Change |
| --- | ---: | ---: | ---: |
| Client p50 | 467.4 ms | 394.0 ms | -15.7% |
| Runtime detect p50 | 439.0 ms | 371.9 ms | -15.3% |
| Mask observations | 48 | 29 | -39.6% |
| Original-teacher mIoU | 0.3989 | 0.3463 | -0.0526 |
| Original-teacher recall at IoU 0.5 | 0.4800 | 0.4400 | -0.0400 |

T07 passed the speed, completion, and zero-state gates but failed both
pre-registered quality-improvement gates. The lower resolution was 20.9%
faster than Original's formal p50, but it removed detections and made teacher
agreement worse. The experiment therefore stopped after ten frames as planned;
no 100-frame T07 result exists and the configuration is not a candidate for
integration.

Screen profiles, predictions, telemetry, runtime log, and input subset:

```text
/mnt/nas/danny/thor-scene-graph/run-artifacts/gi-detector-only-t07-in768-20260811/
```

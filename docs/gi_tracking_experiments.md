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
| T01 | Does GI become useful when five prompts are initialized once and tracked, with grounding only every 30 frames? | Designed; execution pending | Pending |
| T02 | Can shared memory or a Unix socket remove the HTTP/JPEG/NPZ cost after T01 proves model-level value? | Blocked on T01 | Do not implement before T01 |
| T03 | Can Scene Graph schedule high-frequency known-object tracking and low-frequency category discovery? | Blocked on T01 | Do not implement before T01 |

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

Pending execution.

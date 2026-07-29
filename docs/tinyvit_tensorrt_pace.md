# Distilled SAM2 TinyViT TensorRT encoder smoke on PACE

## Result

PACE confirms that TensorRT 11.1 can replace the SAM2.1-L image encoder with all
three local distilled TinyViT checkpoints. Each checkpoint loaded exactly, exported to
ONNX, built as a TensorRT engine, produced the required SAM2 feature shapes, and passed
the feature-parity gate.

This began as an **encoder compatibility and performance smoke**. A subsequent image-mode
test connected each encoder to the official SAM2.1-L prompt and mask decoder and measured
PyTorch-versus-TensorRT mask agreement. It is still not a dataset accuracy result because
the local camera video has no ground-truth masks. The official downstream checkpoint is
`checkpoints/sam2/sam2.1_hiera_large.pt`; its SHA256 is
`2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`.

## Formal run

- Date: 2026-07-22
- Slurm array: `11366469_[0-2]`
- Account: `gts-agarg35-ideas_l40s`
- Partition/GPU: `gpu-l40s`, one NVIDIA L40S per task
- QOS: `embers`
- Precision: FP32, TensorRT TF32 disabled
- Input: one `1x3x1024x1024` random tensor, seed `20260722`
- Timing: 20 warmup iterations and 100 measured iterations
- Software: PyTorch `2.10.0+cu128`, ONNX `1.22.0`, TensorRT `11.1.0.106`
- Reports: `results/pace/tinyvit_trt_encoder/11366469/{tv5,tv11,tv21}/report.json`
- Logs: `logs/tinyvit-trt-11366469_{0,1,2}.out`

| Checkpoint | Parameters | PyTorch eager mean | TensorRT mean | TensorRT FPS | Relative speed | Latency change | Engine size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tv5.pt` | 5,122,676 | 6.520 ms | 3.979 ms | 251.3 | 1.639x | -39.0% | 16.7 MB |
| `tv11.pt` | 10,623,204 | 7.929 ms | 5.061 ms | 197.6 | 1.567x | -36.2% | 31.9 MB |
| `tv21.pt` | 20,803,984 | 15.813 ms | 18.504 ms | 54.0 | 0.855x | +17.0% | 355.9 MB |

The latency numbers cover only the image encoder. They exclude image capture,
normalization, host/device transfer, SAM2 prompt/mask/memory stages, and ROS publication.

## Optimization follow-up

The L40S optimization matrix used TensorRT builder optimization level 5 and explicitly
disabled TF32 in the PyTorch FP32 oracle. The main array was `11367310_[0-13]`; additional
legacy-export rows were `11368507_[14-17]`, the repaired TinyViT-5M BF16 row was
`11367935_3`, and the compact TinyViT-21M TF32 row was `11368732_18`.

Three candidate tiers are retained. `Accuracy-first` means FP32 feature parity is nearly
exact, not that final masks have already passed. `Balanced` permits TensorRT TF32.
`Speed-first` uses FP16 and must pass the downstream mask gate before deployment.

| Encoder | Accuracy-first FP32 | Balanced TF32 | Speed-first FP16 |
| --- | --- | --- | --- |
| TinyViT-5M | dynamo, 4.015 ms, 16.9 MB | dynamo, 3.124 ms, 19.6 MB | **dynamo, 1.274 ms, 785.2 FPS, 10.3 MB** |
| TinyViT-11M | legacy, 5.054 ms, 25.1 MB | dynamo, 3.705 ms, 30.0 MB | **dynamo, 1.394 ms, 717.2 FPS, 18.0 MB** |
| TinyViT-21M | legacy, 17.161 ms, 64.7 MB | legacy, 14.919 ms, 64.9 MB | **legacy, 2.855 ms, 350.3 FPS, 34.3 MB** |

Relative to the original FP32 TensorRT smoke, the selected speed-first candidates are:

| Encoder | Original TensorRT | Optimized TensorRT | Latency reduction | Relative throughput |
| --- | ---: | ---: | ---: | ---: |
| TinyViT-5M | 3.979 ms | 1.274 ms | 68.0% | 3.12x |
| TinyViT-11M | 5.061 ms | 1.394 ms | 72.4% | 3.63x |
| TinyViT-21M | 18.504 ms | 2.855 ms | 84.6% | 6.48x |

FP16 parity against the same-dtype PyTorch encoder remained inside the smoke gate:

| Encoder | Minimum cosine | Maximum relative L2 |
| --- | ---: | ---: |
| TinyViT-5M | 0.99998701 | 0.00508929 |
| TinyViT-11M | 0.99998808 | 0.00488760 |
| TinyViT-21M | 0.99999058 | 0.00436592 |

BF16 was tested after repairing PyTorch Dynamo's mixed-type Conv+BN-folded ONNX
initializers. It was consistently slower than FP16 and had larger feature error, so it is
not a preferred candidate. The repair remains in the smoke runner so future BF16 tests
fail or pass on actual TensorRT behavior rather than an invalid mixed-type graph.

### Layer-wise mixed-precision search

TensorRT 11.1 no longer exposes the legacy `BuilderFlag.FP16` or precision-constraint
builder flags. The layer search therefore uses a strongly typed ONNX graph rather than
silently ignored builder hints:

1. Run and export an FP32 oracle.
2. Rebuild the student before FP16 export so TinyViT attention-bias caches are created in
   FP16 rather than retaining values cached during the FP32 oracle run.
3. Restore the original FP32 initializers for selected nodes from the matching FP32 ONNX.
4. Insert explicit FP16-to-FP32 and FP32-to-FP16 casts around each connected precision
   island. Adjacent selected nodes share one island where graph topology permits it.
5. Build a strongly typed TensorRT engine and compare its outputs directly with the FP32
   PyTorch oracle.

The search profiles are `projection_fp32`, `norm_fp32`, `matmul_fp32`, `conv_fp32`,
`matmul_projection_fp32`, and `conv_matmul_fp32`, plus the FP16 baseline. The projection
profile selects the three final convolutions that produce `high_res_s0`, `high_res_s1`,
and `image_embedding`. The matrix maps seven profiles per model and uses Dynamo for 5M
and 11M while retaining the compact legacy exporter for 21M.

An initial implementation attempt using weak typing failed immediately because the FP16
builder flag has been removed in TensorRT 11.1. A subsequent Cast-island smoke exposed
two implementation issues before the formal sweep: a cached FP32 attention bias in the
FP16 export model and a missing ONNX save after graph rewriting. Both paths were fixed;
a synthetic MatMul graph now verifies that an FP16 weight and its original FP32 copy are
present, the selected MatMul consumes the FP32 copy, and the transformed model passes the
ONNX checker. Results from jobs before that fix are development diagnostics and must not
be used as mixed-precision measurements.

The L40S baseline from `11370104_0` completed at 1.2705 ms / 787.1 FPS and measured
maximum relative L2 `0.005046` against the FP32 oracle. The valid second-round 5M rows
ran sequentially on the same L40S node in `11371234_[1-4]`:

| Profile | Latency | Max relative L2 | FP32 nodes | FP32 initializer data | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| FP16 baseline | 1.3659 ms | 0.005041 | 0 | 0 | speed reference |
| output projections FP32 | 1.4134 ms | 0.004999 | 3 | 0.21 MB | dominated |
| LayerNorm FP32 | 1.2304 ms | 0.005078 | 16 | 0.02 MB | no accuracy gain; timing needs paired rerun |
| all Conv FP32 | 2.8488 ms | 0.003062 | 25 | 0.91 MB | 39% lower error, 109% slower |

Projection and LayerNorm precision do not explain the FP16 error. Convolution weights
are sensitive, but promoting all convolutions is too expensive; the next useful search
is stage-wise Conv promotion. The apparent LayerNorm speed gain is not accepted because
the error is worse and L40S node/DVFS variation has already moved the same FP16 engine by
about seven percent between runs.

The first all-MatMul row failed because Dynamo ONNX contains Int64 shape-calculation
MatMul nodes. A first filter then exposed that anonymous `val_*` initializer names are
not stable between separate FP32 and FP16 exports. The corrected matcher now requires
all known MatMul inputs to be floating point and pairs initializers by node name, input
position, operator, dtype, and shape rather than initializer name. Job `11372229_3`
completed at 1.1928 ms with maximum relative L2 `0.004693`, but its report confirms zero
FP32 initializers were restored. It therefore tests FP32 MatMul accumulation with FP16
weights, not a complete FP32 MatMul island, and is not promoted as a final candidate.
Final Pareto latency comparisons must use paired L40S runs and will ultimately be rerun
on Thor.

### FP8/INT8 post-training quantization search

The speed-first search uses NVIDIA ModelOpt 0.45 to insert explicit Q/DQ nodes into the
FP16 ONNX graph. Calibration uses 32 evenly spaced, SAM2-normalized frames from
`videos/test1.mov`; TensorRT 11.1 then builds the strongly typed engine. Install the
optional dependencies with `python -m pip install -r requirements-quantization.txt`.

The first TinyViT-5M sweep quantized every Conv and MatMul. Its encoder-only timing was
promising, but final mask agreement ruled it out:

| Profile | Isolated L40S latency | Mean mask IoU | Minimum mask IoU | Decision |
| --- | ---: | ---: | ---: | --- |
| FP16 | 1.2736 ms | 0.9793 | 0.9402 | current speed baseline |
| FP8, all Conv + MatMul | 1.1592 ms | 0.5335 | 0.0749 | reject |
| FP8, all MatMul | 1.2432 ms | 0.7558 | 0.1566 | reject |
| FP8, all Conv | 1.2911 ms | not run | not run | reject from feature error and no speed gain |
| INT8, all Conv + MatMul | 1.2889 ms | not run | not run | reject from feature error and no speed gain |

The isolated full-FP8 row initially appeared about 9.0% faster, but job `11373832`
alternated all engines for 200 rounds on the same L40S and removed that clock-state
bias:

| Engine | Paired latency | Speed relative to FP16 |
| --- | ---: | ---: |
| FP16 | 1.2379 ms | baseline |
| FP8, all Conv + MatMul | 1.1974 ms | 3.38% faster |
| FP8, all MatMul | 1.2456 ms | 0.62% slower |
| FP8, all Conv | 1.3088 ms | 5.42% slower |

Full FP8 therefore gives only a small real speed gain and does **not** meet the allowed
five-percent accuracy envelope. Full INT8 is both slower and substantially less
accurate at the encoder outputs. These rows show why feature cosine alone is not an
acceptance metric: full FP8 retained cosine values near 0.98--0.99 while changing
thresholded SAM2 masks substantially. They also show that selective Q/DQ overhead can
outweigh lower-precision compute in this small encoder.

The next sweep partitions the graph by architectural role rather than arbitrary node
index: attention-score MatMul, other linear MatMul, backbone Conv, and the three neck
output Conv layers. The attention-only candidate quantizes eight MatMuls and reduces
maximum feature relative L2 from 9.47% for all-MatMul FP8 to 1.26%; its isolated timing
was 1.4883 ms, so it is retained only for mask diagnosis and not yet as a speed winner.
Jobs `11373770`, `11373772`, `11373773`, and `11373774` contain the L40S sweep. A paired,
alternating engine benchmark is used for final speed decisions because independent
L40S runs show material clock-state variation.

For the speed-first criterion, a quantized candidate passes this proxy only when mean
binary-mask IoU against the same-model PyTorch path is at least 0.95 over the 16 fixed
masks. Minimum IoU remains in reports as a diagnostic but is not the primary gate:
near-empty masks can change it drastically through only a few pixels. This proxy still
does not replace a labelled J&F/mIoU evaluation.

#### Semantic per-layer precision search

Dynamo ONNX retains `pkg.torch.onnx.name_scopes` metadata, so the search maps TensorRT
nodes back to stable TinyViT scopes such as `stages_2.blocks.4.attn.qkv` and
`stages_2.blocks.4.mlp.fc2`. Legacy ONNX node paths are normalized to the same scope
format for TinyViT-21M. Repeated `--quantization-scope-regex` arguments select the exact
Conv/MatMul layers assigned FP8; all unselected encoder layers stay FP16. The report
records both actual ONNX node names and semantic scopes, making the final precision map
auditable and reproducible.

The hierarchical search avoids building every possible engine combination:

1. Quantize one of 16 regions at a time: patch embedding, individual convolutional or
   transformer blocks, downsampling blocks, and individual SAM2 output projections.
2. Independently test layer roles across all transformer blocks: QKV, attention output
   projection, MLP FC1, MLP FC2, and local convolution.
3. Split only promising blocks into their seven individual Conv/MatMul operations.
4. Greedily combine layers in sensitivity order, rerunning the mean-mask-IoU 0.95 gate
   after each addition.
5. Pair the final candidates against FP16 on one GPU, then repeat the selected map for
   11M and 21M and rebuild it on Thor.

Job `11374488` is the 16-region TinyViT-5M sweep and `11374524` is the five-role sweep.
Both target the RTX PRO 6000 Blackwell partition with `embers`, which is architecturally
closer to Thor than L40S; final deployment numbers still require Thor-native engines.

On the first Blackwell attempt, all tasks failed during the FP32 PyTorch oracle with
`CUBLAS_STATUS_NOT_INITIALIZED` before export or quantization. The jobs loaded the CUDA
12.6 module while this Blackwell node needs a newer CUDA runtime. The sensitivity scripts
now accept `CUDA_MODULE`; job `11398195` retries one role with CUDA 12.9 before the full
array is resubmitted. CUDA 12.9 allowed the PyTorch oracle to run in retry `11398195`,
but ONNX Runtime calibration then failed in CUBLAS with `CUBLAS_STATUS_INVALID_VALUE`.
The Blackwell failure is now isolated to the ModelOpt/ORT CUDA calibration environment,
not TinyViT export or TensorRT support; H200 remains the working sensitivity platform.

The H200 role sweep `11374583` completed and quantized eight nodes per real FP8 row:

| Layers assigned FP8 | Maximum feature relative L2 | Isolated H200 latency | Encoder gate |
| --- | ---: | ---: | --- |
| attention QKV | 0.02918 | 1.2759 ms | fail |
| attention output projection | 0.04857 | 1.2114 ms | fail |
| MLP FC1 | 0.04370 | 1.2986 ms | fail |
| MLP FC2 | 0.05023 | 1.3172 ms | fail |
| local depthwise Conv | 0.00478 | 1.1864 ms | invalid FP8 row |

The local-Conv graph contained zero Q/DQ nodes: ModelOpt reported those convolutions as
unsupported and returned an FP16 graph. The runner now fails when explicitly selected
FP8/INT8 nodes produce zero Q/DQ nodes, preventing this false positive. Final mask jobs
`11398190`--`11398193` rejected all four genuinely quantized H200 engines:

| Layers assigned FP8 | Mean mask IoU | Minimum mask IoU |
| --- | ---: | ---: |
| attention QKV | 0.87145 | 0.51540 |
| attention output projection | 0.84257 | 0.48341 |
| MLP FC1 | 0.82209 | 0.46154 |
| MLP FC2 | 0.88217 | 0.66258 |

Quantizing all eight attention-score MatMuls produced mean mask IoU `0.93489` and minimum
IoU `0.80947` in job `11373913`. It is close but outside the declared mean-IoU 0.95 gate.
Job `11398216` therefore quantizes the two attention MatMuls in each transformer block
individually, allowing low-sensitivity blocks to be combined without quantizing all
eight blocks. All eight candidates produced non-finite `image_embedding` tensors and
were rejected. The initially low L2 values for the first two outputs were misleading
because the third output was non-finite. Mask jobs `11398410` and `11398411` confirmed
mean IoU `0.0` for the two stage-1 candidates. Paired H200 job `11398420` measured FP16
at 1.2078 ms, stage-1 block 0 FP8 at 1.5463 ms, and stage-1 block 1 FP8 at 1.3159 ms:
selective attention Q/DQ was both invalid and 28.0%/9.0% slower. Per-block attention FP8
is therefore closed as a candidate.

#### TinyViT-11M/21M selective FP8 follow-up

Jobs `11562237` and `11562238` repeat the semantic region sweep for TV11M
and TV21M on L40S. The feature-parity exit code remains deliberately strict,
so a task can return exit 2 while leaving a valid candidate engine for the
downstream mask gate. Independent timings are used only to nominate
candidates; job `11562920` alternates the nominated engines with FP16 for 200
rounds on one L40S.

| Model and FP8 region | FP16 paired latency | Candidate paired latency | Encoder speedup | Image-embedding cosine | Mean / minimum mask IoU | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| TV11M stage-2 block 4 | 1.4668 ms | 1.3775 ms | 6.48% | 0.999717 | 0.91143 / 0.38983 | reject accuracy |
| TV21M patch embedding | 2.7753 ms | 2.7443 ms | 1.13% | 0.999964 | 0.949893 / 0.32143 | below gate |

The downstream checks are jobs `11562891` and `11562765`. They use 16 frames,
one point and one box per frame, and require mean binary-mask IoU at least
0.95. The minimum remains diagnostic because threshold-sensitive near-empty
masks can dominate it. TV11M demonstrates again that cosine near one does not
guarantee mask agreement.

TV21M misses the mean gate by 0.000107 while providing only 1.13% encoder
speedup, which would save roughly 0.1--0.2 ms in the complete Thor pipeline.
It is not promoted by rounding. Calibration refinement job
`11562961_[0-3]` tests 64/128 calibration frames with max/entropy scales; the
H100 duplicate is `11562983_[0-3]`. The first 128-frame H100 task exposed a
sampling bug: the loader used the requested sample count as the video length
and sought past frame 121 of the 122-frame input. Commit `c4b10ca` samples
repeated positions within the real frame count instead. H100 retry
`11563371_1` runs the corrected max/128 case and `11563442_3` runs the
corrected entropy/128 case; pending array tasks read the corrected workspace
code when they start. The candidate will remain FP16 unless refinement passes
the declared gate and a paired speed check.

The TV21 attention-block H200 sweep `11562690_[0-7]` found that stage-1
blocks 0 and 1 both pass the strict encoder feature gate
(`image_embedding` relative L2 0.00286, cosine 0.999996). Their isolated
latencies were 2.1400 and 2.1507 ms, respectively. Downstream 16-frame mask
checks `11563448` and `11563449` are pending; neither block is a deployment
candidate until that gate and a same-GPU alternating FP16 speed comparison
pass.

Auxiliary-stream jobs show that more streams do not improve TinyViT-5M on L40S:

| Maximum auxiliary streams | L40S latency | A100 latency |
| ---: | ---: | ---: |
| 0 | **1.2682 ms** | 1.9879 ms |
| 1 | 1.2766 ms | **1.9611 ms** |
| 2 | 1.2787 ms | 2.0950 ms |

The alternating same-GPU comparison in job `11398218` confirmed 1.2627, 1.2688, and
1.2684 ms for zero, one, and two auxiliary streams respectively. Zero streams is the
selected L40S setting, providing a repeatable 0.45--0.48% gain without changing model
numerics.

The first INT4 AWQ attempts failed because ModelOpt expects an iterable calibration
reader. After adding iterable support, job `11398217` completed: its 7.58 MB engine ran
at 1.3289 ms on L40S, while image-embedding relative L2 reached 0.779 and cosine fell to
0.676. INT4 is therefore rejected without a downstream mask run.

### TinyViT-21M graph fix

The original Dynamo ONNX contained six expanded attention-bias cache initializers, each
`[12,1024,1024]` FP32 (about 50.3 MB). Together they accounted for about 302 MB of the
355.2 MB graph. The legacy tracing exporter preserves compact relative-bias indexing:

| TinyViT-21M graph | ONNX | Engine | FP32 latency | FP16 latency |
| --- | ---: | ---: | ---: | ---: |
| Dynamo | 355.2 MB FP32 / 178.3 MB FP16 | 356.0 MB / 181.3 MB | 18.522 ms | 2.859 ms |
| Legacy | 58.1 MB FP32 / 33.7 MB FP16 | 64.7 MB / 34.3 MB | 17.161 ms | 2.855 ms |

Legacy export is therefore selected for TinyViT-21M. For 5M/11M, legacy export made the
engines smaller but slowed FP16 by about 8–11%, so Dynamo remains the speed choice.

## Downstream mask parity follow-up

The parity runner uses eight evenly spaced frames from `videos/test1.mov`. For every
frame it runs one center-point prompt and one centered box prompt through the same
official SAM2.1-L prompt/mask decoder, first with the PyTorch student encoder and then
with its TensorRT replacement. Thus each row below summarizes 16 final masks per model
and precision. `Minimum` is deliberately strict and sensitive to single boundary pixels;
`mean` is the average agreement over all masks.

| Encoder | Precision | Minimum mask IoU | Mean mask IoU | Job |
| --- | --- | ---: | ---: | --- |
| TinyViT-5M | FP16 | 0.940239 | 0.979325 | `11368873_0` |
| TinyViT-5M | TF32 | 0.950000 | 0.987043 | `11368954_0` |
| TinyViT-5M | FP32 | **0.992063** | **0.997874** | `11368970_0` |
| TinyViT-11M | FP16 | 0.818182 | 0.972948 | `11368874_1` |
| TinyViT-11M | TF32 | **0.958619** | **0.994160** | `11368940_1` |
| TinyViT-11M | FP32 | 0.944444 | 0.993486 | `11368974_1` |
| TinyViT-21M | FP16 | 0.971706 | 0.989817 | `11368875_2` |
| TinyViT-21M | TF32 | 0.982353 | 0.995064 | `11368941_2` |
| TinyViT-21M | FP32 | **0.995930** | **0.998795** | `11368975_2` |

None passed the provisional `minimum IoU >= 0.999` bit-level agreement gate. This does
not demonstrate a segmentation-accuracy regression: the fixed prompts can produce
near-empty or threshold-sensitive masks, and even FP32 TensorRT does not execute every
operator in the same order as PyTorch. The non-monotonic 11M minimum (TF32 above FP32)
is a concrete example of why the minimum of 16 unlabelled masks is not an accuracy
metric. It does demonstrate that FP16 is not numerically identical at final-mask level.

Deployment choice is therefore explicit:

- Use FP16 only as the speed-first candidate; it needs a ground-truth SA-V/SA1B
  no-regression result before claiming preserved accuracy.
- Use TF32 as the balanced fallback if FP16 misses that dataset gate.
- Use FP32 as the accuracy-first TensorRT fallback. It has the smallest feature error,
  although exact binary-mask identity is neither achieved nor required for equal task
  accuracy.

The first combined array, `11368811_[0-2]`, did not instantiate tasks in the scheduler
and was canceled while still pending. The listed independent jobs all ran on L40S with
the `embers` QOS and produced reports under
`results/pace/tinyvit_trt_mask_parity/<job>/<precision>/<model>/report.json` (the initial
FP16 reports omit the precision directory).

## Compatibility and parity

The TensorRT outputs have the exact feature contract consumed by the SAM2.1-L downstream
path:

```text
high_res_s0     [1,  32, 256, 256]
high_res_s1     [1,  64, 128, 128]
image_embedding [1, 256,  64,  64]
```

The fixed SAM2 image-position tensor is supplied by the downstream model, as in the
existing `SAM2-TensorRT-Thor` exporter; it is not a learned output of TinyViT.

The smoke gate requires exact shapes, finite values, cosine similarity at least
`0.99999`, and relative L2 error at most `0.002` for every feature tensor. These limits
only detect export/runtime corruption. They do not replace dataset-level mask metrics.

| Checkpoint | Minimum cosine | Maximum relative L2 | Verdict |
| --- | ---: | ---: | --- |
| `tv5.pt` | 0.99999917 | 0.00131934 | pass |
| `tv11.pt` | 0.99999928 | 0.00125353 | pass |
| `tv21.pt` | 0.99999958 | 0.00096545 | pass |

Checkpoint identities:

| Checkpoint | SHA256 |
| --- | --- |
| `tv5.pt` | `cd442f19b67be084305ead07908a21a911d25c3980f5f67e4b568db4d88878cf` |
| `tv11.pt` | `62a467bf915f4cf6b3c142743b55d0d4564d09658ee52896a69bcbc0fe5c77ab` |
| `tv21.pt` | `da3a192cfd66aab4ed75fc9c3f804c84e3488540a905728ec2f319f5ab7a29fe` |

## Reproduce

Submit all three checkpoints:

```bash
cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/efficientsam3-benchmark
mkdir -p logs
sbatch scripts/pace_l40s_tinyvit_trt_encoder_smoke.sbatch
```

Run the FP32/TF32/FP16/BF16 and Dynamo/legacy optimization matrix with:

```bash
sbatch scripts/pace_l40s_tinyvit_trt_optimization_matrix.sbatch
```

Run the seven-profile layer-wise mixed-precision matrix with:

```bash
sbatch scripts/pace_l40s_tinyvit_trt_mixed_precision.sbatch
```

Run the semantic FP8 sensitivity screens with:

```bash
sbatch scripts/pace_gpu_tinyvit_trt_layer_sensitivity.sbatch
sbatch scripts/pace_gpu_tinyvit_trt_role_sensitivity.sbatch
```

For a hand-selected precision map, repeat the semantic selector; every selected layer is
FP8 and every other image-encoder layer remains FP16:

```bash
python scripts/pace_tinyvit_trt_encoder_smoke.py \
  --checkpoint checkpoints/distill/tv5.pt \
  --distill-root /path/to/SAM2-Distillation-Pipeline \
  --output-dir results/pace/tinyvit_trt_layer_sensitivity/manual \
  --precision fp32 \
  --quantization-mode fp8 \
  --calibration-video videos/test1.mov \
  --builder-optimization-level 5 \
  --quantization-scope-regex 'stages_2\.blocks\.0\.attn\.qkv$' \
  --quantization-scope-regex 'stages_2\.blocks\.3\.mlp\.fc1$'
```

Run the downstream SAM2-L mask-agreement smoke for one precision tier with:

```bash
sbatch --export=ALL,TIER=fp16 scripts/pace_l40s_tinyvit_trt_mask_parity.sbatch
sbatch --export=ALL,TIER=tf32 scripts/pace_l40s_tinyvit_trt_mask_parity.sbatch
sbatch --export=ALL,TIER=fp32 scripts/pace_l40s_tinyvit_trt_mask_parity.sbatch
```

The array maps tasks `0/1/2` to `tv5/tv11/tv21`. The job runs:

```bash
python scripts/pace_tinyvit_trt_encoder_smoke.py \
  --checkpoint checkpoints/distill/tv5.pt \
  --distill-root /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/SAM2-Distillation-Pipeline \
  --output-dir results/pace/tinyvit_trt_encoder/manual_tv5 \
  --precision fp32
```

ONNX files and TensorRT engines are generated under ignored `results/` directories and
must not be committed.

## End-to-end pipeline optimization

The encoder-only result does not predict camera-pipeline speed. Job `11405814` therefore
ran the original TensorRT path and the selected optimized path sequentially on the same
L40S. It used 32 consecutive 1080p frames after three decoder warmup frames, five model
warmups, one center-point and one centered-box mask per frame, and the TinyViT-5M FP16
zero-aux-stream engine. The timing includes OpenCV FFmpeg software decode, BGR-to-RGB,
preprocessing, host/device transfer, encoder, prompt encoder, mask decoder, full-resolution
mask postprocessing, and the requested CPU mask transfer. It excludes ROS transport,
publication, and overlay rendering.

The accuracy proxy compares every optimized binary mask with the same TinyViT-5M PyTorch
encoder and FP32 SAM2-L downstream path. The selected path passed the declared mean-IoU
gate:

| 32-frame result | Original TensorRT path | Optimized TensorRT path |
| --- | ---: | ---: |
| Mean mask IoU vs PyTorch | 0.979605 | 0.979536 |
| Minimum mask IoU | 0.856471 | 0.832941 |
| `set_image` | 8.227 ms | **2.372 ms** |
| Point prompt | 4.847 ms | **2.031 ms** |
| Model pipeline | 13.074 ms | **4.402 ms** |
| Model-only throughput | 76.5 FPS | **227.1 FPS** |
| Decode plus one point prompt | 21.553 ms | **12.963 ms** |
| Decode plus point throughput | 46.4 FPS | **77.1 FPS** |

The optimized model path is `2.97x` faster than the original TensorRT path and `4.31x`
faster than the PyTorch model path measured in the same optimized run. Including
steady-state software decode, it is `1.66x` faster than the original TensorRT path and
`2.12x` faster than PyTorch. The average mask-IoU change relative to the original
TensorRT path is less than `0.0001`; the strict single-mask minimum remains below 0.95
because threshold-sensitive, nearly empty masks are still present.

The selected changes are:

1. Upload the RGB `uint8` frame before resize and normalization, then run bilinear
   antialiased resize and ImageNet normalization on the GPU. This reduced `set_image`
   from 8.10 to 2.86 ms by itself in screening job `11405215`.
2. In image-predictor mode only, omit the three feature positional tensors whose values
   `SAM2ImagePredictor.set_image` discards. Their shapes are retained. This saved about
   0.42 ms and produced identical masks. Do **not** apply this shortcut to
   `SAM2VideoPredictor`; temporal memory attention consumes positional values.
3. Keep TensorRT encoder features in FP16 instead of materializing three FP32 copies,
   and run the prompt and mask decoder under FP16 autocast.
4. Compile the prompt and mask decoder with `torch.compile(mode="reduce-overhead",
   fullgraph=True, dynamic=False)`. Call
   `torch.compiler.cudagraph_mark_step_begin()` before each prompt so a subsequent
   point/box invocation does not overwrite a CUDA Graph output still referenced by the
   previous invocation.
5. Transfer one full-resolution binary mask rather than FP32 logits, scores, and
   low-resolution logits when those extra values are not consumed.
6. Cache normalized GPU prompt tensors while a point or box is unchanged.

These switches are explicit in `pace_tinyvit_trt_mask_parity.py`; the selected experiment
uses:

```text
--trt-position-mode shape-only
--trt-gpu-preprocess
--mask-transfer binary-only
--decoder-autocast-fp16
--trt-native-outputs
--compile-components reduce-overhead
--cache-prompts
```

### Real streaming throughput

The same formal job processed 64 frames with a point prompt through a real capture loop,
not by adding separately measured stage means:

| Stream execution | Original path | Optimized path |
| --- | ---: | ---: |
| Sequential decode and inference | 32.9 FPS | **88.9 FPS** |
| Producer-thread decode overlapped with inference | 30.1 FPS | **129.1 FPS** |

Overlap is useful only after GPU preprocessing removes CPU contention: it slowed the
original CPU-resize path but improved the optimized path by `1.45x`. The `129.1 FPS`
number is a PACE file-stream throughput result, not a Thor ROS camera claim. Thor should
replace software decode/copy with the JetPack GStreamer/NVDEC and zero-copy camera path,
then rerun the same stage and stream measurements with ROS publication and overlay
enabled.

### Screening and rejected variants

The main single-change sweep is `11405215_[0-9]`; refinement jobs are `11405486`,
`11405685`, and `11405733`. All use `embers`. The useful stage results are:

| Variant | `set_image` | Point prompt | Model pipeline | Decision |
| --- | ---: | ---: | ---: | --- |
| Original TensorRT | 8.098 ms | 4.776 ms | 12.874 ms | baseline |
| Skip unused image PE | 7.674 ms | 4.905 ms | 12.579 ms | keep |
| Binary-only CPU transfer | 8.107 ms | 4.298 ms | 12.404 ms | keep when logits are unused |
| GPU preprocess | 2.858 ms | 4.746 ms | 7.604 ms | keep |
| GPU preprocess + PE skip + binary | 2.410 ms | 4.371 ms | 6.780 ms | keep |
| Add compiled FP16 downstream | 2.367 ms | 2.174 ms | 4.541 ms | keep |
| Add cached GPU prompt | 2.365 ms | 2.017 ms | **4.382 ms** | selected |

Eager FP16 downstream execution was rejected: point-prompt latency increased from about
4.78 to 6.10 ms and it showed large `set_image` outliers. FP16 is useful here only after
the static prompt/mask graphs are compiled. FP32 compiled downstream reached 5.505 ms
for the otherwise optimized model pipeline, roughly 1.1 ms slower than compiled FP16.
`max-autotune` reached 4.507 ms versus 4.541 ms for `reduce-overhead`, which is below
cross-job noise and incurs much longer startup compilation; it is not selected.
FP16 resize/normalization reached 4.317 ms in an independent run, only about 1.5% below
the FP32-preprocess candidate while reducing the 16-frame minimum mask IoU from about
0.979 to 0.971. Same-GPU confirmation job `11405937` did not start under `embers` and
was canceled at zero runtime. Because the apparent gain is within cross-node variation
and its accuracy proxy is worse, FP32 GPU preprocessing remains selected.

The original, unoptimized end-to-end path also completed on A100 (`11401278`) and H200
(`11401276`). TensorRT model-pipeline speedups over PyTorch were `1.31x` and `1.39x`;
end-to-end speedups were `1.21x` and `1.20x`. The broader optimized A100/H200 arrays
`11405290` and `11405291` never received resources under `embers` and were canceled at
zero runtime after the L40S formal run converged. These are scheduling outcomes, not
TensorRT compatibility failures.

Reproduce the formal baseline and selected candidate on one L40S with:

```bash
sbatch \
  --export=ALL,ENGINE_PATH=results/pace/tinyvit_trt_aux_streams/11374004/tv5/aux-0/encoder.fp16.engine \
  scripts/pace_l40s_tinyvit_trt_pipeline_pair.sbatch
```

Reports are written below
`results/pace/tinyvit_trt_pipeline_sweep/<job>/{baseline,all-compile-cache}/report.json`.
Those reports and generated compilation/engine artifacts remain ignored and must not be
committed.

## Interpretation and next work

`tv5.pt` and `tv11.pt` use Dynamo export for maximum speed. `tv21.pt` uses legacy export
to remove expanded attention-bias caches. FP16 is the fastest encoder candidate for all
three, while FP32 remains the accuracy-first fallback.

Next gates, in order:

1. Run PyTorch and all three TensorRT tiers against the same official SA-V/SA1B
   ground-truth subset. Select the fastest tier whose J&F/mIoU delta is inside the
   predeclared no-regression tolerance.
2. Exercise video memory attention/tracking; the completed parity test covers image-mode
   prompt and mask decoding, not temporal propagation.
3. Rebuild and benchmark the accepted engines on Jetson Thor; PACE engines are not
   portable deployment artifacts.
4. If TinyViT-21M FP32/TF32 latency still matters, implement fused relative-position
   attention rather than re-expanding its six 32-by-32-window bias matrices.

## Calibration run

The initial array `11365904_[0-2]` successfully built and ran all three engines. Its first
feature gate used relative L2 `0.001`; TinyViT-5M and 11M exceeded that provisional limit
slightly despite cosine similarity above `0.999999`, so the tasks returned exit 2. The
formal gate was changed to the explicitly documented `0.002` compatibility threshold and
the complete array was rerun as `11366469`; no model/runtime code was changed to obtain
the passing formal results.

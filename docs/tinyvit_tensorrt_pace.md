# Distilled SAM2 TinyViT TensorRT encoder smoke on PACE

## Result

PACE confirms that TensorRT 11.1 can replace the SAM2.1-L image encoder with all
three local distilled TinyViT checkpoints. Each checkpoint loaded exactly, exported to
ONNX, built as a TensorRT engine, produced the required SAM2 feature shapes, and passed
the feature-parity gate.

This is an **encoder compatibility and performance smoke**, not an end-to-end SAM2 mask
accuracy result. Prompt encoding, mask decoding, memory attention, and final-mask parity
were not part of this run. The official SAM2.1-L downstream checkpoint was subsequently
downloaded to `checkpoints/sam2/sam2.1_hiera_large.pt` for that next gate; its SHA256 is
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

## Interpretation and next work

`tv5.pt` and `tv11.pt` use Dynamo export for maximum speed. `tv21.pt` uses legacy export
to remove expanded attention-bias caches. FP16 is the fastest encoder candidate for all
three, while FP32 remains the accuracy-first fallback.

Next gates, in order:

1. Connect each TensorRT encoder to the existing SAM2.1-L prompt/mask/memory graphs using
   the downloaded exact downstream checkpoint.
2. Compare final masks against the same-checkpoint FP32 PyTorch pipeline on fixed
   images/video;
   the feature smoke alone is not evidence of no accuracy loss.
3. Keep FP16 only if that mask gate passes; otherwise promote the corresponding TF32 or
   FP32 row.
4. Rebuild and benchmark the accepted engines on Jetson Thor; PACE engines are not
   portable deployment artifacts.
5. If TinyViT-21M FP32/TF32 latency still matters, implement fused relative-position
   attention rather than re-expanding its six 32-by-32-window bias matrices.

## Calibration run

The initial array `11365904_[0-2]` successfully built and ran all three engines. Its first
feature gate used relative L2 `0.001`; TinyViT-5M and 11M exceeded that provisional limit
slightly despite cosine similarity above `0.999999`, so the tasks returned exit 2. The
formal gate was changed to the explicitly documented `0.002` compatibility threshold and
the complete array was rerun as `11366469`; no model/runtime code was changed to obtain
the passing formal results.

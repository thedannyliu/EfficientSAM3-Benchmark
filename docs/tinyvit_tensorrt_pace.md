# Distilled SAM2 TinyViT TensorRT encoder smoke on PACE

## Result

PACE confirms that TensorRT 11.1 can replace the SAM2.1-L image encoder with all
three local distilled TinyViT checkpoints. Each checkpoint loaded exactly, exported to
ONNX, built as a TensorRT engine, produced the required SAM2 feature shapes, and passed
the feature-parity gate.

This is an **encoder compatibility and performance smoke**, not an end-to-end SAM2 mask
accuracy result. The PACE workspace does not currently contain the SAM2.1-L downstream
checkpoint, so prompt encoding, mask decoding, memory attention, and final-mask parity
were not part of this run.

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

`tv5.pt` and `tv11.pt` are immediately promising encoder replacements: FP32 TensorRT
reduced isolated encoder latency by 36–39% on L40S. `tv21.pt` is functionally supported,
but its exported ONNX is 355.2 MB and its engine is 355.9 MB, compared with 30.7/31.9 MB
for TinyViT-11M. That disproportionate graph size and its slower TensorRT result suggest
constant duplication or an inefficient lowering of the 21M model's attention/window
operations. It should be optimized before promotion.

Next gates, in order:

1. Provide the exact SAM2.1-L downstream checkpoint and connect each TensorRT encoder to
   the existing prompt/mask/memory graphs.
2. Compare final masks against the same-checkpoint PyTorch pipeline on fixed images/video;
   the feature smoke alone is not evidence of no accuracy loss.
3. Inspect the TinyViT-21M ONNX initializer/constant sizes and exported attention graph,
   then rerun FP32 before considering it performance-ready.
4. Evaluate FP16/BF16 only as separate candidates and keep them only if the full mask
   accuracy gate passes.
5. Rebuild and benchmark the accepted engines on Jetson Thor; PACE engines are not
   portable deployment artifacts.

## Calibration run

The initial array `11365904_[0-2]` successfully built and ran all three engines. Its first
feature gate used relative L2 `0.001`; TinyViT-5M and 11M exceeded that provisional limit
slightly despite cosine similarity above `0.999999`, so the tasks returned exit 2. The
formal gate was changed to the explicitly documented `0.002` compatibility threshold and
the complete array was rerun as `11366469`; no model/runtime code was changed to obtain
the passing formal results.

# Benchmark Dataset Protocol

This repo uses segmentation datasets as prompt/eval sources. The prompt is
derived from a fixed ground-truth object, and the prediction is evaluated
against that same object mask.

## Reproduce The Fixed Data

COCO image benchmark:

```bash
bash scripts/prepare_coco_fixed_subset.sh 10
```

For the default `count=10`, `COCO_SEED=20260527` case, this command also runs
the fixed prompt manifest generation.

Outputs:

- `data/manifests/coco_val2017_fixed10.jsonl`
- `data/coco/coco_val2017_fixed10_selection.json`

SA-V video tracking benchmark with official GT:

```bash
bash scripts/download_sav_valtest_subset.sh val 3
```

Outputs:

- `data/manifests/sav_val_fixed3.jsonl`
- `data/sa-v/sav_val_fixed3/official_subset_manifest.json`

Or prepare both datasets in one command:

```bash
bash scripts/prepare_benchmark_datasets.sh
```

Both scripts use seed `20260527` by default. Override with `COCO_SEED=...` or
`SAV_SEED=...` only when intentionally creating a different fixed subset.

The default SA-V selection policy is reproducible but mechanical:

```text
SAV_SELECTION_POLICY=largest_first_mask
```

It selects the largest official object mask in the first annotated frame of
each seeded video. This is valid for official IoU but may choose visually minor
objects such as wires or small signs, because SA-V val/test does not label
which objects are semantically important. For visually meaningful POC videos,
use the salient policy and extract more candidates before pruning back to the
fixed count:

```bash
bash scripts/prepare_sav_salient_subset.sh
```

This writes a separate dataset root and manifest:

```text
data/sa-v/sav_val_salient_fixed3/
data/manifests/sav_val_salient_fixed3.jsonl
data/manifests/sav_val_salient_fixed3_selection.json
overlays/sav/review/sav_val_salient_fixed3/contact_sheet.png
```

## Image Prompts On COCO

COCO is used because it provides both segmentation masks and category names.
For each fixed image:

1. Eligible annotations are non-crowd segmentations with area at least `1024`.
2. Images are shuffled with the fixed seed.
3. The target object is the largest eligible annotation in the image.
4. Text prompt is the target object's COCO category name, for example `cow`.
5. Point prompt is the centroid of the target object's segmentation mask.
6. IoU is computed against that exact selected annotation mask.

The fixed 10 text and point prompts were visually reviewed on 2026-05-27. The
tracked review record is `configs/datasets/coco_val2017_fixed10_prompts.json`.
It records each image id, annotation id, text prompt, positive point prompt, and
a short review note. Text prompts are single COCO category names for the
selected objects.

SAM3 and EfficientSAM3 run both text and point prompts through their native
SAM3-compatible image processors. SAM2, Efficient-SAM2, and EfficientTAM are
point-only in this image benchmark. MobileSAM is also point-only and uses the
same fixed COCO point prompts for the mobile image-segmentation baseline.

`sam-profile-coco` supports four evaluation modes:

- `both`: compute GT IoU metrics and write overlays when an overlay directory is supplied.
- `gt`: compute GT IoU metrics without overlay writing.
- `overlay`: write overlays without decoding GT masks.
- `profile`: collect latency, memory, parameters, and weight sizes only.

The profiling columns are shared across backends for comparison. SAM3 and
EfficientSAM3 populate image/text/transformer/geometry/segmentation and
interactive point-prompt components when those modules exist. SAM2,
Efficient-SAM2, and EfficientTAM populate image encoder, prompt encoder, mask
decoder, and memory components according to their native model objects.

## Video Prompts And Tracking On SA-V

Official SA-V val/test is used because it has frame directories and PNG masks:

```text
JPEGImages_24fps/<video_id>/*.jpg
Annotations_6fps/<video_id>/<object_id>/*.png
```

SA-V val/test masks do not include semantic category names. Therefore this repo
uses point prompts for SA-V tracking:

1. Videos are selected from `sav_val.txt` or `sav_test.txt` with the fixed seed.
2. For each video, the target is the largest object mask in the first available
   annotation frame, unless `SAV_SELECTION_POLICY=salient_first_mask` is used.
3. The prompt is a positive point at that mask centroid.
4. The model initializes a video state, receives the point prompt once, and then
   propagates masks through the video.
5. IoU is computed on annotated frames for that selected object.
6. Prediction PNGs and overlay MP4s are written so quantitative IoU can be
   checked against visual mask quality.

Text-prompt video tracking requires semantic labels or text descriptions for
the target object. Official SA-V val/test does not provide those labels, so a
text-prompt SA-V tracking score would need a separate, documented label source.
Using a generic prompt such as `object` would not be a meaningful text-prompt
benchmark.

Manual text prompts for a fixed SA-V object set are tracked separately:

```bash
sam-sav-text-prompts init \
  --manifest data/manifests/sav_val_fixed3.jsonl \
  --review-dir overlays/sav/review/current_fixed3 \
  --output configs/datasets/sav_val_fixed3_text_prompts.json
```

Edit `text_prompt` and `instance_hint` for each `sample_id` in that JSON after
viewing the matching `review_overlay`. `text_prompt` should be a short
model-facing noun phrase such as `person`; `instance_hint` should identify the
selected official GT object among same-class objects, such as `standing person
centered in the hallway`. Then merge the filled prompts into a text-enabled
manifest:

```bash
sam-sav-text-prompts apply \
  --manifest data/manifests/sav_val_fixed3.jsonl \
  --prompts configs/datasets/sav_val_fixed3_text_prompts.json \
  --output data/manifests/sav_val_fixed3_text.jsonl
```

Use the `_text.jsonl` manifest only for text-prompt/overlay POC runs or for
IoU experiments where the text prompt is explicitly documented as manual.
If multiple same-class objects appear in the frame, text-only top-1 localization
is not guaranteed to select the GT object. Report that as localization
ambiguity, not tracker failure. For later verification, keep two numbers:
top-1 text localization IoU against the selected GT object, and best-instance
GT-assisted IoU as a diagnostic upper bound for whether the text localizer found
the object somewhere among its returned instances.
`sam-profile-yoloe-edgetam` writes these as
`yoloe_initial_top1_gt_iou`, `yoloe_initial_best_gt_iou`,
`yoloe_initial_best_rank`, and `yoloe_initial_localization_note` in the summary
CSV.
For YOLOE+EdgeTAM, the text-enabled manifest can be passed directly:

```bash
sam-profile-yoloe-edgetam \
  --manifest data/manifests/sav_val_fixed3_text.jsonl \
  --csv-output results/yoloe_edgetam/manual_text/frames.csv \
  --summary-output results/yoloe_edgetam/manual_text/summary.json \
  --overlay-root overlays/yoloe_edgetam/manual_text \
  --work-dir results/yoloe_edgetam/manual_text/work
```

Current fixed3 note: the first seeded SA-V subset contains small annotated
targets. For example, `sav_018669` object `000` covers about `0.421%` of the
frame and has a very thin first-frame bbox. It is a valid official GT target,
but it is not a good visual-quality demo target.

## YOLOE-26M-seg + EdgeTAM Video POC

This POC is a separate text-prompt video tracking path:

```text
text prompt -> YOLOE-26M-seg localizer -> box/mask -> EdgeTAM tracker
            -> low-frequency YOLOE validation or re-grounding
```

Use recorded videos with a documented prompt, for example:

```bash
sam-profile-yoloe-edgetam \
  --video-path videos/test1.mov \
  --source-id test1 \
  --text-prompt person \
  --csv-output results/yoloe_edgetam/manual/frames.csv \
  --summary-output results/yoloe_edgetam/manual/summary.json \
  --overlay-root overlays/yoloe_edgetam/manual \
  --work-dir results/yoloe_edgetam/manual/work
```

For SA-V val/test, this path can produce overlays if the operator supplies a
valid text prompt for the selected video. It should not be reported as official
SA-V text-prompt IoU unless that prompt source is documented, because official
SA-V val/test masks do not provide semantic labels.

`sam-profile-sav-video` uses the same `--eval-mode` choices as COCO. In `both`
mode, overlay MP4s include prediction fills and GT contours on annotated frames;
in `overlay` mode, the video is prediction-only visual output.

## Current Fixed Samples

Current COCO fixed10 categories:

```text
cow, train, motorcycle, bird, person, bed, bicycle, zebra, elephant, sink
```

Current COCO fixed10 text prompts:

```text
coco_00_267434_74556 text="cow"
coco_01_217400_174678 text="train"
coco_02_179765_148486 text="motorcycle"
coco_03_31322_40228  text="bird"
coco_04_161879_452906 text="person"
coco_05_450488_317141 text="bed"
coco_06_174482_128189 text="bicycle"
coco_07_68933_589793 text="zebra"
coco_08_293300_583961 text="elephant"
coco_09_223789_1130149 text="sink"
```

Current COCO fixed10 point prompts:

```text
coco_00_267434_74556 cow        point=(189.963, 300.790)
coco_01_217400_174678 train      point=(309.457, 248.736)
coco_02_179765_148486 motorcycle point=(341.923, 246.026)
coco_03_31322_40228  bird       point=(439.702, 261.153)
coco_04_161879_452906 person     point=(145.616, 162.530)
coco_05_450488_317141 bed        point=(79.041, 386.284)
coco_06_174482_128189 bicycle    point=(346.457, 214.646)
coco_07_68933_589793 zebra      point=(380.113, 143.604)
coco_08_293300_583961 elephant   point=(344.107, 196.391)
coco_09_223789_1130149 sink       point=(133.505, 368.334)
```

Current SA-V val fixed3 video IDs:

```text
sav_018669
sav_023216
sav_018332
```

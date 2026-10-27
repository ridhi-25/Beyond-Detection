# Sprint 13 — FWB Architecture, Training, and Baseline Evaluation

## Status summary

**Sprint 13 is prepared but blocked before full architecture validation, pretrained transfer, training, and evaluation.** No training run was started and no FWB performance result exists. The stop was deliberate: this machine is CPU-only and had 0.97 GiB available RAM at the latest gate attempt (3.00 GiB conservative guard configured). A 50-epoch, 640-pixel run over 19,390 training images is not reasonable on this hardware.

The architecture gate's exact status and frozen-checkpoint hashes are in `results/sprint13_architecture_report.md` and `results/sprint13_architecture_metrics.json`. Independent FWB and parser/YAML tests passed, but they do not replace the required full 640×640 detector forward check.

## Objective and hypothesis

E1 is the first labeled optimization experiment: compare the frozen Sprint 5 B1 detector with the same YOLOv8n detector augmented by one Feature Weighting Block (FWB), trained from compatible pretrained YOLOv8n COCO weights. The hypothesis is that channel-wise feature reweighting after SPPF may improve detection, including pothole and traffic-cone detection. The hypothesis remains **untested**; no improvement is assumed.

## Frozen B1 and reference results

- Checkpoint: `results/previous_b1/weights/best.pt`
- SHA-256 before checks: `e8fecd7d2597090b815ac408cb1d4c783f908e56b65a588559db8b6b0afd685d`
- SHA-256 after the architecture gate: `e8fecd7d2597090b815ac408cb1d4c783f908e56b65a588559db8b6b0afd685d`
- Source of baseline metrics: recorded Sprint 5 held-out evaluation artifacts under `results/b1_baseline_eval/`, not training metrics.

| Sprint 5 held-out test metric | B1 |
|---|---:|
| Precision | 0.9006970973457251 |
| Recall | 0.8633117780342082 |
| mAP50 | 0.9285558753282753 |
| mAP50–95 | 0.7766256219114711 |
| Pothole AP50 | 0.6465218643921655 |
| Traffic cone AP50 | 0.8868331576002275 |

The B1 checkpoint, historical results, and B1 evaluation artifacts were not modified.

## Dataset and pretrained initialization

The current `models/detector/data.yaml` resolves to the existing merged dataset and split lists:

- Train: `data/splits/train.txt` — 19,390 images; every listed image path was present.
- Validation: `data/splits/val.txt` — 1,083 images; every listed image path was present.
- Held-out test: `data/splits/test.txt` — 750 images; every listed image path was present.
- Taxonomy: 15 classes, matching `data/merged/classes.txt`.

Recorded SHA-256 values at inspection:

- `data/merged/manifest.csv`: `092631b9c06c457f4318459ce900981852262432a272a82fa3f02ac9ab6cc065`
- `data/merged/classes.txt`: `6552e058091d78b9eaaa25a3951a7b80503f24cb5d6d84fd24a38d6e64448120`
- `data/splits/train.txt`: `3984ca5102e10a4804639eeb9871089e0232217e377d5ea32acc08b3af0a7437`
- `data/splits/val.txt`: `65b21b5e026e2ff37dc3cc59128b83c1f7337fbd6b014f983e33b57f835bf505`
- `data/splits/test.txt`: `5600a7f8f41f5586bdedd0dcc2cebf7656b1e5e3619f3b7c074e13f30d92e6fc`

No dataset or split files were changed. The local `yolov8n.pt` is present (6,549,796 bytes; SHA-256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`). Sprint 3 documented this filename as the official Ultralytics v8.4.0 release asset. No weights were downloaded or transferred in Sprint 13 because the mandatory full architecture gate did not pass.

## FWB definition and placement

The implemented module follows the specified equation without changing its operations:

`F = HSigmoid(LeakyReLU(Conv2d_1x1(AvgPool(f))))`

The weights are broadcast-multiplied with the input feature tensor. The block is intended to be the single layer immediately after SPPF and before the neck. It preserves the input channels and feature-map dimensions. Both activations use non-in-place operations because the specified in-place combination caused backward propagation to fail in testing; the mathematical equation is unchanged.

For the YOLOv8n SPPF width of 256 channels, the standalone FWB has 65,792 trainable parameters (65,536 convolution weights and 256 biases). This is the module's calculated parameter count, **not** a measured full-model parameter count or GFLOPs result. Whole-model parameters and FLOPs remain pending the full architecture check.

## Parser and YAML integration

The custom model definition is derived from the installed Ultralytics 8.4.160 `yolov8.yaml`. The project-local registration adapter adds FWB to that parser version's base-module channel-scaling branch at runtime; it does not edit `site-packages`. The FWB YAML argument scales from 1024 to the actual YOLOv8n channel width, and the block receives equal input/output widths.

The inspected parser source is `.venv/Lib/site-packages/ultralytics/nn/tasks.py`; module resolution uses `globals()[m]`, and its standard base-module branch supplies input channels and width-scales the declared output channels.

All absolute layer references following the insertion were shifted. The neck's final P5 route was explicitly redirected to the FWB output, and the Detect references now target layers 16, 19, and 22. Tests verified the custom YAML against the installed stock YAML and checked a representative registered parser construction/forward pass. The full custom detector model was **not** constructed because available RAM was below the architecture script's safety gate.

## Architecture and weight-transfer gates

- FWB tensor tests: **6/6 passed**, including multiple tensor dimensions, realistic 256-channel feature shape, finite outputs, backward propagation, parameter count, parser registration, and YAML reference checks.
- Pretrained layer-index remapping tests: **3/3 passed**. These unit tests verify index arithmetic only; no actual COCO checkpoint tensor was transferred.
- Existing Sprint 8–12 regression and TTS tests: **39/39 passed**.
- Full YOLOv8n+FWB 640×640 architecture check: **BLOCKED before model construction**. Available RAM was 0.58 GiB at the first gate attempt and 0.97 GiB at the latest attempt; the safety threshold is 3.00 GiB.
- Full-model parameter count and GFLOPs: **pending**.
- Pretrained COCO tensor-transfer check: **pending**. The prepared script maps stock model layers at indices 10 and above to target indices one greater to account for FWB insertion, then checks shape-compatible tensors, regression-head transfer, and that FWB weights retain their own initialization. The transfer report was not produced.
- Training did not begin because architecture and transfer gates are prerequisites.

## Intended training configuration

The prepared training script fixes the experiment at 50 epochs, 640-pixel images, batch 16, SGD, and learning rate 0.01. It also records the historical B1 settings that can be matched from `results/previous_b1/args.yaml`: seed 0, deterministic execution, `close_mosaic=10`, AMP enabled, 2 workers, patience 100, `lrf=0.01`, momentum 0.937, weight decay 0.0005, and 3 warmup epochs. Output run name: `results/fwb_optimization_full/`. The script refuses CPU execution and requires both mandatory gates before training.

The archived B1 args record `yolov8n.pt` with pretrained initialization. E1 uses the same official COCO weight file but requires an explicit remapped transfer because the FWB insertion shifts the neck and Detect layer indices. Exact training equivalence is not claimed: the historical B1 ran in a Kaggle environment, while local Ultralytics is 8.4.160; the old args also contain project-specific fields not established as available in this installed version. Batch size and core optimizer settings are preserved in the prepared workflow.

## Evaluation protocol and comparison

The prepared evaluator uses only the same held-out 750-image test split as Sprint 5, verifies its converted categories/images/annotations against the recorded B1 test ground truth, and uses CPU, 640 image size, batch 2, workers 0, confidence 0.001, IoU 0.7, max detections 300, no augmentation. It reuses Sprint 5 conversion, validation, and COCO evaluation helpers. Raw Ultralytics predictions are preserved; only the derived COCOeval input excludes invalid non-positive-area boxes, using the Sprint 5 rule.

FWB metrics, per-class results, pothole/traffic-cone differences, confusion plots, and prediction validation are **pending** because there is no trained FWB checkpoint. The comparison CSVs contain recorded B1 values only; FWB and delta cells are blank and explicitly marked pending. They do not contain fabricated results.

## Hardware and integrity

- Python 3.12.3; PyTorch 2.14.0+cpu; Ultralytics 8.4.160.
- CUDA: unavailable; no NVIDIA GPU detected.
- System RAM: 7.78 GiB total; 0.58–0.97 GiB available across architecture-gate attempts.
- Free disk space: approximately 222.08 GiB.
- Local pretrained YOLOv8n COCO weights: present.
- Dataset splits: present, and all listed image paths resolved.
- B1 checkpoint hash: unchanged across the architecture gate.
- Dataset split/config, manifest, taxonomy, and selected Sprint 5 metric/ground-truth artifact hashes were recorded. The Sprint 5 evaluator was not run and the B1 evaluation directory was not targeted by Sprint 13 scripts.

## Outcome and next action

The FWB module, parser adapter, model YAML, gate scripts, evaluator, and comparison scripts are prepared. **E1 training and test evaluation have not run. Sprint 13 is not complete.**

The local fresh-process parser-registration check passed with the pinned Ultralytics 8.4.160: the parser resolved FWB and injected the expected 256 input/output channels. This verifies parser registration only; it is not the full-model architecture gate.

Next, follow `results/sprint13_gpu_execution_guide.md` on Kaggle. Stage the unchanged dataset and split contents with Linux paths, preserve the frozen B1 and Sprint 5 references, and verify CUDA, memory, package versions, and checkpoint hashes. Then run the prepared `training/train_fwb.py --execute` command. Its required order is dataset validation → full 640×640 architecture and representative-batch gate → actual COCO tensor-transfer validation and report → 50-epoch training. If either gate fails, training must not start. Do not treat the pending comparison as an experimental outcome.

No unresolved parser-registration defect was found in the tested Ultralytics version. Environment-specific issues to resolve before GPU execution are the Windows absolute paths in the current local split lists/data config (the Kaggle working-copy procedure rewrites these to Linux paths) and confirmation that the mounted dataset contains the exact recorded images/labels. The actual checkpoint tensor-transfer ratio and compatible Detect regression-key coverage remain unverified until the transfer gate runs on Kaggle; if that check is low or fails, stop and diagnose it rather than weakening the guard or training anyway.

Sprint 14 / INT8 quantization has not started.

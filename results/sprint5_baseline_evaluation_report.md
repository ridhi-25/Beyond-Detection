# Sprint 5 — B1 Held-Out Test Evaluation

## Objective and protocol

This sprint evaluates the previously audited B1 detector on the held-out test split only. It is a final evaluation of the existing checkpoint, not a training run or a model-selection step. No training, test-label edits, split edits, or inference-time augmentation were performed.

## Environment and inputs

- Python: 3.12.3
- PyTorch: 2.14.0+cpu
- Ultralytics: 8.4.160
- Device: CPU; CUDA was not used.
- Checkpoint: `results/previous_b1/weights/best.pt` (the checkpoint audited in Sprint 4)
- Dataset config: `models/detector/data.yaml`
- Split: `data/splits/test.txt`, 750 unique images
- Image size: 640; batch size: 2; workers: 0
- Ground truth: 4,132 annotations across the 15-class taxonomy

The checkpoint and dataset files were consumed for evaluation. Ultralytics generated `data/merged/labels.cache` while scanning the split; that exact generated cache was removed after evaluation. The held-out split list, label files, class list, config, and checkpoint were not intentionally changed.

## Overall test metrics

The primary test metrics below are the native Ultralytics results. They are calculated on the held-out test split and are distinct from the historical training/validation metrics recorded in the archived B1 run.

| Metric | Test result |
|---|---:|
| Precision (B) | 0.9006970973457251 |
| Recall (B) | 0.8633117780342082 |
| mAP50 (B) | 0.9285558753282753 |
| mAP50–95 (B) | 0.7766256219114711 |

The independent `pycocotools` COCOeval calculation, using the same test ground truth and valid predictions, reported:

| COCOeval metric | Result |
|---|---:|
| AP50–95 (all sizes) | 0.7753268662629643 |
| AP50 | 0.9268907615899443 |
| AP75 | 0.8643484128291948 |

The small differences between COCOeval and Ultralytics are expected because the COCOeval derivative excludes geometrically degenerate predictions as described under prediction validation below.

## Object-size AP

COCO area categories use the standard pixel-area thresholds: small (`< 32²`), medium (`32²` to `< 96²`), and large (`>= 96²`).

| Ground-truth box size | AP50 | AP50–95 |
|---|---:|---:|
| Small | 0.6475752723897007 | 0.34104228890172944 |
| Medium | 0.7388321040492772 | 0.5268156458693508 |
| Large | 0.9522137150998997 | 0.8131757685935365 |

## Per-class results and test support

Class image counts and object counts describe the test ground truth. Per-class precision/recall/AP are Ultralytics test metrics.

| Class | Images | Objects | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|---:|
| person | 365 | 844 | 0.9078846938240489 | 0.7725118483412322 | 0.9106629739303108 | 0.7110826179554324 |
| rickshaw | 511 | 1,129 | 0.9245142560535655 | 0.9043401240035429 | 0.9750494534216658 | 0.8650660369298148 |
| rickshaw van | 78 | 83 | 0.8303976232802835 | 0.7469879518072289 | 0.834321673342773 | 0.6291487277703245 |
| auto rickshaw | 168 | 228 | 0.9323559998382608 | 0.9210526315789473 | 0.9648718462429892 | 0.8808169811823827 |
| truck | 23 | 29 | 0.9056875554276689 | 1.0 | 0.9901874999999999 | 0.8612425604731524 |
| pickup truck | 65 | 65 | 0.9612891166254496 | 0.9076923076923077 | 0.9556572905006471 | 0.8517703270376803 |
| private car | 355 | 543 | 0.920443910437991 | 0.9162055760569913 | 0.974005330194232 | 0.8691341178929578 |
| motorcycle | 343 | 509 | 0.945770084331064 | 0.8908461379324654 | 0.9707432926597068 | 0.7803621757299495 |
| bicycle | 114 | 121 | 0.909585495984026 | 0.8314371123677342 | 0.9201112125434528 | 0.7454936532484145 |
| bus | 66 | 86 | 0.9625585001346254 | 0.896818142916172 | 0.9760630378635807 | 0.8720198974416359 |
| micro bus | 105 | 105 | 0.9604313518463224 | 0.9246756826686042 | 0.9760159060028605 | 0.9070379212989523 |
| covered van | 24 | 24 | 0.8769868777326438 | 0.9166666666666666 | 0.966402815338742 | 0.8541100909388094 |
| human hauler | 28 | 39 | 0.9483114471510576 | 0.9409944189470189 | 0.9808907758907758 | 0.8729265059540795 |
| pothole | 68 | 189 | 0.7002958358784419 | 0.5316219825776888 | 0.6465218643921655 | 0.380057055386812 |
| traffic cone | 33 | 138 | 0.8239437116404267 | 0.8478260869565217 | 0.8868331576002275 | 0.5691156594316678 |

## Prediction validation

- Ultralytics emitted 40,885 raw prediction records across all 750 test images.
- 40,818 positive-area records were remapped to the derived COCO image IDs and zero-based category IDs for COCOeval.
- 67 predictions had zero or negative width/height (37 zero-width and 30 zero-height). These were excluded only from `predictions_cocoeval.json`, since they are invalid COCO bounding boxes. The original Ultralytics `predictions.json` was retained unchanged.
- No unmatched image names, invalid category IDs, non-finite values, or out-of-range scores were found.
- Ground-truth boxes were converted from the existing YOLO labels to pixel-space COCO boxes without clipping. Some historical test annotations extend beyond image boundaries; this inherited annotation condition was preserved, not corrected.

## Artifacts

Evaluation outputs are under `results/b1_baseline_eval/`:

- `overall_metrics.csv` and `overall_metrics.json`
- `per_class_metrics.csv` and `per_class_map50.csv`
- `test_class_distribution.csv`
- `test_ground_truth_coco.json`
- `predictions.json` (raw Ultralytics output)
- `predictions_cocoeval.json` (validated/remapped COCOeval derivative)
- `prediction_validation.json`
- `cocoeval_summary.txt` and `size_breakdown.csv`
- Ultralytics precision, recall, F1, PR, confusion-matrix, and test-batch plots

All 12 generated plot/batch-image files were opened and decoded successfully. The evaluation script is `evaluation/evaluate_baseline.py`; the reusable ground-truth converter is `evaluation/yolo_to_coco.py`.

## Sprint 5 status

**Complete.** The audited B1 checkpoint was evaluated on the held-out test split, all required overall/per-class/size metrics and outputs were saved, and the experiment log received a distinct Sprint 5 test-evaluation row. No model training was run. Sprint 6 has not been started.

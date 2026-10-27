# Sprint 4 — Dataset Merge and B1 Baseline Preparation

## 1. Sprint objective

Import and verify the supplied Baidu/Wang road-obstacle archive and historical B1 artifacts, produce a 15-class merged dataset with reproducible splits, and prepare/validate the YOLOv8n B1 training configuration without running full training.

## 2. Existing artifacts inspected

- `C:\Users\ridhi\Downloads\datas.zip`: readable, integrity-checked, retained unchanged; contains `Annotation/train.json`, `Annotation/eval.json`, and 6,276 JPGs under `is/`.
- `C:\Users\ridhi\Downloads\B1_baseline_50epochs.zip`: readable, integrity-checked, retained unchanged; contains `weights/best.pt`, `weights/last.pt`, `results.csv`, `args.yaml`, result plots, confusion matrices, and train/validation preview images.

## 3. Baidu dataset verification

- Extracted copy: `data/raw/baidu_road_obstacle/`.
- COCO files: `Annotation/train.json` (5,706 images, 72,252 annotations) and `Annotation/eval.json` (570 images, 6,717 annotations).
- Both JSONs contain `images`, `type`, `annotations`, `categories`. All referenced images were present. Train/eval filenames do not overlap.
- Categories found (train ID order): 1 car, 2 person, 3 trafficcone, 4 pothole. Eval order is 1 person, 2 car, 3 trafficcone, 4 pothole; conversion maps by exact category name.
- Source target annotations: 1,719 pothole and 1,176 trafficcone. 162 target boxes were invalid/out of bounds and excluded (137 train-source, 25 eval-source). No source annotations or images were modified.
- Detailed findings: `results/sprint4_baidu_verification.md`.

## 4. Previous B1 verification

- Extracted historical copy: `results/previous_b1/`; original ZIP was preserved.
- Found `best.pt`, `last.pt`, 50-row `results.csv`, `args.yaml`, result plot, precision/recall/F1/PR plots, confusion matrices, labels preview, training batches, and validation previews.
- Verified configuration: YOLOv8n, 15 classes with names matching this sprint's taxonomy, 50 epochs, image size 640, batch 16, SGD, lr0 0.01. Recorded device is `0`; the archive records a Kaggle path but does not independently establish the hardware used.
- Genuine epoch-50 metrics from archived CSV/checkpoint: precision 0.88449, recall 0.83755, mAP50 0.90954, mAP50–95 0.72882.
- The archive does not contain its referenced `data.yaml` or historical dataset files, so byte-level/content equivalence of its dataset to the new merge cannot be verified.
- Detailed findings: `results/sprint4_previous_b1_report.md`.

## 5. Unified class list

Created `data/merged/classes.txt` with exactly 15 names in the required order. RSUD20K IDs 0–12 are unchanged; pothole maps to 13 and trafficcone to 14. Baidu person/car rows were ignored.

## 6. Conversion results

- Converter: `preprocessing/convert_baidu_to_yolo.py`.
- Valid target annotations converted: 1,582 pothole and 1,151 trafficcone.
- Baidu images copied with `baidu_` filename prefixes: 889.
- Images skipped because they had no valid retained annotation: 5,387.
- Missing/unreadable source images: 0.
- Invalid target annotations skipped: 162; valid boxes were normalized to YOLO format.
- Conversion summary: `results/sprint4_baidu_conversion_report.md`.

## 7. Merged dataset statistics

- RSUD20K copied from the actual verified Sprint 2 location: 20,334 image/label pairs.
- Baidu converted pairs: 889.
- Total merged images: 21,223.
- Total merged label files: 21,223.
- Total annotation rows: 132,733.
- Manifest: `data/merged/manifest.csv` (`filename`, `source_dataset`, `original_filename`, `split`).
- RSUD20K originals remained unchanged; copies are prefixed `rsud20k_<source-split>_` to prevent collisions.
- Dataset details and class distribution: `results/sprint4_dataset_report.md`.

## 8. Split statistics

| Split | Total | RSUD20K | Baidu |
|---|---:|---:|---:|
| Train | 19,390 | 18,681 | 709 |
| Validation | 1,083 | 1,004 | 79 |
| Test | 750 | 649 | 101 |

RSUD20K splits are preserved. Baidu train-source images were deterministically shuffled with seed 42 and divided 90% train / 10% validation; Baidu eval-source images were assigned to test. Split lists are in `data/splits/train.txt`, `val.txt`, and `test.txt`.

## 9. Validation results

- Split-list entries: 19,390 train / 1,083 validation / 750 test.
- Missing images: 0; missing labels: 0; orphan labels: 0.
- Duplicate output filenames: 0.
- Malformed label rows/files: 0.
- Invalid class IDs: 0; non-normalized/non-finite coordinates: 0.
- Train/validation/test overlap: 0.
- One sample image from each split opened successfully.
- RSUD20K source boxes that extend past image bounds remain unchanged in merged labels; such edge geometry is documented in Sprint 2. No clipping was performed.

## 10. `models/detector/data.yaml`

Created with `nc: 15`, exact class names/order, and absolute project paths to the three split-list files. YAML parsing and Ultralytics `check_det_dataset` succeeded; all train/val/test list files and listed images/labels resolve.

## 11. Training script

Created `training/train_detector.py`, a configurable plain YOLOv8n training entry point with defaults of 50 epochs, 640 image size, batch 16, SGD, lr0 0.01, and device `0`. It logs the configuration and does not include depth, MiDaS, fusion, scoring, or tracking.

## 12. Previous B1 status

The previous historical run has the intended B1 model/taxonomy/training settings and its metrics are preserved in `results/experiment_log.csv` with stage `previous_verified_B1`. Exact data-content equivalence is unknown because the historical YAML/data were not in its archive. The historical result was not retrained or overwritten.

## 13. GPU execution status

No local training or smoke-training run was performed. This laptop is CPU-only with approximately 8 GB RAM; 50-epoch training remains inappropriate locally. YOLO import/weights, dataset YAML, sample image/label loading, and split/file/class checks are lightweight validation only. GPU execution was not started in this sprint.

## 14. Remaining action

Before a future GPU run, use the prepared `training/train_detector.py` and `models/detector/data.yaml` in the intended GPU environment. Decide how to handle the previously documented RSUD20K boundary-overrun boxes before any label-changing preprocessing. Historical B1 metrics are reference artifacts; do not equate them to this newly generated dataset without verifying the historical data.

## 15. Sprint 5

**Sprint 5 was not started.**

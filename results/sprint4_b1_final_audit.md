# B1 Baseline Final Audit

## 1. Artifact Verification

Audited the extracted historical archive at `results/previous_b1/`. Files were not modified.

| Artifact | Exists | Size | Readability / result |
|---|---:|---:|---|
| `weights/best.pt` | Yes | 6,236,650 bytes | Loaded successfully with `map_location="cpu"` |
| `weights/last.pt` | Yes | 6,236,650 bytes | Loaded successfully with `map_location="cpu"` |
| `results.csv` | Yes | 5,988 bytes | Parsed successfully; 50 data rows |
| `args.yaml` | Yes | 1,772 bytes | Parsed successfully as YAML |
| `results.png` | Yes | 274,305 bytes | Readable PNG, 2400×1200 |
| `confusion_matrix.png` | Yes | 347,157 bytes | Readable PNG, 3000×2250 |
| `confusion_matrix_normalized.png` | Yes | 371,862 bytes | Readable PNG, 3000×2250 |
| `BoxF1_curve.png` | Yes | 496,138 bytes | Readable PNG, 2250×1500 |
| `BoxPR_curve.png` | Yes | 340,491 bytes | Readable PNG, 2250×1500 |
| `BoxP_curve.png` | Yes | 376,167 bytes | Readable PNG, 2250×1500 |
| `BoxR_curve.png` | Yes | 438,506 bytes | Readable PNG, 2250×1500 |
| `labels.jpg` | Yes | 183,092 bytes | Readable JPEG, 1600×1600 |
| Training batch previews | Yes | 3 early and 3 later previews | All archived PNG/JPEG visual artifacts passed image-read checks |
| Validation label/prediction previews | Yes | 3 label/prediction pairs | All archived PNG/JPEG visual artifacts passed image-read checks |

All 20 visual plot/preview artifacts in the extracted archive were readable. The original archive remains separate and unchanged.

## 2. Training Duration / Epoch Verification

- `args.yaml` records `epochs: 50`.
- `results.csv` contains exactly 50 rows with the complete consecutive epoch sequence 1 through 50.
- First epoch: 1. Final epoch: 50.
- No blank, NaN, infinite, nonnumeric, or otherwise non-finite values were found in any CSV data cell, including training losses and validation metrics.
- The final CSV `time` cell is exactly `11102.7`; the CSV does not label a unit, so no unit is inferred.
- Both checkpoints load successfully, but each serialized checkpoint has `epoch: -1`. Therefore, the epoch count is established by the complete CSV and training args, not by the checkpoint epoch field.

## 3. Training Configuration

Directly verified in `args.yaml` and checkpoint `train_args`:

| Setting | Archived value | Intended B1 | Match |
|---|---|---|---|
| Task / mode | detect / train | Plain detector training | Yes |
| Model | `yolov8n.pt` | YOLOv8n | Yes |
| Epochs | 50 | 50 | Yes |
| Image size | 640 | 640 | Yes |
| Batch | 16 | 16 | Yes |
| Optimizer | SGD | SGD | Yes |
| `lr0` | 0.01 | 0.01 | Yes |
| Pretrained | true | pretrained YOLOv8n start | Yes |
| Device | `0` | Historical run setting | Recorded |
| Run name | `b1_baseline_full-2` | Historical identifier | Recorded |

The checkpoint reports Ultralytics version 8.4.155. The current project environment uses 8.4.160; this is a version difference, not evidence of a different model architecture.

## 4. Model / Checkpoint Verification

- Both `best.pt` and `last.pt` were loaded with PyTorch CPU mapping; neither was modified.
- Both contain a standard Ultralytics `DetectionModel`.
- The model graph is the standard YOLOv8n-style Conv/C2f/SPPF/Upsample/Concat/Detect graph; no custom fusion or added prediction modules were found.
- Checkpoint metadata for both records 15 classes and the same 15 class-name mapping.
- The checkpoint serialized `epoch` value is `-1` for both files. The checkpoint also stores training args and final training metrics; the CSV is used as the epoch-history source of record.
- No inference was run as part of this audit.

## 5. Class Taxonomy Verification

The checkpoint names, `data/merged/classes.txt`, and `models/detector/data.yaml` match exactly:

| ID | Name |
|---:|---|
| 0 | person |
| 1 | rickshaw |
| 2 | rickshaw van |
| 3 | auto rickshaw |
| 4 | truck |
| 5 | pickup truck |
| 6 | private car |
| 7 | motorcycle |
| 8 | bicycle |
| 9 | bus |
| 10 | micro bus |
| 11 | covered van |
| 12 | human hauler |
| 13 | pothole |
| 14 | traffic cone |

The current merged taxonomy has exactly 15 classes. `models/detector/data.yaml` has `nc: 15`, 15 names, and its names list matches the checkpoint mapping.

## 6. Final Metrics

Exact values from the epoch-50 row of `results.csv` (also present in `best.pt` checkpoint `train_metrics`):

| Metric | Final value |
|---|---:|
| `metrics/precision(B)` | 0.88449 |
| `metrics/recall(B)` | 0.83755 |
| `metrics/mAP50(B)` | 0.90954 |
| `metrics/mAP50-95(B)` | 0.72882 |

## 7. Best Metrics

The maximum values in the 50-row CSV are attained at epoch 50:

| Metric | Best epoch | Best value |
|---|---:|---:|
| `metrics/mAP50(B)` | 50 | 0.90954 |
| `metrics/mAP50-95(B)` | 50 | 0.72882 |

## 8. Dataset Relationship

**A. Directly verified from historical B1 artifacts**

- Training args reference `/kaggle/working/beyond-detection/models/detector/data.yaml`.
- Checkpoint and args identify a 15-class YOLOv8n detector; checkpoint class names match the intended unified taxonomy.
- Training arguments and the 50-epoch CSV support the recorded model, image size, batch, optimizer, learning rate, epochs, and final metrics.
- The historical archive does not contain the referenced YAML or raw training dataset, so historical file identity/split identity is not independently verifiable from this archive.

**B. Directly verified from the current reconstructed project**

- `data/merged/classes.txt` contains exactly 15 classes in the intended order.
- `models/detector/data.yaml` has `nc: 15`; its names match both the class file and checkpoint. Ultralytics successfully parsed and resolved its train/val/test paths in the current project.
- The reconstructed dataset was validated independently and its current split/count/label checks passed, as recorded in `results/sprint4_dataset_report.md`.

**C. Based on the user's documented project history**

- The user confirms the archived 50-epoch B1 was trained on the merged RSUD20K + Baidu dataset.
- This audit accepts that provenance statement as project history, but does not claim byte-for-byte identity between that historical dataset and the reconstruction in `data/merged/`.

## 9. Plain-Baseline Verification

- The recorded task is `detect` and mode is `train`; the checkpoint is an ordinary YOLOv8n-style detector graph.
- No depth/MiDaS module, depth fusion, distance estimator, threat-scoring component, attention module, or custom fusion component appears in the checkpoint graph or its training arguments.
- The args file contains a `tracker: tracktrack.yaml` metadata value. Tracking is not part of the YOLO detection training graph or this detector checkpoint; no tracking component was found in the model architecture. This inert/general configuration field is noted rather than hidden.
- The detector is plain YOLOv8n fine-tuning for the unified 15-class taxonomy.

## 10. Conclusion

### B1 VERIFIED — READY FOR SPRINT 5

The archived CSV verifies all 50 consecutive epochs with finite training and validation metrics. Both checkpoints load; model architecture, configuration, and 15-class names match the intended B1 baseline; final and best metrics are present and consistent between the CSV and checkpoint metadata. The current reconstructed taxonomy/config also matches the checkpoint. No evidence contradicts the B1 definition.

The historical raw dataset is not inside the B1 archive, so byte-for-byte identity with the current reconstruction remains unproven. This limitation does not invalidate the user-confirmed historical dataset provenance and is not a reason to retrain. No training, inference, or Sprint 5 work was performed during this audit.

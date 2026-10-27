# Sprint 4 — Previous B1 Artifact Verification

## Source and handling

- Source archive: `C:\Users\ridhi\Downloads\B1_baseline_50epochs.zip` (18,832,503 bytes).
- Archive integrity verified; original ZIP retained unchanged.
- Extracted copy: `results/previous_b1/`.
- This is historical research work. It was inspected and copied, not rerun or treated as a Sprint 4 training result.

## Verified artifacts

- `weights/best.pt` (6,236,650 bytes)
- `weights/last.pt` (6,236,650 bytes)
- `results.csv` (50 epoch rows)
- `args.yaml`
- `results.png`
- `confusion_matrix.png`, `confusion_matrix_normalized.png`
- `BoxF1_curve.png`, `BoxPR_curve.png`, `BoxP_curve.png`, `BoxR_curve.png`
- `labels.jpg`
- Training batch previews and validation label/prediction previews

The archive does not include the referenced `data.yaml` or source merged dataset.

## Verified previous-run configuration

| Setting | Recorded value |
|---|---|
| Model | `yolov8n.pt` |
| Epochs | 50 (CSV rows 1–50; args.yaml configured 50) |
| Image size | 640 |
| Batch | 16 |
| Optimizer | SGD |
| Initial learning rate (`lr0`) | 0.01 |
| Device | `0` (as recorded in historical args.yaml) |
| Run name | `b1_baseline_full-2` |
| Dataset config reference | `/kaggle/working/beyond-detection/models/detector/data.yaml` (file not included in archive) |
| Class count | 15 (`nc: 15` in checkpoint model YAML) |
| Checkpoint class names | IDs 0–14 match the Sprint 4 unified taxonomy |
| Pretrained flag | true |
| `cls_remap` | true |

The historical model/configuration matches the intended plain B1 configuration (YOLOv8n, 15 taxonomy names, 640, SGD, lr0 0.01, 50 epochs). The exact historical merged dataset contents/splits cannot be compared because the referenced dataset config and data are not in the archive. The archived args indicate a Kaggle GPU device, but the checkpoint does not independently establish hardware identity.

## Metrics from archived `results.csv`

The values below are the genuine epoch-50 row in the archived CSV (not a new Sprint 4 run):

| Metric | Epoch 50 |
|---|---:|
| Precision (B) | 0.88449 |
| Recall (B) | 0.83755 |
| mAP50 (B) | 0.90954 |
| mAP50–95 (B) | 0.72882 |

The mAP50 and mAP50–95 maxima in the 50-row CSV are also at epoch 50 with these same values. Epoch-50 metrics are also recorded in checkpoint `train_metrics`.

## Information not established by the archive

- Exact contents/version of the historical train/validation/test dataset.
- Whether the historical image/label files are byte-for-byte the same as the Sprint 4 merged dataset created here.
- Any metric not recorded in `results.csv` or checkpoint metadata.
- No values were inferred or fabricated.

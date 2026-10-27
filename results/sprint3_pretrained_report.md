# Sprint 3 — Basic Pretrained YOLOv8n Inference

## Objective

Verify image loading and the pretrained YOLO inference pipeline on real RSUD20K road-scene images before any custom training. This was a tooling check, not an accuracy evaluation.

## Environment

- Project: `C:\beyond-detection`
- Python: 3.12.3
- PyTorch: 2.14.0+cpu
- Ultralytics: 8.4.160
- CUDA available: No
- Inference device: CPU

## Model

- Model: YOLOv8n pretrained on COCO (`YOLO("yolov8n.pt")`)
- Weight source: Official Ultralytics release asset at `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt`; downloaded automatically by Ultralytics because it was not cached.
- No fine-tuning, custom weights, merged dataset, or previous baseline weights were used.

## Input

All three images were verified as readable JPEGs and were left unchanged:

1. Primary: `C:\beyond-detection\data\raw\rsud20k_download\verified_download\extracted\rsud20k\images\train\train0.jpg` — 1920 × 1080.
2. Additional: `C:\beyond-detection\data\raw\rsud20k_download\verified_download\extracted\rsud20k\images\val\val0.jpg` — 1920 × 1080.
3. Additional: `C:\beyond-detection\data\raw\rsud20k_download\verified_download\extracted\rsud20k\images\val\val1.jpg` — 1920 × 1080.

## Results

### Primary image — `train0.jpg`

- Detections: 8
- Annotated output: `results/sprint3/image_1_annotated.jpg`

| COCO class | Confidence | Bounding box (x1, y1, x2, y2) |
|---|---:|---|
| person | 0.739585 | (184.14, 550.35, 311.49, 738.71) |
| car | 0.713195 | (819.39, 692.73, 929.74, 769.08) |
| person | 0.713033 | (561.71, 628.82, 619.38, 794.73) |
| car | 0.601593 | (744.61, 666.27, 808.74, 708.66) |
| person | 0.481484 | (525.62, 652.52, 561.33, 755.43) |
| bicycle | 0.399810 | (556.64, 705.03, 614.71, 803.78) |
| car | 0.360654 | (919.72, 651.29, 967.31, 716.51) |
| person | 0.329144 | (561.10, 628.14, 666.37, 774.18) |

### Validation image — `val0.jpg`

- Detections: 8
- Annotated output: `results/sprint3/image_2_annotated.jpg`

| COCO class | Confidence | Bounding box (x1, y1, x2, y2) |
|---|---:|---|
| car | 0.885967 | (1310.35, 555.41, 1885.75, 943.63) |
| car | 0.712720 | (1176.12, 575.12, 1432.00, 794.87) |
| truck | 0.649140 | (883.04, 552.13, 1112.83, 769.91) |
| car | 0.634603 | (1106.32, 601.17, 1215.55, 724.36) |
| car | 0.435418 | (337.47, 382.33, 826.55, 1067.03) |
| car | 0.366710 | (867.74, 617.00, 905.94, 679.75) |
| car | 0.309706 | (868.88, 615.99, 908.66, 703.79) |
| car | 0.305627 | (786.72, 606.97, 881.19, 672.82) |

### Validation image — `val1.jpg`

- Detections: 7
- Annotated output: `results/sprint3/image_3_annotated.jpg`

| COCO class | Confidence | Bounding box (x1, y1, x2, y2) |
|---|---:|---|
| car | 0.939711 | (1270.88, 517.49, 1919.29, 1064.37) |
| bus | 0.932666 | (874.93, 259.62, 1546.00, 831.81) |
| person | 0.914670 | (309.26, 429.57, 676.50, 1066.10) |
| car | 0.914465 | (1.85, 538.48, 290.13, 1057.86) |
| motorcycle | 0.602300 | (289.65, 826.14, 575.04, 1071.39) |
| car | 0.395677 | (711.61, 574.18, 815.57, 698.38) |
| truck | 0.295087 | (710.11, 572.66, 814.91, 700.43) |

All three inference calls completed, produced detections, and wrote non-empty annotated images. COCO box coordinates are pixel-space `(x1, y1, x2, y2)`.

## Interpretation

This confirms the pretrained inference path can load RSUD20K images, run YOLOv8n on CPU, return COCO class labels/confidences/boxes, and save annotated images. The model is COCO-pretrained and does not know the 13 RSUD20K class definitions. Missing or incorrect recognition of rickshaw variants, micro buses, covered vans, human haulers, or other Bangladesh-specific objects is expected. No ground-truth accuracy evaluation was performed.

## Previous Baseline Status

A previous 50-epoch baseline exists outside this fresh VS Code workspace and was not available to or used by this Sprint 3 run. No baseline artifacts were imported, copied, or modified.

## Sprint 3 Status

**Completed successfully.** Pretrained YOLOv8n CPU inference completed on one train and two validation images, and all annotated output files were verified to exist and be non-empty. No training or dataset modifications were performed.

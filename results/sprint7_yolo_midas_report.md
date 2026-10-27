# Sprint 7 — YOLO + MiDaS Combined Inference

## Objective

Run the audited B1 YOLOv8n detector and pretrained MiDaS-small on the same RSUD20K frame, then associate each detection with the simple mean relative inverse-depth value inside its bounding box. This sprint implements no risk or fusion formula.

## Models

- **YOLO:** audited B1 checkpoint at `results/previous_b1/weights/best.pt`; checkpoint loaded successfully and its 15 class names matched `data/merged/classes.txt`.
- **MiDaS:** pretrained MiDaS-small loaded through PyTorch Hub and evaluated with the model's `small_transform`.
- The optional `results/b1_baseline_full/weights/best.pt` path was not present; the verified `previous_b1` checkpoint was used.

## Input

`C:\beyond-detection\data\raw\rsud20k_download\verified_download\extracted\rsud20k\images\train\train0.jpg`

Input dimensions: **1920 × 1080**.

This is the same sample image used in Sprint 3 and Sprint 6. It was read only.

## Device

**CPU** (`torch.cuda.is_available()` returned `False`). The project virtual environment used Python 3.12.3, PyTorch 2.14.0+cpu, Ultralytics 8.4.160, and timm 1.0.30.

## Detection-depth association

For each YOLO bounding box, coordinates were clipped to the depth-map boundary and the arithmetic mean of MiDaS relative inverse-depth values in the box region was sampled. Values were then normalized with the full depth map's minimum and maximum for readability:

`(raw_depth - depth_min) / (depth_max - depth_min + 1e-6)`

## Results

Depth map shape: **1080 × 1920**. All depth values were finite; map range was **-11.01 to 366.29**. YOLO produced 7 detections and all 7 received a finite depth sample.

| Class | Confidence | Raw average depth | Normalized depth (0–1; 1 = relatively closer) |
|---|---:|---:|---:|
| rickshaw | 0.94 | 268.0 | 0.74 |
| bicycle | 0.91 | 79.7 | 0.24 |
| auto rickshaw | 0.89 | 53.0 | 0.17 |
| private car | 0.83 | 45.4 | 0.15 |
| rickshaw | 0.79 | 61.9 | 0.19 |
| rickshaw | 0.76 | 53.2 | 0.17 |
| motorcycle | 0.64 | 73.3 | 0.22 |

## Output

Annotated image: `results/sprint7_yolo_midas_combined.jpg`

The image was opened and visually inspected. All 7 green YOLO boxes have a corresponding class/depth label; their locations align with the detected objects in the original frame.

## Interpretation

Normalized values represent relative inverse depth only. A higher normalized value indicates greater relative inverse depth / a closer appearance within this image; it is **not metric distance** and is not measured in metres. Values should not be interpreted as calibrated physical distances.

## Verification

- YOLO B1 checkpoint loaded, and its class taxonomy matched the expected 15-class project taxonomy.
- MiDaS-small loaded and used `small_transform`.
- Both models processed the same input image.
- The depth map matched the image dimensions and contained only finite values.
- All 7 detections were printed, sampled, and drawn; normalized values were finite and in the 0–1 range.
- `sample_box_depth()` passed targeted checks for mean sampling, boundary clipping, and invalid boxes.
- The annotated output exists, is non-empty, and was inspected successfully.

## Limitations

- MiDaS produces relative inverse depth, not metric distance.
- A simple arithmetic mean is used for each box; no robust depth statistic is applied.
- No temporal information, tracking, risk score, threat score, or risk fusion is included.

## Scope and status

No training or fine-tuning was performed. No dataset images, labels, split files, or B1 checkpoint were modified. No KITTI data was used. **Sprint 7 is complete; Sprint 8 has not been started.**

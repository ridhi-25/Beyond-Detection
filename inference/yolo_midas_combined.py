"""Run the verified B1 detector and MiDaS-small on the same image."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.risk_fusion import compute_risk
from inference.decision import decide_alert
from utils.zone_mapping import assign_zone, in_bottom_third_roi


CHECKPOINT = PROJECT_ROOT / "results" / "previous_b1" / "weights" / "best.pt"
OUTPUT_PATH = PROJECT_ROOT / "results" / "sprint7_yolo_midas_combined.jpg"
EXPECTED_CLASSES = [
    "person",
    "rickshaw",
    "rickshaw van",
    "auto rickshaw",
    "truck",
    "pickup truck",
    "private car",
    "motorcycle",
    "bicycle",
    "bus",
    "micro bus",
    "covered van",
    "human hauler",
    "pothole",
    "traffic cone",
]


def load_midas_small(device: str):
    torch.hub.list(
        "rwightman/gen-efficientnet-pytorch",
        trust_repo=True,
        verbose=False,
    )
    model = torch.hub.load(
        "intel-isl/MiDaS",
        "MiDaS_small",
        trust_repo=True,
    )
    model.to(device).eval()

    transform = torch.hub.load(
        "intel-isl/MiDaS",
        "transforms",
        trust_repo=True,
    ).small_transform

    return model, transform


def get_depth_map(img_rgb, midas_model, midas_transform, device: str) -> np.ndarray:
    input_batch = midas_transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas_model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    if depth_map.shape != img_rgb.shape[:2]:
        raise ValueError(
            f"Depth map shape {depth_map.shape} does not match image {img_rgb.shape[:2]}"
        )
    if not np.isfinite(depth_map).all():
        raise ValueError("MiDaS depth map contains non-finite values")
    return depth_map


def sample_box_depth(depth_map: np.ndarray, xyxy) -> float | None:
    x1, y1, x2, y2 = [int(value) for value in xyxy]

    x1, y1 = max(x1, 0), max(y1, 0)
    x2 = min(x2, depth_map.shape[1] - 1)
    y2 = min(y2, depth_map.shape[0] - 1)

    if x2 <= x1 or y2 <= y1:
        return None

    region = depth_map[y1:y2, x1:x2]
    if region.size == 0:
        return None
    return float(np.mean(region))


def score_and_tag_detection(
    class_name: str,
    confidence: float,
    xyxy: list[float],
    raw_depth: float,
    depth_min: float,
    depth_max: float,
    frame_width: int,
    frame_height: int,
) -> dict:
    """Apply the existing Sprint 7-9 depth, risk, and spatial calculations."""
    normalized_depth = (raw_depth - depth_min) / (depth_max - depth_min + 1e-6)
    if not np.isfinite(normalized_depth) or not -1e-6 <= normalized_depth <= 1 + 1e-6:
        raise ValueError(f"Invalid normalized depth for detection {class_name}")

    norm_depth = max(0.0, min(1.0, normalized_depth))
    risk_result = compute_risk(class_name, confidence, norm_depth)
    zone = assign_zone(xyxy, frame_width)
    in_roi = in_bottom_third_roi(xyxy, frame_height)
    return {
        "class_name": class_name,
        "confidence": confidence,
        "raw_depth": raw_depth,
        "normalized_depth": normalized_depth,
        "norm_depth": norm_depth,
        "tier": risk_result["tier"],
        "risk": risk_result["risk"],
        "zone": zone,
        "in_roi": in_roi,
        "box_xyxy": xyxy,
    }


def main(image_path: str | Path) -> int:
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Verified B1 checkpoint does not exist: {CHECKPOINT}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading verified B1 checkpoint: {CHECKPOINT}")
    yolo_model = YOLO(str(CHECKPOINT))
    class_names = [yolo_model.names[index] for index in range(len(yolo_model.names))]
    if class_names != EXPECTED_CLASSES:
        raise ValueError(f"B1 class taxonomy does not match expected classes: {class_names}")

    midas_model, midas_transform = load_midas_small(device)

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    frame_height, frame_width = img_bgr.shape[:2]

    detections = yolo_model.predict(
        source=str(image_path),
        device=device,
        verbose=False,
    )[0]
    depth_map = get_depth_map(
        img_rgb,
        midas_model,
        midas_transform,
        device,
    )

    dmin = float(depth_map.min())
    dmax = float(depth_map.max())
    boxes = detections.boxes
    detection_count = 0 if boxes is None else len(boxes)
    print(f"\nCombined output for {image_path}:")
    print(f"Input dimensions: {img_rgb.shape[1]} x {img_rgb.shape[0]}")
    print(f"Depth map shape: {depth_map.shape}")
    print(f"Detection count: {detection_count}")
    print(f"Depth value range: min={dmin:.2f}, max={dmax:.2f}")
    print("Depth is relative inverse depth; normalized values are not metres.")

    annotated = img_bgr.copy()
    sampled_count = 0
    all_detections = []
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            class_name = yolo_model.names[cls_id]

            raw_depth = sample_box_depth(depth_map, xyxy)
            if raw_depth is not None:
                if not np.isfinite(raw_depth):
                    raise ValueError(f"Non-finite sampled depth for detection {class_name}")
                detection = score_and_tag_detection(
                    class_name,
                    confidence,
                    xyxy,
                    raw_depth,
                    dmin,
                    dmax,
                    frame_width,
                    frame_height,
                )
                sampled_count += 1
                normalized_depth = detection["normalized_depth"]
                zone = detection["zone"]
                in_roi = detection["in_roi"]
                risk_result = {
                    "tier": detection["tier"],
                    "risk": detection["risk"],
                }
                print(f"    -> zone={zone:6s} in_roi={in_roi}")
                print(
                    f"  class={class_name:15s} conf={confidence:.2f} "
                    f"avg_depth_raw={raw_depth:.1f} "
                    f"avg_depth_norm(0-1, 1=closest)={normalized_depth:.2f}"
                )
                all_detections.append(detection)
                print(
                    f"    -> tier={risk_result['tier']:8s} "
                    f"risk={risk_result['risk']:.3f}"
                )
                label = f"{class_name} d={normalized_depth:.2f}"
            else:
                zone = assign_zone(xyxy, frame_width)
                in_roi = in_bottom_third_roi(xyxy, frame_height)
                print(f"    -> zone={zone:6s} in_roi={in_roi}")
                print(
                    f"  class={class_name:15s} conf={confidence:.2f} "
                    "avg_depth_raw=None avg_depth_norm=None"
                )
                label = f"{class_name} d=None"

            x1, y1, x2, y2 = [int(value) for value in xyxy]
            x1 = min(max(x1, 0), img_bgr.shape[1] - 1)
            y1 = min(max(y1, 0), img_bgr.shape[0] - 1)
            x2 = min(max(x2, 0), img_bgr.shape[1] - 1)
            y2 = min(max(y2, 0), img_bgr.shape[0] - 1)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    if sampled_count != detection_count:
        raise ValueError(
            f"Expected a depth sample for every detection: "
            f"{sampled_count}/{detection_count} valid samples"
        )

    alert = decide_alert(all_detections)
    if alert:
        print(f"\n>>> ALERT: {alert['message']}")
    else:
        print("\n>>> (silent — nothing above threshold in path)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT_PATH), annotated):
        raise OSError(f"Could not save annotated image: {OUTPUT_PATH}")
    if not OUTPUT_PATH.is_file() or OUTPUT_PATH.stat().st_size == 0:
        raise OSError(f"Annotated image is missing or empty: {OUTPUT_PATH}")
    print(f"Detections with valid mean-depth samples: {sampled_count}/{detection_count}")
    print(f"Annotated image (boxes + normalized depth) saved to: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python inference/yolo_midas_combined.py <path_to_image>",
            file=sys.stderr,
        )
        raise SystemExit(1)

    raise SystemExit(main(sys.argv[1]))

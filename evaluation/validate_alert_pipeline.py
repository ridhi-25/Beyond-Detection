"""Run a deterministic descriptive check of the existing alert pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.decision import ALERT_THRESHOLD, decide_alert, get_ahead_objects
from inference.yolo_midas_combined import (
    CHECKPOINT,
    EXPECTED_CLASSES,
    get_depth_map,
    load_midas_small,
    sample_box_depth,
    score_and_tag_detection,
)

DATASET_TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "rsud20k_download"
    / "verified_download"
    / "extracted"
    / "rsud20k"
    / "images"
    / "test"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "alert_validation"
IMAGE_COUNT = 30
LOW_CONFIDENCE_REVIEW_LIMIT = 0.5
EXTREME_DEPTH_LIMITS = (0.05, 0.95)
MANY_DETECTIONS_LIMIT = 10

IMAGE_FIELDS = [
    "image_name",
    "image_path",
    "image_width",
    "image_height",
    "num_detections",
    "num_ahead_objects",
    "max_ahead_risk",
    "alert",
    "alert_class",
    "alert_risk",
    "urgency",
    "direction",
    "stop_required",
    "max_left_risk",
    "max_right_risk",
    "highest_risk_detection_class",
    "highest_risk_detection_risk",
    "alert_is_highest_risk_detection",
    "review_flags",
]
DETECTION_FIELDS = [
    "image_name",
    "image_path",
    "class_name",
    "confidence",
    "raw_depth",
    "normalized_depth",
    "tier",
    "risk",
    "zone",
    "in_roi",
    "is_ahead",
    "box_x1",
    "box_y1",
    "box_x2",
    "box_y2",
]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_test_sort_key(path: Path) -> tuple[int, str]:
    match = re.fullmatch(r"test(\d+)", path.stem, flags=re.IGNORECASE)
    return (int(match.group(1)), path.name.lower()) if match else (10**12, path.name.lower())


def select_images() -> list[Path]:
    available = sorted(
        (
            path
            for path in DATASET_TEST_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ),
        key=numeric_test_sort_key,
    )
    if len(available) < IMAGE_COUNT:
        raise RuntimeError(
            f"Expected at least {IMAGE_COUNT} test images, found {len(available)} "
            f"in {DATASET_TEST_DIR}"
        )
    indices = [
        round(index * (len(available) - 1) / (IMAGE_COUNT - 1))
        for index in range(IMAGE_COUNT)
    ]
    selected = [available[index].resolve() for index in indices]
    if len(set(selected)) != IMAGE_COUNT:
        raise RuntimeError("Deterministic test-image selection produced duplicate paths")
    return selected


def overlay_frame(
    img_bgr: np.ndarray,
    detections: list[dict],
    alert: dict | None,
) -> np.ndarray:
    image = img_bgr.copy()
    height, width = image.shape[:2]
    panel_width = 720
    annotated = np.full(
        (height, width + panel_width, 3),
        (32, 32, 32),
        dtype=np.uint8,
    )
    annotated[:, :width] = image

    for x in (width // 3, (2 * width) // 3):
        cv2.line(annotated, (x, 0), (x, height - 1), (255, 180, 0), 1)
    cv2.line(
        annotated,
        (0, (2 * height) // 3),
        (width - 1, (2 * height) // 3),
        (255, 180, 0),
        1,
    )

    alert_index = None
    if alert is not None:
        ahead = get_ahead_objects(detections)
        selected = max(ahead, key=lambda detection: detection["risk"])
        alert_index = next(
            index for index, detection in enumerate(detections) if detection is selected
        )

    for index, detection in enumerate(detections):
        x1, y1, x2, y2 = [int(value) for value in detection["box_xyxy"]]
        x1 = min(max(x1, 0), width - 1)
        y1 = min(max(y1, 0), height - 1)
        x2 = min(max(x2, 0), width - 1)
        y2 = min(max(y2, 0), height - 1)
        color = (0, 0, 255) if index == alert_index else (0, 220, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, (x1, y1), 5, color, -1)

    if alert:
        banner = f"ALERT: {alert['message'].upper()}"
    else:
        banner = "NO ALERT"
    cv2.rectangle(annotated, (0, 0), (width - 1, 44), (25, 25, 25), -1)
    cv2.putText(
        annotated,
        banner,
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 255) if alert else (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    panel_x = width + 18
    cv2.putText(
        annotated,
        "Detection details (original-frame boxes)",
        (panel_x, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        "c=confidence  d=relative normalized depth  risk=existing score",
        (panel_x, 63),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    line_y = 96
    row_height = 42
    for index, detection in enumerate(detections):
        if line_y + row_height > height:
            cv2.putText(
                annotated,
                f"... {len(detections) - index} further detections in CSV",
                (panel_x, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            break
        color = (0, 0, 255) if index == alert_index else (0, 220, 0)
        ahead = detection["zone"] == "center" and detection["in_roi"]
        detail = (
            f"{index + 1}. {detection['class_name']} "
            f"c={detection['confidence']:.2f} "
            f"d={detection['normalized_depth']:.3f} "
            f"risk={detection['risk']:.4f}"
        )
        spatial = (
            f"zone={detection['zone']} ROI={detection['in_roi']} ahead={ahead}"
        )
        cv2.putText(
            annotated,
            detail,
            (panel_x, line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            spatial,
            (panel_x + 18, line_y + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        line_y += row_height
    return annotated


def descriptive_stats(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "min": None,
            "q25": None,
            "median": None,
            "mean": None,
            "q75": None,
            "max": None,
            "std_population": None,
        }
    data = np.asarray(values, dtype=np.float64)
    return {
        "count": int(data.size),
        "min": float(data.min()),
        "q25": float(np.quantile(data, 0.25)),
        "median": float(np.median(data)),
        "mean": float(data.mean()),
        "q75": float(np.quantile(data, 0.75)),
        "max": float(data.max()),
        "std_population": float(data.std()),
    }


def save_distributions(
    image_rows: list[dict],
    detection_rows: list[dict],
    output_dir: Path,
) -> dict:
    metrics_dir = output_dir / "metrics"
    alert_rows = [row for row in image_rows if row["alert"]]
    detection_risks = [float(row["risk"]) for row in detection_rows]
    depths = [float(row["normalized_depth"]) for row in detection_rows]
    max_ahead = [
        float(row["max_ahead_risk"])
        for row in image_rows
        if row["max_ahead_risk"] not in ("", None)
    ]

    risk_rows = []
    for measure, values in (
        ("all_detection_risk", detection_risks),
        ("normalized_depth", depths),
        ("per_image_max_ahead_risk", max_ahead),
    ):
        risk_rows.append({"measure": measure, **descriptive_stats(values)})
    write_csv(
        metrics_dir / "risk_distribution.csv",
        [
            "measure",
            "count",
            "min",
            "q25",
            "median",
            "mean",
            "q75",
            "max",
            "std_population",
        ],
        risk_rows,
    )

    class_counter = Counter(row["class_name"] for row in detection_rows)
    alert_class_counter = Counter(row["alert_class"] for row in alert_rows)
    write_csv(
        metrics_dir / "class_distribution.csv",
        ["class_name", "detection_count", "alert_count"],
        [
            {
                "class_name": class_name,
                "detection_count": class_counter[class_name],
                "alert_count": alert_class_counter[class_name],
            }
            for class_name in sorted(set(class_counter) | set(alert_class_counter))
        ],
    )

    alert_distribution = []
    for dimension, key in (
        ("urgency", "urgency"),
        ("direction", "direction"),
        ("alert_class", "alert_class"),
    ):
        counts = Counter(row[key] for row in alert_rows if row[key])
        alert_distribution.extend(
            {"dimension": dimension, "value": value, "count": count}
            for value, count in sorted(counts.items())
        )
    write_csv(
        metrics_dir / "alert_distribution.csv",
        ["dimension", "value", "count"],
        alert_distribution,
    )

    zone_counter = Counter(row["zone"] for row in detection_rows)
    roi_inside = sum(row["in_roi"] is True for row in detection_rows)
    roi_outside = sum(row["in_roi"] is False for row in detection_rows)
    zone_rows = [
        {"dimension": "zone", "value": zone, "count": zone_counter[zone]}
        for zone in ("left", "center", "right")
    ]
    zone_rows.extend(
        [
            {"dimension": "roi", "value": "inside", "count": roi_inside},
            {"dimension": "roi", "value": "outside", "count": roi_outside},
        ]
    )
    write_csv(
        metrics_dir / "zone_distribution.csv",
        ["dimension", "value", "count"],
        zone_rows,
    )

    ahead_rows = [
        row
        for row in detection_rows
        if row["zone"] == "center" and row["in_roi"] is True
    ]
    below = sum(float(row["risk"]) < ALERT_THRESHOLD for row in ahead_rows)
    at_or_above = sum(float(row["risk"]) >= ALERT_THRESHOLD for row in ahead_rows)
    near = [
        row
        for row in ahead_rows
        if 0.30 <= float(row["risk"]) <= 0.40
    ]
    threshold_rows = [
        {"category": "ahead_risk_below_0.35", "count": below},
        {"category": "ahead_risk_at_or_above_0.35", "count": at_or_above},
        {"category": "ahead_risk_near_threshold_0.30_to_0.40_inclusive", "count": len(near)},
    ]
    threshold_rows.extend(
        {
            "category": "near_threshold_detection",
            "count": 1,
            "image_name": row["image_name"],
            "class_name": row["class_name"],
            "risk": row["risk"],
        }
        for row in near
    )
    write_csv(
        metrics_dir / "threshold_analysis.csv",
        ["category", "count", "image_name", "class_name", "risk"],
        threshold_rows,
    )

    urgency_counter = Counter(row["urgency"] for row in alert_rows if row["urgency"])
    direction_counter = Counter(row["direction"] for row in alert_rows if row["direction"])
    ahead_per_image = [int(row["num_ahead_objects"]) for row in image_rows]
    summary = {
        "images": {
            "total": len(image_rows),
            "with_detections": sum(int(row["num_detections"]) > 0 for row in image_rows),
            "with_ahead_objects": sum(int(row["num_ahead_objects"]) > 0 for row in image_rows),
            "with_alerts": len(alert_rows),
            "without_alerts": len(image_rows) - len(alert_rows),
            "alert_rate": len(alert_rows) / len(image_rows) if image_rows else 0.0,
        },
        "detections": {
            "total": len(detection_rows),
            "zero_detection_images": sum(int(row["num_detections"]) == 0 for row in image_rows),
            "many_detection_images_at_least_10": sum(
                int(row["num_detections"]) >= MANY_DETECTIONS_LIMIT for row in image_rows
            ),
            "by_zone": {
                zone: zone_counter[zone] for zone in ("left", "center", "right")
            },
            "inside_roi": roi_inside,
            "outside_roi": roi_outside,
            "ahead": len(ahead_rows),
            "by_class": dict(sorted(class_counter.items())),
        },
        "alerts": {
            "by_urgency": {
                level: urgency_counter[level] for level in ("caution", "warning", "danger")
            },
            "stop": direction_counter["stop"],
            "step_left": direction_counter["step left"],
            "step_right": direction_counter["step right"],
            "by_class": dict(sorted(alert_class_counter.items())),
        },
        "ahead_risk_threshold_analysis": {
            "threshold": ALERT_THRESHOLD,
            "below_threshold": below,
            "at_or_above_threshold": at_or_above,
            "near_threshold_definition": "0.30 <= risk <= 0.40",
            "near_threshold_count": len(near),
        },
        "distributions": {
            "all_detection_risk": descriptive_stats(detection_risks),
            "normalized_depth": descriptive_stats(depths),
            "per_image_max_ahead_risk": descriptive_stats(max_ahead),
            "ahead_objects_per_image": descriptive_stats([float(value) for value in ahead_per_image]),
        },
    }
    return summary


def main() -> int:
    for subdirectory in ("annotated", "metrics", "failure_cases"):
        (OUTPUT_DIR / subdirectory).mkdir(parents=True, exist_ok=True)
    prior_review_path = OUTPUT_DIR / "failure_cases" / "manual_visual_review.csv"
    prior_reviews = {}
    if prior_review_path.is_file():
        with prior_review_path.open(encoding="utf-8", newline="") as stream:
            prior_reviews = {
                row["image_name"]: row for row in csv.DictReader(stream)
            }
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Verified B1 checkpoint does not exist: {CHECKPOINT}")

    selected = select_images()
    image_hashes = {path: sha256_file(path) for path in selected}
    checkpoint_hash_before = sha256_file(CHECKPOINT)

    device = "cpu"
    model = YOLO(str(CHECKPOINT))
    class_names = [model.names[index] for index in range(len(model.names))]
    if class_names != EXPECTED_CLASSES:
        raise ValueError(f"B1 class taxonomy does not match expected classes: {class_names}")
    midas_model, midas_transform = load_midas_small(device)

    image_rows: list[dict] = []
    detection_rows: list[dict] = []
    failure_rows: list[dict] = []
    direction_rows: list[dict] = []
    implementation_issues: list[dict] = []

    for index, image_path in enumerate(selected, start=1):
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise OSError(f"Could not read validation image: {image_path}")
        frame_height, frame_width = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = model.predict(source=str(image_path), device=device, verbose=False)[0]
        depth_map = get_depth_map(img_rgb, midas_model, midas_transform, device)
        depth_min = float(depth_map.min())
        depth_max = float(depth_map.max())
        boxes = result.boxes
        count = 0 if boxes is None else len(boxes)
        detections: list[dict] = []

        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                xyxy = [float(value) for value in box.xyxy[0].tolist()]
                class_name = model.names[class_id]
                raw_depth = sample_box_depth(depth_map, xyxy)
                if raw_depth is None:
                    implementation_issues.append(
                        {
                            "image_name": image_path.name,
                            "class_name": class_name,
                            "issue": "detected box had no valid sampled depth region",
                            "box_xyxy": xyxy,
                        }
                    )
                    continue
                if not np.isfinite(raw_depth):
                    implementation_issues.append(
                        {
                            "image_name": image_path.name,
                            "class_name": class_name,
                            "issue": "non-finite sampled depth",
                            "box_xyxy": xyxy,
                        }
                    )
                    continue
                detection = score_and_tag_detection(
                    class_name,
                    confidence,
                    xyxy,
                    raw_depth,
                    depth_min,
                    depth_max,
                    frame_width,
                    frame_height,
                )
                detection["is_ahead"] = (
                    detection["zone"] == "center" and detection["in_roi"]
                )
                if not 0.0 <= detection["risk"] <= 1.0:
                    implementation_issues.append(
                        {
                            "image_name": image_path.name,
                            "class_name": class_name,
                            "issue": f"risk outside [0, 1]: {detection['risk']}",
                        }
                    )
                if not 0.0 <= detection["norm_depth"] <= 1.0:
                    implementation_issues.append(
                        {
                            "image_name": image_path.name,
                            "class_name": class_name,
                            "issue": f"normalized depth outside [0, 1]: {detection['norm_depth']}",
                        }
                    )
                detections.append(detection)
                x1, y1, x2, y2 = xyxy
                detection_rows.append(
                    {
                        "image_name": image_path.name,
                        "image_path": str(image_path),
                        "class_name": class_name,
                        "confidence": confidence,
                        "raw_depth": raw_depth,
                        "normalized_depth": detection["normalized_depth"],
                        "tier": detection["tier"],
                        "risk": detection["risk"],
                        "zone": detection["zone"],
                        "in_roi": detection["in_roi"],
                        "is_ahead": detection["is_ahead"],
                        "box_x1": x1,
                        "box_y1": y1,
                        "box_x2": x2,
                        "box_y2": y2,
                    }
                )

        if len(detections) != count:
            implementation_issues.append(
                {
                    "image_name": image_path.name,
                    "issue": f"detection/depth detail mismatch: {count} boxes, {len(detections)} valid samples",
                }
            )

        ahead = get_ahead_objects(detections)
        alert = decide_alert(detections)
        left_risks = [detection["risk"] for detection in detections if detection["zone"] == "left"]
        right_risks = [detection["risk"] for detection in detections if detection["zone"] == "right"]
        max_left = max(left_risks, default=0.0)
        max_right = max(right_risks, default=0.0)
        max_ahead = max((detection["risk"] for detection in ahead), default=None)
        highest_overall = max(detections, key=lambda detection: detection["risk"], default=None)
        selected_alert = max(ahead, key=lambda detection: detection["risk"]) if alert else None
        same_highest = (
            selected_alert is not None
            and highest_overall is not None
            and detections.index(selected_alert) == detections.index(highest_overall)
        )

        flags = []
        if not alert and ahead:
            flags.append("no_alert_with_ahead_object")
        if alert and selected_alert and selected_alert["confidence"] < LOW_CONFIDENCE_REVIEW_LIMIT:
            flags.append("alert_low_confidence")
        if alert and selected_alert and (
            selected_alert["normalized_depth"] < EXTREME_DEPTH_LIMITS[0]
            or selected_alert["normalized_depth"] > EXTREME_DEPTH_LIMITS[1]
        ):
            flags.append("alert_extreme_normalized_depth")
        if alert and alert["direction"] == "stop":
            flags.append("stop_decision")
        if len(ahead) > 1:
            flags.append("multiple_ahead_competing_risks")
        left_count = sum(detection["zone"] == "left" for detection in detections)
        right_count = sum(detection["zone"] == "right" for detection in detections)
        if left_count and right_count and (left_count > 1 or right_count > 1):
            flags.append("multiple_side_objects_competing")
        if alert and selected_alert and highest_overall and not same_highest:
            flags.append("highest_overall_detection_not_alert_object")
        if (left_count or right_count) and not ahead:
            flags.append("side_objects_without_ahead_object")
        if count >= MANY_DETECTIONS_LIMIT:
            flags.append("many_detections")
        if count == 0:
            flags.append("zero_detections")

        alert_class = alert["class_name"] if alert else ""
        alert_risk = alert["risk"] if alert else ""
        urgency = alert["urgency"] if alert else ""
        direction = alert["direction"] if alert else ""
        image_rows.append(
            {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "image_width": frame_width,
                "image_height": frame_height,
                "num_detections": count,
                "num_ahead_objects": len(ahead),
                "max_ahead_risk": "" if max_ahead is None else max_ahead,
                "alert": alert is not None,
                "alert_class": alert_class,
                "alert_risk": alert_risk,
                "urgency": urgency,
                "direction": direction,
                "stop_required": bool(alert and alert["direction"] == "stop"),
                "max_left_risk": max_left,
                "max_right_risk": max_right,
                "highest_risk_detection_class": (
                    highest_overall["class_name"] if highest_overall else ""
                ),
                "highest_risk_detection_risk": (
                    highest_overall["risk"] if highest_overall else ""
                ),
                "alert_is_highest_risk_detection": same_highest if alert else "",
                "review_flags": ";".join(flags),
            }
        )

        if alert:
            if alert["direction"] == "stop":
                direction_reason = (
                    f"STOP because max_left={max_left:.6f} and "
                    f"max_right={max_right:.6f} are both >= {ALERT_THRESHOLD}"
                )
            elif alert["direction"] == "step left":
                direction_reason = (
                    f"step left because max_left={max_left:.6f} "
                    f"is lower than max_right={max_right:.6f}"
                )
            else:
                direction_reason = (
                    f"step right because max_left={max_left:.6f} "
                    f"is >= max_right={max_right:.6f} or risks tie"
                )
            direction_rows.append(
                {
                    "image_name": image_path.name,
                    "alert_class": alert["class_name"],
                    "alert_risk": alert["risk"],
                    "max_left_risk": max_left,
                    "max_right_risk": max_right,
                    "direction": alert["direction"],
                    "both_sides_at_or_above_threshold": (
                        max_left >= ALERT_THRESHOLD and max_right >= ALERT_THRESHOLD
                    ),
                    "reason": direction_reason,
                }
            )

        if flags:
            prior = prior_reviews.get(image_path.name, {})
            failure_rows.append(
                {
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "review_flags": ";".join(flags),
                    "manual_review_classification": prior.get(
                        "manual_review_classification", ""
                    ),
                    "manual_review_observation": prior.get(
                        "manual_review_observation", ""
                    ),
                }
            )

        annotated = overlay_frame(img_bgr, detections, alert)
        output_image = OUTPUT_DIR / "annotated" / f"{image_path.stem}_annotated.jpg"
        if not cv2.imwrite(str(output_image), annotated):
            raise OSError(f"Could not write annotated validation image: {output_image}")
        print(
            f"[{index:02d}/{IMAGE_COUNT}] {image_path.name}: "
            f"detections={count}, ahead={len(ahead)}, "
            f"max_ahead={max_ahead}, alert={alert['message'] if alert else 'silent'}"
        )

    write_csv(OUTPUT_DIR / "validation_results.csv", IMAGE_FIELDS, image_rows)
    write_csv(OUTPUT_DIR / "detection_details.csv", DETECTION_FIELDS, detection_rows)
    write_csv(
        OUTPUT_DIR / "failure_cases" / "review_candidates.csv",
        [
            "image_name",
            "image_path",
            "review_flags",
            "manual_review_classification",
            "manual_review_observation",
        ],
        failure_rows,
    )
    manual_rows = [
        {
            "image_name": row["image_name"],
            "image_path": row["image_path"],
            "review_flags": row["review_flags"],
            "manual_review_classification": row["manual_review_classification"],
            "manual_review_observation": row["manual_review_observation"],
        }
        for row in failure_rows
    ]
    manual_names = {row["image_name"] for row in manual_rows}
    manual_rows.extend(
        prior
        for name, prior in prior_reviews.items()
        if name not in manual_names
    )
    write_csv(
        prior_review_path,
        [
            "image_name",
            "image_path",
            "review_flags",
            "manual_review_classification",
            "manual_review_observation",
        ],
        manual_rows,
    )
    write_csv(
        OUTPUT_DIR / "metrics" / "direction_checks.csv",
        [
            "image_name",
            "alert_class",
            "alert_risk",
            "max_left_risk",
            "max_right_risk",
            "direction",
            "both_sides_at_or_above_threshold",
            "reason",
        ],
        direction_rows,
    )
    manifest_rows = [
        {
            "selection_index": index,
            "image_name": path.name,
            "image_path": str(path),
            "sha256_before": image_hashes[path],
            "selection_method": "30 evenly spaced indices over numerically sorted RSUD20K test image filenames",
        }
        for index, path in enumerate(selected, start=1)
    ]
    write_csv(
        OUTPUT_DIR / "metrics" / "selection_manifest.csv",
        ["selection_index", "image_name", "image_path", "sha256_before", "selection_method"],
        manifest_rows,
    )

    summary = save_distributions(image_rows, detection_rows, OUTPUT_DIR)
    classifications = Counter(
        row["manual_review_classification"]
        for row in manual_rows
        if row["manual_review_classification"]
    )
    summary["experiment"]["manual_visual_review"] = {
        "flagged_candidates_reviewed": len(failure_rows),
        "flagged_classifications": dict(sorted(classifications.items())),
        "additional_manual_concerns": sum(
            row["image_name"] not in {case["image_name"] for case in failure_rows}
            for row in manual_rows
        ),
        "possible_failure_cases": classifications["possible failure case"],
    }
    hashes_unchanged = all(sha256_file(path) == image_hashes[path] for path in selected)
    checkpoint_hash_after = sha256_file(CHECKPOINT)
    summary["experiment"] = {
        "image_count": IMAGE_COUNT,
        "dataset_test_dir": str(DATASET_TEST_DIR.resolve()),
        "selection_method": (
            "Select 30 unique, evenly spaced indices including first and last from "
            "numerically sorted testN image filenames."
        ),
        "device": device,
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "checkpoint_unchanged": checkpoint_hash_before == checkpoint_hash_after,
        "selected_image_hashes_unchanged": hashes_unchanged,
        "implementation_issues": implementation_issues,
        "review_candidate_images": len(failure_rows),
        "review_candidate_flags": dict(
            Counter(
                flag
                for row in failure_rows
                for flag in row["review_flags"].split(";")
                if flag
            )
        ),
    }
    (OUTPUT_DIR / "metrics" / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Completed {IMAGE_COUNT} images; {summary['detections']['total']} detections; "
        f"{summary['images']['with_alerts']} alerts; "
        f"{len(implementation_issues)} implementation issue(s); "
        f"source hashes unchanged={hashes_unchanged}; "
        f"checkpoint unchanged={summary['experiment']['checkpoint_unchanged']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

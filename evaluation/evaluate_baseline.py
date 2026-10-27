"""Evaluate the archived B1 checkpoint on the held-out test split only."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
import ultralytics
import torch

from yolo_to_coco import (
    DEFAULT_CLASSES,
    DEFAULT_LABELS,
    DEFAULT_OUTPUT as DEFAULT_GT,
    DEFAULT_SPLIT,
    convert_split,
    read_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = PROJECT_ROOT / "results" / "previous_b1" / "weights" / "best.pt"
DATA_YAML = PROJECT_ROOT / "models" / "detector" / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "results" / "b1_baseline_eval"
EXPECTED_NAMES = [
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


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_test_distribution(stats: dict, output_dir: Path) -> None:
    rows = [
        {
            "class": stats["class_names"][class_id],
            "image_count": stats["class_image_counts"][class_id],
            "annotation_count": stats["class_annotation_counts"][class_id],
        }
        for class_id in range(len(stats["class_names"]))
    ]
    write_csv(output_dir / "test_class_distribution.csv", list(rows[0]), rows)


def export_ultralytics_metrics(metrics, output_dir: Path, class_names: list[str]) -> dict:
    box_metrics = metrics.box
    ap_indexes = {int(class_id): index for index, class_id in enumerate(box_metrics.ap_class_index)}
    rows = []
    for class_id, class_name in enumerate(class_names):
        if class_id in ap_indexes:
            precision, recall, ap50, ap5095 = box_metrics.class_result(ap_indexes[class_id])
            values = [float(precision), float(recall), float(ap50), float(ap5095)]
        else:
            values = [float("nan")] * 4
        rows.append(
            {
                "class": class_name,
                "precision": values[0],
                "recall": values[1],
                "mAP50": values[2],
                "mAP50-95": values[3],
            }
        )
    write_csv(output_dir / "per_class_metrics.csv", list(rows[0]), rows)
    write_csv(
        output_dir / "per_class_map50.csv",
        ["class", "mAP50"],
        [{"class": row["class"], "mAP50": row["mAP50"]} for row in rows],
    )
    return {key: float(value) for key, value in metrics.results_dict.items()}


def validate_and_remap_predictions(
    raw_path: Path,
    output_path: Path,
    image_name_to_id: dict[str, int],
) -> tuple[list[dict], dict]:
    predictions = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError(f"Expected Ultralytics predictions JSON list: {raw_path}")
    remapped = []
    missing_names = []
    invalid = Counter()
    degenerate_box_count = 0
    seen_image_ids = set()
    for prediction in predictions:
        image_name = prediction.get("file_name")
        image_id = image_name_to_id.get(image_name)
        if image_id is None:
            missing_names.append(image_name)
            continue
        category_id = prediction.get("category_id")
        if not isinstance(category_id, int) or not 1 <= category_id <= 15:
            invalid["category_id"] += 1
            continue
        bbox = prediction.get("bbox")
        score = prediction.get("score")
        if not isinstance(bbox, list) or len(bbox) != 4:
            invalid["bbox_shape"] += 1
            continue
        try:
            x, y, width, height = map(float, bbox)
            confidence = float(score)
        except (TypeError, ValueError):
            invalid["nonnumeric"] += 1
            continue
        if not all(math.isfinite(value) for value in (x, y, width, height, confidence)):
            invalid["nonfinite"] += 1
            continue
        if not 0 <= confidence <= 1:
            invalid["bbox_or_score_range"] += 1
            continue
        if width <= 0 or height <= 0:
            degenerate_box_count += 1
            continue
        remapped.append(
            {
                "image_id": image_id,
                "category_id": category_id - 1,
                "bbox": [x, y, width, height],
                "score": confidence,
            }
        )
        seen_image_ids.add(image_id)
    if missing_names or invalid:
        raise ValueError(
            f"Prediction validation failed: unmatched filenames={missing_names[:5]}, "
            f"invalid={dict(invalid)}"
        )
    output_path.write_text(json.dumps(remapped, separators=(",", ":")), encoding="utf-8")
    return remapped, {
        "prediction_count": len(remapped),
        "images_with_predictions": len(seen_image_ids),
        "prediction_image_ids": seen_image_ids,
        "degenerate_box_count": degenerate_box_count,
        "invalid_prediction_count": sum(invalid.values()),
    }


def validate_coco_ground_truth(coco_data: dict, stats: dict, image_paths: list[Path]) -> None:
    if len(coco_data["images"]) != stats["image_count"] or len(coco_data["annotations"]) != stats["annotation_count"]:
        raise ValueError("COCO GT record counts do not match the YOLO test split")
    if [category["id"] for category in coco_data["categories"]] != list(range(15)):
        raise ValueError("COCO category IDs are not exactly 0–14")
    if [category["name"] for category in coco_data["categories"]] != EXPECTED_NAMES:
        raise ValueError("COCO category names do not match the expected taxonomy")
    if len(image_paths) != len(coco_data["images"]):
        raise ValueError("COCO image count does not match test.txt")
    for image_id, (record, path) in enumerate(zip(coco_data["images"], image_paths), start=1):
        if record["id"] != image_id or record["file_name"] != path.name or not path.is_file():
            raise ValueError(f"COCO image ID/path mismatch at record {image_id}")
    valid_ids = set(range(15))
    for annotation in coco_data["annotations"]:
        x, y, width, height = annotation["bbox"]
        if annotation["image_id"] not in range(1, len(image_paths) + 1):
            raise ValueError(f"Unknown GT image ID: {annotation['image_id']}")
        if annotation["category_id"] not in valid_ids:
            raise ValueError(f"Unknown GT category ID: {annotation['category_id']}")
        if width <= 0 or height <= 0 or not math.isclose(annotation["area"], width * height, rel_tol=1e-12):
            raise ValueError(f"Invalid GT bbox/area in annotation {annotation['id']}")
        if not all(math.isfinite(value) for value in (x, y, width, height, annotation["area"])):
            raise ValueError(f"Non-finite GT bbox/area in annotation {annotation['id']}")


def run_cocoeval(gt_path: Path, predictions_path: Path, output_dir: Path) -> dict:
    ground_truth = COCO(str(gt_path))
    detections = ground_truth.loadRes(str(predictions_path))
    eval_results = {}
    for label, thresholds in (("AP50-95", np.linspace(0.5, 0.95, 10)), ("AP50", np.array([0.5]))):
        evaluator = COCOeval(ground_truth, detections, iouType="bbox")
        evaluator.params.iouThrs = thresholds
        evaluator.params.areaRng = [
            [0.0, 1.0e10],
            [0.0, 32.0**2],
            [32.0**2, 96.0**2],
            [96.0**2, 1.0e10],
        ]
        evaluator.params.areaRngLbl = ["all", "small", "medium", "large"]
        evaluator.params.imgIds = sorted(image["id"] for image in ground_truth.dataset["images"])
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        stat_indexes = (3, 4, 5)
        eval_results[("all", label)] = float(evaluator.stats[0 if label == "AP50-95" else 1])
        if label == "AP50-95":
            eval_results[("all", "AP75")] = float(evaluator.stats[2])
        for size, stat_index in zip(("small", "medium", "large"), stat_indexes):
            eval_results[(size, label)] = float(evaluator.stats[stat_index])
        if label == "AP50-95":
            per_class = {}
            precision = evaluator.eval["precision"]
            # precision dimensions: IoU, recall threshold, category, area range, max detections.
            for category_index, category in enumerate(ground_truth.dataset["categories"]):
                values = precision[:, :, category_index, 0, -1]
                valid = values[values > -1]
                per_class[category["id"]] = float(np.mean(valid)) if valid.size else float("nan")
            eval_results["coco_per_class_map50_95"] = per_class
    (output_dir / "cocoeval_summary.txt").write_text(
        "\n".join(
            [
                f"AP50-95 (all): {eval_results[('all', 'AP50-95')]:.6f}",
                f"AP50: {eval_results[('all', 'AP50')]:.6f}",
                f"AP75: {eval_results[('all', 'AP75')]:.6f}",
                f"AP50-95 (small): {eval_results[('small', 'AP50-95')]:.6f}",
                f"AP50-95 (medium): {eval_results[('medium', 'AP50-95')]:.6f}",
                f"AP50-95 (large): {eval_results[('large', 'AP50-95')]:.6f}",
                f"AP50 (small): {eval_results[('small', 'AP50')]:.6f}",
                f"AP50 (medium): {eval_results[('medium', 'AP50')]:.6f}",
                f"AP50 (large): {eval_results[('large', 'AP50')]:.6f}",
            ]
        ),
        encoding="utf-8",
    )
    return eval_results


def main() -> int:
    owned_outputs = {
        "test_ground_truth_coco.json",
        "test_class_distribution.csv",
        "predictions.json",
        "predictions_cocoeval.json",
        "prediction_validation.json",
        "per_class_metrics.csv",
        "per_class_map50.csv",
        "overall_metrics.json",
        "overall_metrics.csv",
        "size_breakdown.csv",
        "cocoeval_summary.txt",
        "BoxF1_curve.png",
        "BoxPR_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "val_batch0_labels.jpg",
        "val_batch0_pred.jpg",
        "val_batch1_labels.jpg",
        "val_batch1_pred.jpg",
        "val_batch2_labels.jpg",
        "val_batch2_pred.jpg",
    }
    if OUTPUT_DIR.exists():
        unexpected = [path.name for path in OUTPUT_DIR.iterdir() if path.name not in owned_outputs]
        if unexpected:
            raise FileExistsError(
                f"Refusing to overwrite unrelated evaluation output in {OUTPUT_DIR}: {unexpected}"
            )
    for required in (CHECKPOINT, DATA_YAML, DEFAULT_SPLIT, DEFAULT_CLASSES):
        if not required.is_file():
            raise FileNotFoundError(f"Required evaluation input missing: {required}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = read_split(DEFAULT_SPLIT)
    coco_gt, gt_stats = convert_split()
    validate_coco_ground_truth(coco_gt, gt_stats, image_paths)
    DEFAULT_GT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_GT.write_text(json.dumps(coco_gt, separators=(",", ":")), encoding="utf-8")
    export_test_distribution(gt_stats, OUTPUT_DIR)

    checkpoint = YOLO(str(CHECKPOINT))
    checkpoint_names = [checkpoint.names[index] for index in range(len(checkpoint.names))]
    if checkpoint_names != EXPECTED_NAMES:
        raise ValueError(f"Checkpoint taxonomy mismatch: {checkpoint_names}")

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Dataset config: {DATA_YAML}")
    print(f"Split: test ({len(image_paths)} images)")
    print(f"Ground truth annotations: {gt_stats['annotation_count']}")
    print("Evaluation device: CPU")
    print("No training or inference-time augmentation requested.")

    metrics = checkpoint.val(
        data=str(DATA_YAML),
        split="test",
        device="cpu",
        batch=2,
        workers=0,
        imgsz=640,
        save_json=True,
        plots=True,
        project=str(PROJECT_ROOT / "results"),
        name="b1_baseline_eval",
        exist_ok=True,
    )
    if Path(metrics.save_dir).resolve() != OUTPUT_DIR.resolve():
        raise RuntimeError(f"Ultralytics used unexpected save directory: {metrics.save_dir}")

    overall = export_ultralytics_metrics(metrics, OUTPUT_DIR, EXPECTED_NAMES)
    (OUTPUT_DIR / "overall_metrics.json").write_text(
        json.dumps(
            {
                "split": "test",
                "checkpoint": str(CHECKPOINT),
                "metrics": overall,
                "ultralytics_version": ultralytics.__version__,
                "torch_version": torch.__version__,
                "device": "cpu",
                "test_image_count": gt_stats["image_count"],
                "test_annotation_count": gt_stats["annotation_count"],
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    write_csv(
        OUTPUT_DIR / "overall_metrics.csv",
        ["metric", "value"],
        [{"metric": key, "value": value} for key, value in overall.items()],
    )

    raw_predictions_path = OUTPUT_DIR / "predictions.json"
    if not raw_predictions_path.is_file():
        raise FileNotFoundError(
            f"Ultralytics did not save COCO predictions JSON despite save_json=True: {raw_predictions_path}"
        )
    predictions, prediction_stats = validate_and_remap_predictions(
        raw_predictions_path,
        OUTPUT_DIR / "predictions_cocoeval.json",
        gt_stats["image_name_to_id"],
    )
    if any(item["image_id"] not in set(range(1, gt_stats["image_count"] + 1)) for item in predictions):
        raise ValueError("Prediction image ID is not represented in held-out test GT")
    (OUTPUT_DIR / "prediction_validation.json").write_text(
        json.dumps(
            {
                "raw_prediction_count": len(json.loads(raw_predictions_path.read_text(encoding="utf-8"))),
                **{
                    key: value
                    for key, value in prediction_stats.items()
                    if key != "prediction_image_ids"
                },
                "note": (
                    "Degenerate zero/non-positive-width-or-height detections were omitted from the "
                    "COCOeval input. The raw Ultralytics predictions.json is retained unchanged."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    size_metrics = run_cocoeval(
        DEFAULT_GT,
        OUTPUT_DIR / "predictions_cocoeval.json",
        OUTPUT_DIR,
    )
    size_rows = [
        {
            "size_category": size,
            "AP50": size_metrics[(size, "AP50")],
            "AP50_95": size_metrics[(size, "AP50-95")],
        }
        for size in ("small", "medium", "large")
    ]
    write_csv(OUTPUT_DIR / "size_breakdown.csv", list(size_rows[0]), size_rows)

    metrics_names = list(overall)
    print("Overall metrics:", overall)
    print("Prediction records:", prediction_stats["prediction_count"])
    print("Images with predictions:", prediction_stats["images_with_predictions"])
    print("Degenerate prediction boxes omitted from COCOeval:", prediction_stats["degenerate_box_count"])
    print("Size metrics:", size_rows)
    print("Per-class metrics:", (OUTPUT_DIR / "per_class_metrics.csv"))
    print("Ultralytics metrics available:", metrics_names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

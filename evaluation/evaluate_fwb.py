"""Evaluate a trained Sprint 13 FWB checkpoint on the held-out test split."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

from models.detector.fwb_registration import register_fwb

import evaluate_baseline as baseline
from yolo_to_coco import convert_split, read_split

DATA_YAML = PROJECT_ROOT / "models" / "detector" / "data.yaml"
CHECKPOINT = PROJECT_ROOT / "results" / "fwb_optimization_full" / "weights" / "best.pt"
BASELINE_GT = PROJECT_ROOT / "results" / "b1_baseline_eval" / "test_ground_truth_coco.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "fwb_optimization_full" / "evaluation"
EXPECTED_CLASSES = baseline.EXPECTED_NAMES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_same_ground_truth(coco_gt: dict, baseline_path: Path) -> None:
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Sprint 5 ground-truth reference missing: {baseline_path}")
    baseline_gt = json.loads(baseline_path.read_text(encoding="utf-8"))
    for key in ("images", "annotations", "categories"):
        if coco_gt[key] != baseline_gt.get(key):
            raise ValueError(
                f"FWB {key} differ from the Sprint 5 B1 test protocol; "
                "evaluation stopped to prevent an invalid comparison"
            )


def main() -> int:
    for path in (DATA_YAML, CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(f"Required FWB evaluation input missing: {path}")
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing FWB evaluation output: {OUTPUT_DIR}")

    data_config = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    split_path = Path(data_config["test"])
    if not split_path.is_absolute():
        split_path = Path(data_config["path"]) / split_path
    test_images = read_split(split_path)
    if len(test_images) != 750:
        raise ValueError(f"Sprint 5 protocol expects 750 test images, found {len(test_images)}")
    coco_gt, gt_stats = convert_split(
        split_file=split_path,
        classes_file=baseline.DEFAULT_CLASSES,
        labels_dir=baseline.DEFAULT_LABELS,
    )
    baseline.validate_coco_ground_truth(coco_gt, gt_stats, test_images)
    _validate_same_ground_truth(coco_gt, BASELINE_GT)

    register_fwb()
    model = YOLO(str(CHECKPOINT), task="detect")
    class_names = [model.names[index] for index in range(len(model.names))]
    if class_names != EXPECTED_CLASSES:
        raise ValueError(f"FWB checkpoint class taxonomy mismatch: {class_names}")
    if int(model.model.model[-1].nc) != 15:
        raise ValueError(f"FWB checkpoint must have 15 classes, found {model.model.model[-1].nc}")

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(data_config["path"]) / "labels.cache"
    cache_existed = cache_path.exists()
    print(f"FWB checkpoint: {CHECKPOINT}")
    print(f"Test split: {split_path} ({len(test_images)} images)")
    print(f"Annotations: {gt_stats['annotation_count']}")
    print("Evaluation protocol: CPU, imgsz=640, batch=2, workers=0, conf=0.001, iou=0.7, max_det=300")

    try:
        metrics = model.val(
            data=str(DATA_YAML),
            split="test",
            device="cpu",
            batch=2,
            workers=0,
            imgsz=640,
            conf=0.001,
            iou=0.7,
            max_det=300,
            augment=False,
            cache=False,
            save_json=True,
            plots=True,
            project=str(OUTPUT_DIR.parent),
            name=OUTPUT_DIR.name,
            exist_ok=False,
        )
        actual_output = Path(metrics.save_dir).resolve()
        if actual_output != OUTPUT_DIR.resolve():
            raise RuntimeError(f"Ultralytics saved to unexpected directory: {actual_output}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        gt_path = OUTPUT_DIR / "test_ground_truth_coco.json"
        gt_path.write_text(json.dumps(coco_gt, separators=(",", ":")), encoding="utf-8")
        baseline.export_test_distribution(gt_stats, OUTPUT_DIR)
        overall = baseline.export_ultralytics_metrics(metrics, OUTPUT_DIR, EXPECTED_CLASSES)
        overall_payload = {
            "split": "test",
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": _sha256(CHECKPOINT),
            "dataset_config": str(DATA_YAML),
            "test_split_file": str(split_path.resolve()),
            "test_split_sha256": _sha256(split_path.resolve()),
            "metrics": overall,
            "ultralytics_version": ultralytics.__version__,
            "torch_version": torch.__version__,
            "device": "cpu",
            "imgsz": 640,
            "batch": 2,
            "workers": 0,
            "confidence": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "test_image_count": gt_stats["image_count"],
            "test_annotation_count": gt_stats["annotation_count"],
        }
        (OUTPUT_DIR / "overall_metrics.json").write_text(
            json.dumps(overall_payload, indent=2, allow_nan=False), encoding="utf-8"
        )
        baseline.write_csv(
            OUTPUT_DIR / "overall_metrics.csv",
            ["metric", "value"],
            [{"metric": key, "value": value} for key, value in overall.items()],
        )

        raw_path = OUTPUT_DIR / "predictions.json"
        if not raw_path.is_file():
            raise FileNotFoundError(f"Ultralytics raw predictions missing: {raw_path}")
        predictions, prediction_stats = baseline.validate_and_remap_predictions(
            raw_path,
            OUTPUT_DIR / "predictions_cocoeval.json",
            gt_stats["image_name_to_id"],
        )
        (OUTPUT_DIR / "prediction_validation.json").write_text(
            json.dumps(
                {
                    "raw_prediction_count": len(json.loads(raw_path.read_text(encoding="utf-8"))),
                    **{
                        key: value
                        for key, value in prediction_stats.items()
                        if key != "prediction_image_ids"
                    },
                    "note": (
                        "Degenerate zero/non-positive-area predictions are excluded only from "
                        "the derived COCOeval input. Raw predictions.json is retained unchanged."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        baseline.run_cocoeval(
            gt_path,
            OUTPUT_DIR / "predictions_cocoeval.json",
            OUTPUT_DIR,
        )
        print(json.dumps(overall, indent=2))
        print(f"Predictions validated: {prediction_stats['prediction_count']}")
        print(f"Degenerate boxes excluded from COCOeval: {prediction_stats['degenerate_box_count']}")
        print(f"FWB held-out evaluation saved to {OUTPUT_DIR}")
    finally:
        if not cache_existed and cache_path.exists():
            cache_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

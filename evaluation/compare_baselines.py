"""Compare recorded Sprint 5 B1 metrics with an evaluated Sprint 13 FWB run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

EXPECTED_CLASS_COUNT = 15
REQUIRED_METRICS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
B1_DIR = PROJECT_ROOT / "results" / "b1_baseline_eval"
FWB_DIR = PROJECT_ROOT / "results" / "fwb_optimization_full" / "evaluation"
ARCH_METRICS = PROJECT_ROOT / "results" / "sprint13_architecture_metrics.json"
OUTPUT = PROJECT_ROOT / "results" / "sprint13_fwb_comparison.csv"
PER_CLASS_OUTPUT = PROJECT_ROOT / "results" / "sprint13_fwb_per_class_comparison.csv"
METRICS = (
    ("Precision", "metrics/precision(B)"),
    ("Recall", "metrics/recall(B)"),
    ("mAP50", "metrics/mAP50(B)"),
    ("mAP50-95", "metrics/mAP50-95(B)"),
)


def read_metric_json(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Recorded metric file missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Expected metrics object in {path}")
    return {key: float(value) for key, value in metrics.items()}


def read_test_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != "test" or payload.get("test_image_count") != 750:
        raise ValueError(
            f"Expected held-out 750-image test metrics in {path}; "
            f"split={payload.get('split')!r}, images={payload.get('test_image_count')!r}"
        )
    metrics = read_metric_json(path)
    missing = [key for key in REQUIRED_METRICS if key not in metrics]
    nonfinite = [key for key in REQUIRED_METRICS if key in metrics and not math.isfinite(metrics[key])]
    if missing or nonfinite:
        raise ValueError(f"Invalid held-out metrics in {path}: missing={missing}, nonfinite={nonfinite}")
    return metrics


def read_per_class(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Per-class metrics missing: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_CLASS_COUNT or len({row["class"] for row in rows}) != EXPECTED_CLASS_COUNT:
        raise ValueError(f"Expected {EXPECTED_CLASS_COUNT} unique class rows in {path}, got {len(rows)}")
    metrics = {
        row["class"]: {
            key: float(row[key])
            for key in ("precision", "recall", "mAP50", "mAP50-95")
        }
        for row in rows
    }
    if any(
        not math.isfinite(value)
        for class_metrics in metrics.values()
        for value in class_metrics.values()
    ):
        raise ValueError(f"Non-finite per-class metric in {path}")
    return metrics


def _number(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return format(value, ".17g")


def _delta_text(delta: float | None, *, lower_is_better: bool = False) -> str:
    if delta is None:
        return "pending"
    improved = delta < 0 if lower_is_better else delta > 0
    if improved:
        return "improved"
    declined = delta > 0 if lower_is_better else delta < 0
    if declined:
        return "declined"
    return "unchanged"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()

    b1_metrics = read_test_metrics(B1_DIR / "overall_metrics.json")
    b1_classes = read_per_class(B1_DIR / "per_class_metrics.csv")
    fwb_metrics_path = FWB_DIR / "overall_metrics.json"
    fwb_classes_path = FWB_DIR / "per_class_metrics.csv"
    has_fwb_evaluation = fwb_metrics_path.is_file() and fwb_classes_path.is_file()
    has_partial_fwb_evaluation = fwb_metrics_path.exists() or fwb_classes_path.exists()
    if has_partial_fwb_evaluation and not has_fwb_evaluation:
        raise FileNotFoundError(
            "FWB evaluation output is incomplete: both overall_metrics.json and "
            "per_class_metrics.csv are required before comparison."
        )
    if not has_fwb_evaluation and not args.allow_pending:
        raise FileNotFoundError(
            "FWB held-out evaluation is not available. Run evaluation/evaluate_fwb.py "
            "after a completed training run, or pass --allow-pending to write B1-only rows."
        )

    fwb_metrics = read_test_metrics(fwb_metrics_path) if has_fwb_evaluation else {}
    fwb_classes = read_per_class(fwb_classes_path) if has_fwb_evaluation else {}
    if has_fwb_evaluation and set(fwb_classes) != set(b1_classes):
        raise ValueError("FWB per-class metrics do not contain the exact B1 class set")
    arch = (
        json.loads(ARCH_METRICS.read_text(encoding="utf-8"))
        if ARCH_METRICS.is_file()
        else {}
    )
    arch_passed = arch.get("status") == "passed"

    comparison_rows: list[dict[str, str]] = []
    for label, key in METRICS:
        baseline_value = b1_metrics[key]
        fwb_value = fwb_metrics.get(key)
        delta = fwb_value - baseline_value if fwb_value is not None else None
        comparison_rows.append(
            {
                "metric": label,
                "B1": _number(baseline_value),
                "B1+FWB": _number(fwb_value),
                "delta_FWB_minus_B1": _number(delta),
                "interpretation": _delta_text(delta),
                "status": "complete" if delta is not None else "pending_training_and_evaluation",
            }
        )
    for display, class_name in (
        ("Pothole AP50", "pothole"),
        ("Traffic cone AP50", "traffic cone"),
    ):
        baseline_value = b1_classes[class_name]["mAP50"]
        fwb_value = fwb_classes.get(class_name, {}).get("mAP50")
        delta = fwb_value - baseline_value if fwb_value is not None else None
        comparison_rows.append(
            {
                "metric": display,
                "B1": _number(baseline_value),
                "B1+FWB": _number(fwb_value),
                "delta_FWB_minus_B1": _number(delta),
                "interpretation": _delta_text(delta),
                "status": "complete" if delta is not None else "pending_training_and_evaluation",
            }
        )
    for display, baseline_key, fwb_key in (
        ("Parameter count", "stock_parameters", "fwb_parameters"),
        ("GFLOPs at 640", "stock_gflops", "fwb_gflops"),
    ):
        baseline_value = float(arch[baseline_key]) if arch_passed else None
        fwb_value = float(arch[fwb_key]) if arch_passed else None
        delta = fwb_value - baseline_value if baseline_value is not None else None
        comparison_rows.append(
            {
                "metric": display,
                "B1": _number(baseline_value),
                "B1+FWB": _number(fwb_value),
                "delta_FWB_minus_B1": _number(delta),
                "interpretation": _delta_text(delta, lower_is_better=True),
                "status": "complete" if delta is not None else "pending_architecture_gate",
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["metric", "B1", "B1+FWB", "delta_FWB_minus_B1", "interpretation", "status"]
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison_rows)

    classes = sorted(b1_classes)
    if len(classes) != 15:
        raise ValueError(f"Expected 15 B1 classes, found {len(classes)}")
    per_class_fields = [
        "class",
        "metric",
        "B1",
        "B1+FWB",
        "delta_FWB_minus_B1",
        "interpretation",
        "status",
    ]
    per_class_rows = []
    for class_name in classes:
        for metric in ("precision", "recall", "mAP50", "mAP50-95"):
            baseline_value = b1_classes[class_name][metric]
            fwb_value = fwb_classes.get(class_name, {}).get(metric)
            delta = fwb_value - baseline_value if fwb_value is not None else None
            per_class_rows.append(
                {
                    "class": class_name,
                    "metric": metric,
                    "B1": _number(baseline_value),
                    "B1+FWB": _number(fwb_value),
                    "delta_FWB_minus_B1": _number(delta),
                    "interpretation": _delta_text(delta),
                    "status": "complete" if delta is not None else "pending_training_and_evaluation",
                }
            )
    with PER_CLASS_OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=per_class_fields)
        writer.writeheader()
        writer.writerows(per_class_rows)

    print(f"Comparison saved: {OUTPUT}")
    print(f"Per-class comparison saved: {PER_CLASS_OUTPUT}")
    print("FWB evaluation status:", "complete" if has_fwb_evaluation else "pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

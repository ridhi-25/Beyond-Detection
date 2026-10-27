"""GPU-ready plain YOLOv8n B1 training entry point; do not run 50 epochs locally."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "models" / "detector" / "data.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="b1_baseline")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = args.data.resolve()
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 1:
        raise ValueError("epochs, batch, and imgsz must be positive integers")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {data_path}")

    print("B1 plain YOLOv8n training configuration")
    print("Model: yolov8n.pt")
    print(f"Dataset: {data_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch: {args.batch}")
    print("Optimizer: SGD")
    print("Initial learning rate: 0.01")
    print(f"Device: {args.device}")
    print(f"Run name: {args.name}")
    print("Architecture: plain YOLOv8n detector; no depth, fusion, scoring, or tracking")
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="SGD",
        lr0=0.01,
        device=args.device,
        name=args.name,
        project=str(PROJECT_ROOT / "results"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

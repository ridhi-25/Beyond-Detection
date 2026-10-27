"""Run untouched pretrained YOLOv8n COCO inference on a few RSUD20K images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import torch
import ultralytics
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "sprint3_pretrained_check.jpg"
MULTI_OUTPUT_DIR = PROJECT_ROOT / "results" / "sprint3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        type=Path,
        help="Path to one RSUD20K image to run inference on.",
    )
    parser.add_argument(
        "--additional",
        type=Path,
        nargs="*",
        default=[],
        help="Optional additional images (the normal maximum is three total).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_paths = [args.image, *args.additional]
    if len(image_paths) > 3:
        print("Error: Sprint 3 permits at most three images per run.", file=sys.stderr)
        return 2

    resolved_paths: list[Path] = []
    for image_path in image_paths:
        resolved = image_path.expanduser().resolve()
        if not resolved.is_file():
            print(f"Error: image does not exist: {resolved}", file=sys.stderr)
            return 2
        resolved_paths.append(resolved)

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("Inference device: CPU")
    print("Loading untouched pretrained COCO weights: yolov8n.pt")
    model = YOLO("yolov8n.pt")

    output_paths = (
        [DEFAULT_OUTPUT]
        if len(resolved_paths) == 1
        else [
            MULTI_OUTPUT_DIR / f"image_{index}_annotated.jpg"
            for index in range(1, len(resolved_paths) + 1)
        ]
    )
    for image_path, output_path in zip(resolved_paths, output_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: OpenCV could not load image: {image_path}", file=sys.stderr)
            return 1
        height, width = image.shape[:2]
        print(f"\nImage path: {image_path}")
        print(f"Image dimensions: {width}x{height}")

        results = model.predict(
            source=str(image_path),
            device="cpu",
            verbose=False,
        )
        if len(results) != 1:
            print(
                f"Error: expected one prediction result for this image, got {len(results)}.",
                file=sys.stderr,
            )
            return 1
        result = results[0]
        boxes = result.boxes
        detection_count = 0 if boxes is None else len(boxes)
        print(f"Detection count: {detection_count}")
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls.item())
                class_name = model.names[class_id]
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                print(
                    f"  {class_name}: confidence={confidence:.6f}, "
                    f"box_xyxy=({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})"
                )

        annotated = result.plot()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), annotated):
            print(f"Error: could not save annotated image: {output_path}", file=sys.stderr)
            return 1
        if not output_path.is_file() or output_path.stat().st_size == 0:
            print(f"Error: annotated output missing or empty: {output_path}", file=sys.stderr)
            return 1
        print(f"Annotated output: {output_path}")
        print(f"Output size: {output_path.stat().st_size} bytes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

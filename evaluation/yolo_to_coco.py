"""Convert one YOLO image-list split and its labels to COCO detection JSON."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = PROJECT_ROOT / "data" / "splits" / "test.txt"
DEFAULT_CLASSES = PROJECT_ROOT / "data" / "merged" / "classes.txt"
DEFAULT_LABELS = PROJECT_ROOT / "data" / "merged" / "labels"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "b1_baseline_eval" / "test_ground_truth_coco.json"


def read_split(path: Path) -> list[Path]:
    images = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        value = line.strip()
        if not value:
            continue
        image = Path(value).expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(f"Split image missing on line {line_number}: {image}")
        images.append(image)
    if len(images) != len(set(images)):
        raise ValueError(f"Duplicate image paths in split file: {path}")
    if not images:
        raise ValueError(f"Split file contains no image paths: {path}")
    return images


def convert_split(
    split_file: Path = DEFAULT_SPLIT,
    classes_file: Path = DEFAULT_CLASSES,
    labels_dir: Path = DEFAULT_LABELS,
) -> tuple[dict, dict]:
    class_names = [
        line.strip()
        for line in classes_file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(class_names) != 15:
        raise ValueError(f"Expected exactly 15 class names, found {len(class_names)}")
    if len(set(class_names)) != len(class_names):
        raise ValueError("Class names must be unique")

    image_paths = read_split(split_file)
    if len({image.name.casefold() for image in image_paths}) != len(image_paths):
        raise ValueError("Image basenames must be unique to construct prediction ID mapping")

    coco = {
        "info": {
            "description": "Derived COCO ground truth from the Beyond-Detection held-out test split",
            "source_split": str(split_file.resolve()),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": class_id, "name": name, "supercategory": "object"}
            for class_id, name in enumerate(class_names)
        ],
    }
    class_annotations = Counter()
    class_images: dict[int, set[int]] = defaultdict(set)
    annotation_id = 1
    out_of_bounds = Counter()
    image_name_to_id = {}

    for image_id, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )
        image_name_to_id[image_path.name] = image_id
        label_path = labels_dir.resolve() / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Label missing for test image {image_path}: {label_path}")
        for line_number, raw in enumerate(
            label_path.read_text(encoding="utf-8-sig", errors="strict").splitlines(),
            start=1,
        ):
            line = raw.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Expected 5 YOLO fields at {label_path}:{line_number}")
            try:
                class_id = int(fields[0])
                x_center, y_center, norm_width, norm_height = map(float, fields[1:])
            except ValueError as exc:
                raise ValueError(f"Invalid YOLO value at {label_path}:{line_number}") from exc
            coordinates = (x_center, y_center, norm_width, norm_height)
            if not 0 <= class_id < len(class_names):
                raise ValueError(f"Class ID {class_id} outside 0..14 at {label_path}:{line_number}")
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in coordinates):
                raise ValueError(f"Invalid normalized coordinates at {label_path}:{line_number}")
            if norm_width <= 0 or norm_height <= 0:
                raise ValueError(f"Non-positive box size at {label_path}:{line_number}")

            box_width = norm_width * width
            box_height = norm_height * height
            x_min = (x_center - norm_width / 2) * width
            y_min = (y_center - norm_height / 2) * height
            if box_width <= 0 or box_height <= 0 or not all(
                math.isfinite(value) for value in (x_min, y_min, box_width, box_height)
            ):
                raise ValueError(f"Invalid pixel bbox at {label_path}:{line_number}")
            if x_min < 0 or y_min < 0 or x_min + box_width > width or y_min + box_height > height:
                out_of_bounds[class_id] += 1

            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x_min, y_min, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            class_annotations[class_id] += 1
            class_images[class_id].add(image_id)
            annotation_id += 1

    stats = {
        "image_count": len(coco["images"]),
        "annotation_count": len(coco["annotations"]),
        "class_names": class_names,
        "class_annotation_counts": class_annotations,
        "class_image_counts": {class_id: len(class_images[class_id]) for class_id in range(15)},
        "out_of_bounds_by_class": out_of_bounds,
        "image_name_to_id": image_name_to_id,
        "split_file": split_file.resolve(),
    }
    return coco, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--classes-file", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    coco, stats = convert_split(args.split_file, args.classes_file, args.labels_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(coco, separators=(",", ":")), encoding="utf-8")
    print(f"COCO ground truth written: {args.output.resolve()}")
    print(f"Images: {stats['image_count']}; annotations: {stats['annotation_count']}")
    print(f"Class annotation counts: {dict(sorted(stats['class_annotation_counts'].items()))}")
    print(f"Class image counts: {stats['class_image_counts']}")
    print(f"Source boundary-overrun annotation incidences: {dict(stats['out_of_bounds_by_class'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

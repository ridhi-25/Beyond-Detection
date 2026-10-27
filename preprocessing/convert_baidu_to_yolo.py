"""Convert the supplied Baidu COCO-style road-obstacle annotations to YOLO."""

from __future__ import annotations

import argparse
import filecmp
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image

TARGET_CLASS_IDS = {"pothole": 13, "trafficcone": 14, "traffic cone": 14}
CLASS_NAMES = [
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


def normalized_category(name: str) -> str:
    return " ".join(name.casefold().replace("_", " ").split())


def image_path_for(source_root: Path, image_dir: Path, filename: str) -> Path:
    relative = PurePosixPath(filename.replace("\\", "/"))
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Unsafe COCO image filename: {filename!r}")
    candidates = [image_dir.joinpath(*relative.parts), source_root.joinpath(*relative.parts)]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_relative_to(source_root.resolve()) and resolved.is_file():
            return resolved
    return candidates[0]


def load_coco(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not all(
        isinstance(data.get(key), list) for key in ("images", "annotations", "categories")
    ):
        raise ValueError(f"{path} is not a COCO-style JSON with images/annotations/categories arrays")
    return data


def convert_coco_dataset(
    source_root: Path,
    output_root: Path,
    annotation_files: list[Path] | None = None,
) -> dict:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    image_dir = source_root / "is"
    annotation_files = annotation_files or sorted(source_root.rglob("*.json"))
    coco_files = []
    for path in annotation_files:
        data = load_coco(path)
        coco_files.append((path.resolve(), data))
    if not coco_files:
        raise ValueError(f"No COCO annotation JSONs found under {source_root}")

    images_dir = output_root / "images"
    labels_dir = output_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    category_names = {}
    records = []
    summary = {
        "source_image_count": 0,
        "source_annotation_count": 0,
        "category_counts": Counter(),
        "retained_category_counts": Counter(),
        "converted_category_counts": Counter(),
        "invalid_annotation_count": 0,
        "invalid_retained_annotation_count": 0,
        "missing_image_count": 0,
        "skipped_image_count": 0,
        "converted_image_count": 0,
        "split_summaries": {},
        "files": [],
    }
    all_filename_keys = set()

    for json_path, data in coco_files:
        categories = {}
        for category in data["categories"]:
            category_id = category.get("id")
            name = category.get("name")
            if not isinstance(category_id, int) or not isinstance(name, str):
                raise ValueError(f"Invalid category record in {json_path}: {category!r}")
            if category_id in categories:
                raise ValueError(f"Duplicate category ID {category_id} in {json_path}")
            categories[category_id] = name
            category_names[normalized_category(name)] = name

        split_name = json_path.stem.casefold()
        image_records = {}
        for image_record in data["images"]:
            image_id = image_record.get("id")
            filename = image_record.get("file_name")
            width, height = image_record.get("width"), image_record.get("height")
            if not isinstance(image_id, int) or not isinstance(filename, str):
                raise ValueError(f"Invalid COCO image record in {json_path}: {image_record!r}")
            if image_id in image_records:
                raise ValueError(f"Duplicate image ID {image_id} in {json_path}")
            image_records[image_id] = image_record
            filename_key = filename.casefold()
            if filename_key in all_filename_keys:
                raise ValueError(f"Image filename occurs in multiple COCO splits: {filename}")
            all_filename_keys.add(filename_key)

        annotations_by_image = defaultdict(list)
        for annotation in data["annotations"]:
            summary["source_annotation_count"] += 1
            category_id = annotation.get("category_id")
            if category_id not in categories:
                summary["invalid_annotation_count"] += 1
                continue
            category_name = normalized_category(categories[category_id])
            summary["category_counts"][categories[category_id]] += 1
            if category_name in TARGET_CLASS_IDS:
                summary["retained_category_counts"][category_name] += 1
                annotations_by_image[annotation.get("image_id")].append(annotation)

        summary["source_image_count"] += len(image_records)
        split_report = {
            "images": len(image_records),
            "annotations": len(data["annotations"]),
            "categories": categories,
            "retained_annotation_counts": Counter(),
            "invalid_annotations": 0,
            "missing_images": 0,
            "skipped_images": 0,
            "converted_images": 0,
        }
        for image_id, image_record in image_records.items():
            source_image = image_path_for(source_root, image_dir, image_record["file_name"])
            if not source_image.is_file():
                summary["missing_image_count"] += 1
                split_report["missing_images"] += 1
                continue
            try:
                with Image.open(source_image) as image:
                    actual_width, actual_height = image.size
                    image.verify()
            except (OSError, ValueError) as exc:
                summary["missing_image_count"] += 1
                split_report["missing_images"] += 1
                print(f"Unreadable source image skipped: {source_image} ({exc})")
                continue

            expected_width = image_record.get("width")
            expected_height = image_record.get("height")
            if (actual_width, actual_height) != (expected_width, expected_height):
                raise ValueError(
                    f"COCO dimensions disagree with image {source_image}: "
                    f"JSON={expected_width}x{expected_height}, file={actual_width}x{actual_height}"
                )

            yolo_rows = []
            for annotation in annotations_by_image.get(image_id, []):
                category_name = normalized_category(categories[annotation["category_id"]])
                class_id = TARGET_CLASS_IDS[category_name]
                bbox = annotation.get("bbox")
                valid = (
                    isinstance(bbox, list)
                    and len(bbox) == 4
                    and all(isinstance(value, (int, float)) and math.isfinite(value) for value in bbox)
                )
                if valid:
                    x, y, width, height = map(float, bbox)
                    valid = (
                        width > 0
                        and height > 0
                        and x >= 0
                        and y >= 0
                        and x + width <= actual_width
                        and y + height <= actual_height
                        and 0 <= class_id < len(CLASS_NAMES)
                    )
                if not valid:
                    summary["invalid_annotation_count"] += 1
                    summary["invalid_retained_annotation_count"] += 1
                    split_report["invalid_annotations"] += 1
                    continue
                x_center = (x + width / 2) / actual_width
                y_center = (y + height / 2) / actual_height
                normalized_width = width / actual_width
                normalized_height = height / actual_height
                if not all(
                    math.isfinite(value) and 0 <= value <= 1
                    for value in (x_center, y_center, normalized_width, normalized_height)
                ):
                    summary["invalid_annotation_count"] += 1
                    summary["invalid_retained_annotation_count"] += 1
                    split_report["invalid_annotations"] += 1
                    continue
                yolo_rows.append(
                    f"{class_id} {x_center:.8f} {y_center:.8f} "
                    f"{normalized_width:.8f} {normalized_height:.8f}"
                )
                summary["converted_category_counts"][category_name] += 1
                split_report["retained_annotation_counts"][category_name] += 1

            if not yolo_rows:
                summary["skipped_image_count"] += 1
                split_report["skipped_images"] += 1
                continue

            destination_name = f"baidu_{Path(image_record['file_name']).name}"
            destination_image = images_dir / destination_name
            destination_label = labels_dir / f"{Path(destination_name).stem}.txt"
            label_contents = "\n".join(yolo_rows) + "\n"
            if destination_image.exists() or destination_label.exists():
                if not (
                    destination_image.is_file()
                    and destination_label.is_file()
                    and filecmp.cmp(source_image, destination_image, shallow=False)
                    and destination_label.read_text(encoding="utf-8") == label_contents
                ):
                    raise FileExistsError(f"Refusing to overwrite converted file: {destination_name}")
            else:
                shutil.copy2(source_image, destination_image)
                destination_label.write_text(label_contents, encoding="utf-8")
            records.append(
                {
                    "filename": destination_name,
                    "source_dataset": "Baidu road-obstacle archive",
                    "original_filename": image_record["file_name"],
                    "source_split": split_name,
                }
            )
            summary["converted_image_count"] += 1
            split_report["converted_images"] += 1

        summary["split_summaries"][json_path.name] = split_report

    summary["category_names_found"] = category_names
    summary["records"] = records
    return summary


def report_text(summary: dict, source_root: Path, output_root: Path) -> str:
    lines = [
        "# Sprint 4 — Baidu COCO Conversion Report",
        "",
        "## Source",
        "",
        f"- Source directory: `{source_root}`",
        f"- Converted output: `{output_root}`",
        f"- Source image records: {summary['source_image_count']}",
        f"- Source annotation records: {summary['source_annotation_count']}",
        f"- Source category annotation counts: {dict(summary['category_counts'])}",
        f"- Source category names (exact as found): {dict(summary['category_names_found'])}",
        "",
        "## Conversion",
        "",
        "- Retained source classes: exact category names `pothole` and `trafficcone` (also accepts `traffic cone` if present).",
        "- Mapping: `pothole` → 13; `trafficcone` / `traffic cone` → 14.",
        "- COCO `person` and `car` annotations were ignored.",
        f"- Source pothole annotations: {summary['retained_category_counts'].get('pothole', 0)}",
        f"- Source traffic-cone annotations: {summary['retained_category_counts'].get('trafficcone', 0) + summary['retained_category_counts'].get('traffic cone', 0)}",
        f"- Valid retained pothole annotations converted: {summary['converted_category_counts'].get('pothole', 0)}",
        f"- Valid retained traffic-cone annotations converted: {summary['converted_category_counts'].get('trafficcone', 0) + summary['converted_category_counts'].get('traffic cone', 0)}",
        f"- Converted images with at least one valid retained annotation: {summary['converted_image_count']}",
        f"- Skipped images with no valid retained annotation: {summary['skipped_image_count']}",
        f"- Missing or unreadable source images: {summary['missing_image_count']}",
        f"- Invalid target annotations skipped: {summary['invalid_retained_annotation_count']}",
        "",
        "## Per-file source summary",
        "",
    ]
    for name, item in summary["split_summaries"].items():
        lines.extend(
            [
                f"### `{name}`",
                "",
                f"- Image records: {item['images']}",
                f"- Annotation records: {item['annotations']}",
                f"- Categories: {item['categories']}",
                f"- Valid retained annotations converted: {dict(item['retained_annotation_counts'])}",
                f"- Invalid annotations skipped: {item['invalid_annotations']}",
                f"- Converted images: {item['converted_images']}",
                f"- Skipped images: {item['skipped_images']}",
                f"- Missing or unreadable images: {item['missing_images']}",
                "",
            ]
        )
    lines.extend(
        [
            "Only copied images and generated YOLO labels were written. The original archive and extracted COCO data were not modified.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=project_root / "data" / "raw" / "baidu_road_obstacle",
    )
    parser.add_argument("--output-root", type=Path, default=project_root / "data" / "merged")
    parser.add_argument(
        "--report",
        type=Path,
        default=project_root / "results" / "sprint4_baidu_conversion_report.md",
    )
    args = parser.parse_args()
    summary = convert_coco_dataset(args.source_root, args.output_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_text(summary, args.source_root, args.output_root), encoding="utf-8")
    print(report_text(summary, args.source_root, args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

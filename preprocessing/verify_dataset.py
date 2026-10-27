"""Verify RSUD20K structure, YOLO labels, and class metadata incrementally."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
ANNOTATION_EXTENSIONS = {".txt", ".json", ".xml"}
METADATA_NAMES = {
    "classes.txt",
    "classes.names",
    "data.yaml",
    "dataset.yaml",
    "readme",
    "readme.md",
    "readme.txt",
}
SPLITS = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
}
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
]


def all_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def annotation_files(root: Path) -> list[Path]:
    return [
        path
        for path in all_files(root)
        if path.suffix.lower() in ANNOTATION_EXTENSIONS
        and path.name.lower() not in METADATA_NAMES
    ]


def image_files(root: Path) -> list[Path]:
    return [
        path for path in all_files(root) if path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def split_for(path: Path, root: Path) -> str:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    for part in parts:
        split = SPLITS.get(part.casefold())
        if split:
            return split
    return "unspecified"


def relative_pair_key(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root).parts)
    for index, part in enumerate(parts):
        if part.casefold() in {"images", "labels", "annotations"}:
            parts = parts[index + 1 :]
            break
    return str(Path(*parts).with_suffix("")).casefold()


def find_dataset_root(search_root: Path) -> Path | None:
    if not search_root.exists():
        return None

    candidates: set[Path] = set()
    for path in all_files(search_root):
        if path.name.casefold() in {"classes.txt", "classes.names"}:
            candidates.add(path.parent)
    for path in search_root.rglob("*"):
        if path.is_dir() and path.name.casefold() in {"images", "labels"}:
            candidates.add(path.parent)

    ranked: list[tuple[int, Path]] = []
    for candidate in candidates:
        imgs = image_files(candidate)
        anns = annotation_files(candidate)
        if not imgs:
            continue
        score = 2 * bool(
            any(p.name.casefold() in {"classes.txt", "classes.names"} for p in all_files(candidate))
        )
        score += bool(anns)
        score += 2 * ("rsud" in str(candidate).casefold())
        ranked.append((score, candidate))

    if ranked:
        return max(ranked, key=lambda item: (item[0], len(image_files(item[1]))))[1]

    # A user may point directly at a dataset root without classes.txt or named folders.
    return search_root if image_files(search_root) else None


def class_names(root: Path) -> tuple[list[str] | None, list[Path]]:
    metadata = [
        path
        for path in all_files(root)
        if path.name.casefold()
        in {"classes.txt", "classes.names", "data.yaml", "dataset.yaml"}
    ]
    names: list[str] | None = None
    class_file = next(
        (p for p in metadata if p.name.casefold() in {"classes.txt", "classes.names"}),
        None,
    )
    if class_file:
        names = [
            line.strip()
            for line in class_file.read_text(encoding="utf-8-sig", errors="replace").splitlines()
            if line.strip()
        ]

    for path in metadata:
        if path.suffix.casefold() not in {".yaml", ".yml"}:
            continue
        try:
            import yaml

            config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            yaml_names = config.get("names") if isinstance(config, dict) else None
            if isinstance(yaml_names, list):
                names = [str(value) for value in yaml_names]
            elif isinstance(yaml_names, dict):
                names = [
                    str(yaml_names[key])
                    for key in sorted(yaml_names, key=lambda value: int(value))
                ]
        except (OSError, ValueError, TypeError, ImportError) as exc:
            print(f"Metadata parse warning for {path}: {exc}")
    return names, metadata


def inspect_yolo_file(path: Path) -> dict:
    result = {
        "objects": 0,
        "empty": False,
        "format_errors": Counter(),
        "coordinate_issues": Counter(),
        "class_counts": Counter(),
        "class_ids": [],
        "normalized_lines": 0,
        "out_of_range_coordinate_lines": 0,
    }
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if not text.strip():
        result["empty"] = True
        return result

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = re.split(r"\s+", line)
        if len(fields) != 5:
            result["format_errors"]["field_count"] += 1
            continue
        try:
            class_id = int(fields[0])
        except ValueError:
            result["format_errors"]["class_id_not_integer"] += 1
            continue
        if class_id < 0:
            result["format_errors"]["negative_class_id"] += 1
        try:
            coords = [float(value) for value in fields[1:]]
        except ValueError:
            result["format_errors"]["coordinate_not_numeric"] += 1
            continue
        if not all(math.isfinite(value) for value in coords):
            result["format_errors"]["non_finite_coordinate"] += 1
            continue
        x_center, y_center, width, height = coords
        if width <= 0 or height <= 0:
            result["coordinate_issues"]["non_positive_width_or_height"] += 1
        if all(0 <= value <= 1 for value in coords):
            result["normalized_lines"] += 1
            if x_center - width / 2 < 0 or x_center + width / 2 > 1:
                result["coordinate_issues"]["box_exceeds_horizontal_bounds"] += 1
            if y_center - height / 2 < 0 or y_center + height / 2 > 1:
                result["coordinate_issues"]["box_exceeds_vertical_bounds"] += 1
        else:
            result["out_of_range_coordinate_lines"] += 1
            result["coordinate_issues"]["coordinate_outside_0_1"] += 1
        result["objects"] += 1
        result["class_counts"][class_id] += 1
        result["class_ids"].append(class_id)
    return result


def inspect_json_xml(path: Path) -> str | None:
    try:
        if path.suffix.casefold() == ".json":
            json.loads(path.read_text(encoding="utf-8-sig"))
        else:
            ET.parse(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ET.ParseError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def describe_structure(root: Path) -> list[str]:
    lines = []
    for path in sorted(root.rglob("*")):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if len(relative.parts) > 4:
            continue
        if path.is_dir():
            lines.append(f"{relative}\\")
        elif path.name.casefold() in METADATA_NAMES or path.suffix.casefold() in {".yaml", ".yml"}:
            lines.append(str(relative))
    return lines[:200]


def verify(root: Path) -> dict:
    images = image_files(root)
    labels = annotation_files(root)
    names, metadata = class_names(root)
    image_keys: dict[str, list[Path]] = defaultdict(list)
    label_keys: dict[str, list[Path]] = defaultdict(list)
    image_split_counts = Counter()
    label_split_counts = Counter()
    for image in images:
        image_keys[relative_pair_key(image, root)].append(image)
        image_split_counts[split_for(image, root)] += 1
    for label in labels:
        label_keys[relative_pair_key(label, root)].append(label)
        label_split_counts[split_for(label, root)] += 1

    missing_images = [path for key, paths in image_keys.items() if key not in label_keys for path in paths]
    orphan_labels = [path for key, paths in label_keys.items() if key not in image_keys for path in paths]
    yolo_labels = [path for path in labels if path.suffix.casefold() == ".txt"]
    per_class = Counter()
    format_errors = Counter()
    coordinate_issues = Counter()
    coordinate_line_stats = Counter()
    empty_labels: list[Path] = []
    malformed_files: list[tuple[Path, Counter]] = []
    invalid_coordinate_files: list[tuple[Path, Counter]] = []
    split_examples: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    class_ids: list[int] = []

    for label in yolo_labels:
        try:
            inspected = inspect_yolo_file(label)
        except OSError as exc:
            malformed_files.append((label, Counter({f"read_error: {exc}": 1})))
            continue
        if inspected["empty"]:
            empty_labels.append(label)
        per_class.update(inspected["class_counts"])
        class_ids.extend(inspected["class_ids"])
        format_errors.update(inspected["format_errors"])
        coordinate_issues.update(inspected["coordinate_issues"])
        coordinate_line_stats["normalized"] += inspected["normalized_lines"]
        coordinate_line_stats["outside_0_1"] += inspected["out_of_range_coordinate_lines"]
        split = split_for(label, root)
        if len(split_examples[split]) < 3:
            split_examples[split].append((label, inspected))
        if inspected["format_errors"]:
            malformed_files.append((label, inspected["format_errors"]))
        if inspected["coordinate_issues"]:
            invalid_coordinate_files.append((label, inspected["coordinate_issues"]))

    invalid_class_ids = Counter(
        {
            class_id: count
            for class_id, count in per_class.items()
            if class_id < 0 or (names is not None and class_id >= len(names))
        }
    )

    structured_errors: list[tuple[Path, str]] = []
    for path in labels:
        if path.suffix.casefold() in {".json", ".xml"}:
            error = inspect_json_xml(path)
            if error:
                structured_errors.append((path, error))

    return {
        "root": root,
        "images": images,
        "labels": labels,
        "metadata": metadata,
        "class_names": names,
        "image_split_counts": image_split_counts,
        "label_split_counts": label_split_counts,
        "missing_images": missing_images,
        "orphan_labels": orphan_labels,
        "yolo_labels": yolo_labels,
        "empty_labels": empty_labels,
        "malformed_files": malformed_files,
        "invalid_coordinate_files": invalid_coordinate_files,
        "format_errors": format_errors,
        "coordinate_issues": coordinate_issues,
        "coordinate_line_stats": coordinate_line_stats,
        "invalid_class_ids": invalid_class_ids,
        "structured_errors": structured_errors,
        "per_class": per_class,
        "class_ids": class_ids,
        "split_examples": split_examples,
        "structure": describe_structure(root),
    }


def print_report(result: dict) -> None:
    root = result["root"]
    print(f"Dataset root: {root}")
    print(f"Images: {len(result['images'])}")
    print(f"Annotation files: {len(result['labels'])}")
    print(f"Annotation formats: {dict(Counter(p.suffix.lower() for p in result['labels']))}")
    print(f"YOLO TXT labels: {len(result['yolo_labels'])}")
    print("Split counts:")
    for split in ("train", "validation", "test", "unspecified"):
        print(
            f"  {split}: images={result['image_split_counts'][split]}, "
            f"labels={result['label_split_counts'][split]}"
        )
    print(
        f"Image/label correspondence: matched image files={len(result['images']) - len(result['missing_images'])}; "
        f"images without labels={len(result['missing_images'])}; orphan labels={len(result['orphan_labels'])}"
    )
    print(f"Empty YOLO labels: {len(result['empty_labels'])}")
    print(f"Malformed YOLO files (field/type/read errors): {len(result['malformed_files'])}")
    print(f"Malformed YOLO field/type counts: {dict(result['format_errors'])}")
    print(f"Invalid-coordinate files: {len(result['invalid_coordinate_files'])}")
    print(f"Coordinate issue counts: {dict(result['coordinate_issues'])}")
    print(
        "YOLO coordinate normalization: "
        f"{result['coordinate_line_stats']['normalized']} boxes with all four coordinates in [0, 1]; "
        f"{result['coordinate_line_stats']['outside_0_1']} boxes outside [0, 1]"
    )
    print(f"Class IDs outside the class list: {dict(result['invalid_class_ids'])}")
    print(f"Malformed JSON/XML files: {len(result['structured_errors'])}")
    print(f"YOLO object annotations counted: {sum(result['per_class'].values())}")
    class_ids = result["class_ids"]
    print(f"Class ID range: {min(class_ids) if class_ids else 'none'} .. {max(class_ids) if class_ids else 'none'}")
    print(f"Per-class annotation counts: {dict(sorted(result['per_class'].items()))}")
    if result["class_names"] is not None:
        print(f"Class names ({len(result['class_names'])}), in dataset order:")
        for class_id, name in enumerate(result["class_names"]):
            match = class_id < len(EXPECTED_CLASSES) and name == EXPECTED_CLASSES[class_id]
            print(f"  {class_id}: {name} [{ 'MATCH' if match else 'DIFFERS' }]")
        if len(result["class_names"]) != len(EXPECTED_CLASSES):
            print(f"Expected class count: {len(EXPECTED_CLASSES)}")
    else:
        print("Class names: no classes.txt/classes.names or YAML names metadata found")
    print("Expected/actual class comparison:")
    actual_names = result["class_names"] or []
    for class_id, expected in enumerate(EXPECTED_CLASSES):
        actual = actual_names[class_id] if class_id < len(actual_names) else "<missing>"
        print(f"  {class_id}: expected={expected!r}; actual={actual!r}; match={actual == expected}")
    print("Representative YOLO label files (up to 3 per split):")
    for split, examples in sorted(result["split_examples"].items()):
        for path, inspected in examples:
            print(
                f"  {split}: {path} (objects={inspected['objects']}, "
                f"class_ids={sorted(inspected['class_counts'])}, "
                f"format_errors={dict(inspected['format_errors'])}, "
                f"coordinate_issues={dict(inspected['coordinate_issues'])})"
            )
    print("Dataset structure (up to depth 4):")
    for line in result["structure"]:
        print(f"  {line}")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-root",
        type=Path,
        default=project_root / "data" / "raw",
        help="Folder to search for the extracted dataset (defaults to data/raw).",
    )
    args = parser.parse_args()
    search_root = args.search_root.resolve()
    print(f"Search root: {search_root}")
    root = find_dataset_root(search_root)
    if root is None:
        print("No extracted image dataset found. No annotation or class counts are available.")
        return 2
    print_report(verify(root))
    return 0


if __name__ == "__main__":
    sys.exit(main())

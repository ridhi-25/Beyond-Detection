"""Build the 15-class merged dataset and deterministic train/val/test lists."""

from __future__ import annotations

import csv
import filecmp
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.convert_baidu_to_yolo import (  # noqa: E402
    CLASS_NAMES,
    convert_coco_dataset,
    report_text as conversion_report_text,
)

RSUD_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "rsud20k_download"
    / "verified_download"
    / "extracted"
    / "rsud20k"
)
BAIDU_ROOT = PROJECT_ROOT / "data" / "raw" / "baidu_road_obstacle"
MERGED_ROOT = PROJECT_ROOT / "data" / "merged"
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"
SEED = 42
BAIDU_VALIDATION_FRACTION = 0.10


def ensure_fresh_outputs() -> None:
    expected_classes = "\n".join(CLASS_NAMES) + "\n"
    classes_file = MERGED_ROOT / "classes.txt"
    if not classes_file.is_file() or classes_file.read_text(encoding="utf-8") != expected_classes:
        raise FileNotFoundError(f"Unified class definition is missing or differs: {classes_file}")
    for path in (
        MERGED_ROOT / "images",
        MERGED_ROOT / "labels",
        MERGED_ROOT / "manifest.csv",
        SPLIT_ROOT / "train.txt",
        SPLIT_ROOT / "val.txt",
        SPLIT_ROOT / "test.txt",
    ):
        if path.exists():
            if path in {MERGED_ROOT / "images", MERGED_ROOT / "labels"} and path.is_dir():
                continue
            if path in {
                MERGED_ROOT / "manifest.csv",
                SPLIT_ROOT / "train.txt",
                SPLIT_ROOT / "val.txt",
                SPLIT_ROOT / "test.txt",
            } and path.is_file():
                continue
            if path.is_dir() and any(path.iterdir()):
                raise FileExistsError(f"Refusing to overwrite non-empty output: {path}")
            if path.is_file():
                raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    if not RSUD_ROOT.is_dir():
        raise FileNotFoundError(f"Verified RSUD20K root not found: {RSUD_ROOT}")
    if not BAIDU_ROOT.is_dir():
        raise FileNotFoundError(f"Extracted Baidu archive root not found: {BAIDU_ROOT}")


def copy_rsud20k() -> list[dict[str, str]]:
    images_dir = MERGED_ROOT / "images"
    labels_dir = MERGED_ROOT / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source_split in ("train", "val", "test"):
        source_images = RSUD_ROOT / "images" / source_split
        source_labels = RSUD_ROOT / "labels" / source_split
        if not source_images.is_dir() or not source_labels.is_dir():
            raise FileNotFoundError(f"Missing RSUD20K image/label split: {source_split}")
        for source_image in sorted(source_images.iterdir()):
            if not source_image.is_file() or source_image.suffix.casefold() not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
            }:
                continue
            source_label = source_labels / f"{source_image.stem}.txt"
            if not source_label.is_file():
                raise FileNotFoundError(f"RSUD20K image has no label: {source_image}")
            destination_name = f"rsud20k_{source_split}_{source_image.name}"
            destination_image = images_dir / destination_name
            destination_label = labels_dir / f"{Path(destination_name).stem}.txt"
            if destination_image.exists() or destination_label.exists():
                if not (
                    destination_image.is_file()
                    and destination_label.is_file()
                    and filecmp.cmp(source_image, destination_image, shallow=False)
                    and filecmp.cmp(source_label, destination_label, shallow=False)
                ):
                    raise FileExistsError(f"RSUD20K destination collision: {destination_name}")
            else:
                shutil.copy2(source_image, destination_image)
                shutil.copy2(source_label, destination_label)
            records.append(
                {
                    "filename": destination_name,
                    "source_dataset": "RSUD20K",
                    "original_filename": source_image.name,
                    "split": {"train": "train", "val": "val", "test": "test"}[source_split],
                    "source_split": source_split,
                }
            )
    return records


def split_baidu_records(records: list[dict]) -> None:
    train_pool = sorted(
        [
            record
            for record in records
            if record["source_dataset"] == "Baidu road-obstacle archive"
            and record["source_split"] == "train"
        ],
        key=lambda record: record["filename"].casefold(),
    )
    baidu_records = [
        record for record in records if record["source_dataset"] == "Baidu road-obstacle archive"
    ]
    test_records = [record for record in baidu_records if record["source_split"] == "eval"]
    unknown = [
        record
        for record in baidu_records
        if record["source_split"] not in {"train", "eval"}
    ]
    if unknown:
        raise ValueError(
            "Baidu annotation JSON split names must identify train/eval; "
            f"found {[record['source_split'] for record in unknown[:5]]}"
        )
    random.Random(SEED).shuffle(train_pool)
    validation_count = (
        int(len(train_pool) * BAIDU_VALIDATION_FRACTION + 0.5)
        if len(train_pool) > 1
        else 0
    )
    for record in train_pool[validation_count:]:
        record["split"] = "train"
    for record in train_pool[:validation_count]:
        record["split"] = "val"
    for record in test_records:
        record["split"] = "test"


def labels_for_split_records(records: list[dict]) -> dict[str, list[Path]]:
    result = defaultdict(list)
    for record in records:
        image_path = (MERGED_ROOT / "images" / record["filename"]).resolve()
        result[record["split"]].append(image_path)
    return result


def validate_merged(records: list[dict]) -> dict:
    filename_counts = Counter(record["filename"].casefold() for record in records)
    duplicate_filenames = [name for name, count in filename_counts.items() if count > 1]
    image_paths = set()
    label_paths = set()
    class_counts = Counter()
    malformed_labels = []
    invalid_class_ids = Counter()
    invalid_coordinates = Counter()
    boundary_overruns = Counter()
    split_image_paths = defaultdict(set)
    source_counts = Counter()
    split_source_counts = Counter()

    for record in records:
        image_path = (MERGED_ROOT / "images" / record["filename"]).resolve()
        label_path = (MERGED_ROOT / "labels" / f"{Path(record['filename']).stem}.txt").resolve()
        image_paths.add(image_path)
        label_paths.add(label_path)
        source_counts[record["source_dataset"]] += 1
        split_source_counts[(record["source_dataset"], record["split"])] += 1
        split_image_paths[record["split"]].add(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Manifest image is missing: {image_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"Manifest label is missing: {label_path}")
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8-sig", errors="replace").splitlines(),
            start=1,
        ):
            fields = line.split()
            if len(fields) != 5:
                malformed_labels.append((str(label_path), line_number, "field_count"))
                continue
            try:
                class_id = int(fields[0])
                coords = [float(value) for value in fields[1:]]
            except ValueError:
                malformed_labels.append((str(label_path), line_number, "nonnumeric"))
                continue
            if not 0 <= class_id < len(CLASS_NAMES):
                invalid_class_ids[class_id] += 1
            class_counts[class_id] += 1
            if not all(value == value and abs(value) != float("inf") for value in coords):
                invalid_coordinates["nonfinite"] += 1
                continue
            if any(not 0 <= value <= 1 for value in coords):
                invalid_coordinates["outside_0_1"] += 1
                continue
            x_center, y_center, width, height = coords
            if width <= 0 or height <= 0:
                invalid_coordinates["nonpositive_width_or_height"] += 1
            if x_center - width / 2 < 0 or x_center + width / 2 > 1:
                boundary_overruns["horizontal"] += 1
            if y_center - height / 2 < 0 or y_center + height / 2 > 1:
                boundary_overruns["vertical"] += 1

    actual_images = {path.resolve() for path in (MERGED_ROOT / "images").iterdir() if path.is_file()}
    actual_labels = {path.resolve() for path in (MERGED_ROOT / "labels").iterdir() if path.is_file()}
    missing_images = len(image_paths - actual_images)
    orphan_images = len(actual_images - image_paths)
    orphan_labels = len(actual_labels - label_paths)
    split_overlaps = []
    split_names = ("train", "val", "test")
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = split_image_paths[left] & split_image_paths[right]
            if overlap:
                split_overlaps.append((left, right, len(overlap)))

    if duplicate_filenames or missing_images or orphan_images or orphan_labels:
        raise ValueError(
            f"File correspondence failures: duplicate_names={len(duplicate_filenames)}, "
            f"missing_images={missing_images}, unmanifested_images={orphan_images}, "
            f"orphan_labels={orphan_labels}"
        )
    if malformed_labels or invalid_class_ids or invalid_coordinates or split_overlaps:
        raise ValueError(
            f"Dataset validation failed: malformed={len(malformed_labels)}, "
            f"invalid_class_ids={dict(invalid_class_ids)}, "
            f"invalid_coordinates={dict(invalid_coordinates)}, overlaps={split_overlaps}"
        )

    samples_checked = []
    for split in split_names:
        candidates = sorted(split_image_paths[split])
        if candidates:
            sample = candidates[0]
            with Image.open(sample) as image:
                image.verify()
            samples_checked.append(str(sample))

    return {
        "image_count": len(image_paths),
        "label_file_count": len(label_paths),
        "annotation_count": sum(class_counts.values()),
        "class_counts": class_counts,
        "malformed_labels": malformed_labels,
        "invalid_class_ids": invalid_class_ids,
        "invalid_coordinates": invalid_coordinates,
        "boundary_overruns": boundary_overruns,
        "duplicate_filenames": duplicate_filenames,
        "missing_images": missing_images,
        "orphan_images": orphan_images,
        "orphan_labels": orphan_labels,
        "split_overlaps": split_overlaps,
        "source_counts": source_counts,
        "split_source_counts": split_source_counts,
        "split_counts": Counter(record["split"] for record in records),
        "samples_checked": samples_checked,
    }


def write_manifest_and_splits(records: list[dict]) -> None:
    MERGED_ROOT.mkdir(parents=True, exist_ok=True)
    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = MERGED_ROOT / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("filename", "source_dataset", "original_filename", "split"),
        )
        writer.writeheader()
        for record in sorted(records, key=lambda row: (row["split"], row["filename"].casefold())):
            writer.writerow({key: record[key] for key in writer.fieldnames})

    paths_by_split = labels_for_split_records(records)
    for split, filename in (("train", "train.txt"), ("val", "val.txt"), ("test", "test.txt")):
        contents = "".join(f"{path.as_posix()}\n" for path in sorted(paths_by_split[split]))
        (SPLIT_ROOT / filename).write_text(contents, encoding="utf-8")


def dataset_report(summary: dict, conversion: dict, records: list[dict]) -> str:
    lines = [
        "# Sprint 4 — Merged Dataset Report",
        "",
        "## Dataset counts",
        "",
        f"- RSUD20K images/labels copied: {summary['source_counts']['RSUD20K']:,}",
        f"- Valid Baidu images/labels converted: {summary['source_counts']['Baidu road-obstacle archive']:,}",
        f"- Merged images: {summary['image_count']:,}",
        f"- Merged label files: {summary['label_file_count']:,}",
        f"- Merged annotation rows: {summary['annotation_count']:,}",
        "",
        "## Split counts",
        "",
        "| Split | Total images | RSUD20K | Baidu |",
        "|---|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        label = {"val": "Validation"}.get(split, split.title())
        lines.append(
            f"| {label} | {summary['split_counts'][split]:,} | "
            f"{summary['split_source_counts'][('RSUD20K', split)]:,} | "
            f"{summary['split_source_counts'][('Baidu road-obstacle archive', split)]:,} |"
        )
    lines.extend(
        [
            "",
            f"- Baidu train-source assignment: deterministic shuffle (seed {SEED}); "
            f"{1-BAIDU_VALIDATION_FRACTION:.0%} train / {BAIDU_VALIDATION_FRACTION:.0%} validation.",
            "- Baidu eval-source images are assigned to test; RSUD20K's original split allocation is preserved.",
            "",
            "## Class distribution (annotation rows)",
            "",
            "| ID | Class | Rows |",
            "|---:|---|---:|",
        ]
    )
    for class_id, class_name in enumerate(CLASS_NAMES):
        lines.append(f"| {class_id} | {class_name} | {summary['class_counts'][class_id]:,} |")
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Missing image files: {summary['missing_images']}",
            f"- Images without manifest entries: {summary['orphan_images']}",
            f"- Orphan label files: {summary['orphan_labels']}",
            f"- Duplicate output filenames: {len(summary['duplicate_filenames'])}",
            f"- Malformed label rows/files: {len(summary['malformed_labels'])}",
            f"- Invalid class IDs: {dict(summary['invalid_class_ids'])}",
            f"- Invalid YOLO coordinates: {dict(summary['invalid_coordinates'])}",
            f"- Source boundary-overrun incidences retained unchanged: {dict(summary['boundary_overruns'])}",
            f"- Split overlap: {summary['split_overlaps']}",
            f"- Split list sample images opened successfully: {summary['samples_checked']}",
            "- Annotation IDs 0–12 from RSUD20K were preserved; source RSUD20K images and labels remain unchanged.",
            "- Only Baidu pothole and traffic-cone annotations were converted; source person/car categories were ignored.",
            "- Invalid Baidu boxes were skipped, not clamped or written.",
            "",
            "## Baidu conversion detail",
            "",
            f"- Source image records: {conversion['source_image_count']:,}",
            f"- Source annotation records: {conversion['source_annotation_count']:,}",
            f"- Converted target annotations: {dict(conversion['converted_category_counts'])}",
            f"- Invalid Baidu annotations skipped: {conversion['invalid_annotation_count']}",
            f"- Converted Baidu images: {conversion['converted_image_count']}",
            f"- Skipped Baidu images: {conversion['skipped_image_count']}",
            f"- Missing/unreadable Baidu images: {conversion['missing_image_count']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    ensure_fresh_outputs()
    records = copy_rsud20k()
    conversion = convert_coco_dataset(BAIDU_ROOT, MERGED_ROOT)
    records.extend(conversion["records"])
    split_baidu_records(records)
    write_manifest_and_splits(records)
    summary = validate_merged(records)

    conversion_report = PROJECT_ROOT / "results" / "sprint4_baidu_conversion_report.md"
    conversion_report.parent.mkdir(parents=True, exist_ok=True)
    conversion_report.write_text(
        conversion_report_text(conversion, BAIDU_ROOT, MERGED_ROOT),
        encoding="utf-8",
    )
    report_path = PROJECT_ROOT / "results" / "sprint4_dataset_report.md"
    report_path.write_text(dataset_report(summary, conversion, records), encoding="utf-8")
    print(dataset_report(summary, conversion, records))
    print(f"Manifest: {MERGED_ROOT / 'manifest.csv'}")
    print(f"Split lists: {SPLIT_ROOT / 'train.txt'}, {SPLIT_ROOT / 'val.txt'}, {SPLIT_ROOT / 'test.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

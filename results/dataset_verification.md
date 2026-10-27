# Sprint 2 — Dataset Verification

## Environment

- Python: 3.12.3 (project environment: `C:\beyond-detection\.venv`)
- Project path: `C:\beyond-detection`
- Free space before download: approximately 266.7 GiB
- Free space after download and extraction: 245.12 GiB (263,196,995,584 bytes)
- Download date: 2026-09-27 (Kaggle CLI; exact completion time was not recorded)
- Kaggle CLI package: 2.2.4 (`requirements.txt` already records `kaggle==2.2.4`)

## RSUD20K Source

- Kaggle identifier: `hasibzunair/rsud20k-bangladesh-road-scene-understanding`
- Download status: Downloaded successfully using the authenticated project-venv Kaggle CLI.
- Kaggle-reported download size: approximately 10.2 GiB.
- Downloaded archive:
  `data/raw/rsud20k_download/verified_download/rsud20k-bangladesh-road-scene-understanding.zip`
- Archive size: 10,970,943,623 bytes (10.97 GB / 10.22 GiB).
- Extracted location:
  `data/raw/rsud20k_download/verified_download/extracted/rsud20k`
- Extracted size: 10,987,087,790 bytes (10.99 GB / 10.23 GiB), 40,669 files.
- The archive was checked for ZIP readability and unsafe paths before extraction. It contained 40,669 entries, had no unsafe paths, and all entries were extracted by streaming without loading the dataset into memory. The original downloaded ZIP was retained.
- A pre-existing 5,532,286,976-byte archive at
  `data/raw/rsud20k_download/rsud20k-bangladesh-road-scene-understanding.zip`
  was not a readable ZIP. It and its 74-byte `.kaggle-partial` sidecar were preserved unchanged; the valid download was kept separately.

## Dataset Structure

Actual dataset root:
`data/raw/rsud20k_download/verified_download/extracted/rsud20k`

```text
rsud20k/
├── classes.txt
├── images/
│   ├── test/
│   ├── train/
│   └── val/
└── labels/
    ├── test/
    ├── train/
    └── val/
```

- Image files: 20,334 `.jpg`
- Label files: 20,334 `.txt`
- Other dataset metadata found: `classes.txt`
- No YAML, JSON, XML, or README metadata files were present in the extracted dataset.
- Dataset files are split into `train`, `val` (reported below as validation), and `test`.

## Dataset Counts

| Split | Images | Label files |
|---|---:|---:|
| Train | 18,681 | 18,681 |
| Validation (`val`) | 1,004 | 1,004 |
| Test | 649 | 649 |
| **Total** | **20,334** | **20,334** |

- YOLO object annotations (non-empty label rows): 130,000
- Image/label matching: all 20,334 images matched a label by split-relative path and stem.
- Images without a matching label: 0
- Orphan labels: 0
- Empty YOLO label files: 0

## Annotation Format and Validation

- Annotation format: YOLO object-detection text labels.
- All 20,334 label files were parsed (not only sampled); representative files from train, validation, and test were also printed by the verifier.
- Every parsed annotation row had exactly five fields in the form `class_id x_center y_center width height`.
- Class IDs were integers, non-negative, and within the discovered class list: observed range 0–12; no out-of-list IDs.
- Coordinates were numeric and finite. All 130,000 rows had all four values within `[0, 1]`; all widths and heights were positive.
- Malformed YOLO files (field/type/read errors): 0.
- Malformed JSON/XML annotation files: 0 (no JSON/XML annotations were present).
- Geometric boundary warnings: 3,399 label files contain at least one box extending beyond the normalized image boundary. The verifier counted 2,639 horizontal and 1,144 vertical boundary-overrun incidences; a file/box may contribute to both. These are coordinate-geometry warnings, not malformed five-field YOLO rows. No annotations were changed.

## Class Verification

The actual `classes.txt` contains the following 13 entries, in this order. Every entry matches the methodology's expected name and class ID exactly.

| Class ID | Expected name | Actual `classes.txt` name | Match | Annotation rows |
|---:|---|---|---|---:|
| 0 | person | person | Yes | 35,645 |
| 1 | rickshaw | rickshaw | Yes | 25,410 |
| 2 | rickshaw van | rickshaw van | Yes | 3,050 |
| 3 | auto rickshaw | auto rickshaw | Yes | 13,327 |
| 4 | truck | truck | Yes | 1,600 |
| 5 | pickup truck | pickup truck | Yes | 1,174 |
| 6 | private car | private car | Yes | 20,080 |
| 7 | motorcycle | motorcycle | Yes | 16,170 |
| 8 | bicycle | bicycle | Yes | 2,983 |
| 9 | bus | bus | Yes | 6,029 |
| 10 | micro bus | micro bus | Yes | 3,323 |
| 11 | covered van | covered van | Yes | 692 |
| 12 | human hauler | human hauler | Yes | 517 |
| **Total** |  |  |  | **130,000** |

## Wang/Baidu Dataset

The original Wang et al. road-obstacle dataset / Baidu AI Studio dataset ID 198589 was not found locally in the project. No substitute dataset was downloaded. Any substitute selection remains pending explicit user approval.

## Decision Required

Before later preprocessing changes any annotations, decide how to handle boxes that extend outside image boundaries. The source labels are intact and were not corrected. If the Wang/Baidu dataset is still required, its availability or an explicitly approved substitute must also be resolved separately.

## Conclusion

Sprint 2 verification is complete for the downloaded RSUD20K source: its structure, split counts, label correspondence, YOLO syntax, class ordering, and annotation counts were inspected and recorded. The source is available for later work, with the boundary-overrun finding above requiring an explicit handling decision before modifying labels. Dataset merging and training were not performed.

## Files Created or Modified

- Updated: `preprocessing/verify_dataset.py` — searches under `data/raw`, reports discovered structure and split counts, validates YOLO labels incrementally, checks class metadata and image/label correspondence, and reports coordinate warnings without modifying the source.
- Updated: `results/dataset_verification.md` — this report.
- Updated: `README.md` — Sprint 2 summary appended; Sprint 1 setup details retained.
- `requirements.txt` already contained the verified `kaggle==2.2.4` entry and was not changed unnecessarily.

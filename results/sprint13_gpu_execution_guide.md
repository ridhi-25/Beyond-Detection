# Sprint 13 — Kaggle GPU Execution Guide

## Purpose and current status

This guide prepares the already implemented Sprint 13 E1 experiment for a Kaggle GPU notebook. It does not claim that the full architecture check or pretrained transfer has passed. No training or FWB evaluation has run.

The local blocker was resource availability: the last architecture gate stopped **before model construction** with 0.97 GiB RAM available against a configured 3.00 GiB guard (the process also reported current available RAM of 1.02 GiB). This project environment has no CUDA GPU. Do not retry the full check or start training on that local machine.

The 3 GiB free host-RAM and free-VRAM limits in `training/check_fwb_arch.py` are deliberately conservative operational guardrails. They are not measured model minimums, and there is not enough evidence here to claim that 3 GiB is strictly required or unnecessarily restrictive. They were retained rather than bypassed. Use a Kaggle session where the script's normal guards pass; do not lower the threshold just to force a check.

## 1. Required Kaggle resources

Enable a Kaggle GPU accelerator (for example, a supported T4/P100-class GPU) and confirm CUDA is visible. The full architecture gate additionally requires at least 3 GiB available host RAM and 3 GiB free GPU memory. Training uses the prescribed 640 image size and batch 16; a gate pass alone does not guarantee sufficient memory for the full training run.

Make these project files available under `/kaggle/working/beyond-detection` by cloning or uploading the project:

- `models/detector/fwb_module.py`
- `models/detector/fwb_registration.py`
- `models/detector/yolov8n_fwb.yaml`
- `models/detector/data.yaml`
- `training/check_fwb_arch.py`
- `training/train_fwb.py`
- `evaluation/evaluate_fwb.py`
- `evaluation/compare_baselines.py`
- `evaluation/evaluate_baseline.py` and `evaluation/yolo_to_coco.py`
- `results/previous_b1/weights/best.pt`
- Sprint 5 baseline artifacts under `results/b1_baseline_eval/`, especially `overall_metrics.json`, `per_class_metrics.csv`, `test_ground_truth_coco.json`, and `prediction_validation.json`
- `data/splits/train.txt`, `data/splits/val.txt`, `data/splits/test.txt`
- `results/experiment_log.csv` and the Sprint 13 reports for provenance

Keep the B1 checkpoint at the specified project path. Verify it against the frozen SHA-256 before running any gates:

`e8fecd7d2597090b815ac408cb1d4c783f908e56b65a588559db8b6b0afd685d`

Attach the merged dataset as a Kaggle Dataset. It must provide:

- `images/`
- `labels/`
- `classes.txt`
- `manifest.csv` (if retained in the packaged dataset)

The source dataset must be the same merged 15-class dataset represented by the committed split lists and B1 test evaluation. Do not use the test split for training, do not regenerate or repartition the test images, and do not change source labels.

## 2. Runtime and dependency checks

Run in a Kaggle notebook cell before setup:

```bash
%%bash
python - <<'PY'
import sys, torch
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    print(f"Free GPU memory: {free_bytes / 1024**3:.2f} / {total_bytes / 1024**3:.2f} GiB")
PY
free -h
df -h /kaggle/working
```

Install the version required by the project-local parser adapter, without force-reinstalling or upgrading the Kaggle CUDA PyTorch build:

```bash
%%bash
python -m pip install "ultralytics==8.4.160" "pycocotools==2.0.11" psutil pyyaml
```

If pip proposes replacing PyTorch with a CPU-only build, stop and resolve the Kaggle environment instead of accepting the replacement. Verify:

```bash
%%bash
python - <<'PY'
import torch, ultralytics, yaml, psutil
import pycocotools
assert ultralytics.__version__ == "8.4.160", ultralytics.__version__
assert torch.cuda.is_available(), "Kaggle GPU is not enabled/visible"
print("Ultralytics:", ultralytics.__version__)
print("PyTorch:", torch.__version__)
print("CUDA GPU:", torch.cuda.get_device_name(0))
print("Dependencies import successfully")
PY
```

Do not change the pinned Ultralytics version without separately validating the local parser adapter against that version.

## 3. Stage dataset paths without editing the source dataset

Replace the sample input path below with the mounted Kaggle Dataset path shown by `!find /kaggle/input -maxdepth 4 -type d`. For example:

```bash
%%bash
export PROJECT=/kaggle/working/beyond-detection
export DATA_INPUT=/kaggle/input/beyond-detection-merged/data/merged
```

Verify the mounted folders and preserve the source as read-only. The project scripts and Ultralytics need a writable labels directory for cache metadata. The following stages the labels to working storage and points the project image directory at the read-only Kaggle images by symlink; it does not alter image or label contents in the input dataset:

```bash
%%bash
export PROJECT=/kaggle/working/beyond-detection
export DATA_INPUT=/kaggle/input/beyond-detection-merged/data/merged
test -d "$DATA_INPUT/images"
test -d "$DATA_INPUT/labels"
test -f "$DATA_INPUT/classes.txt"
mkdir -p "$PROJECT/data/merged" "$PROJECT/data/splits" "$PROJECT/models/detector"
ln -s "$DATA_INPUT/images" "$PROJECT/data/merged/images"
cp -a "$DATA_INPUT/labels" "$PROJECT/data/merged/labels"
cp "$DATA_INPUT/classes.txt" "$PROJECT/data/merged/classes.txt"
if test -f "$DATA_INPUT/manifest.csv"; then
  cp "$DATA_INPUT/manifest.csv" "$PROJECT/data/merged/manifest.csv"
fi
```

If the project upload already contains `data/merged/images` or `data/merged/labels`, do not overwrite or merge blindly. Start with a clean Kaggle working copy of the project code, or inspect and remove only the disposable Kaggle copy before staging.

The committed split files contain Windows absolute paths. Rebuild only the Kaggle working-copy split lists from those original lists, preserving each exact image basename and order:

```bash
%%bash
export PROJECT=/kaggle/working/beyond-detection
export DATA_INPUT=/kaggle/input/beyond-detection-merged/data/merged
cd "$PROJECT"
python - <<'PY'
import os
from pathlib import Path

project = Path(os.environ["PROJECT"]).resolve()
images = project / "data" / "merged" / "images"
source_project = project / "data" / "splits"
expected_counts = {"train": 19390, "val": 1083, "test": 750}
all_names = set()
for split in ("train", "val", "test"):
    source = source_project / f"{split}.txt"
    rows = [line.strip() for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    names = [Path(value.replace("\\", "/")).name for value in rows]
    if len(names) != expected_counts[split]:
        raise ValueError(f"{split} expected {expected_counts[split]} paths; found {len(names)}")
    if len(names) != len(set(names)) or all_names.intersection(names):
        raise ValueError(f"Duplicate image basenames found in or across {split} split")
    all_names.update(names)
    targets = [images / name for name in names]
    missing = [str(path) for path in targets if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{split}: {len(missing)} images missing; first: {missing[0]}")
    (source / f"{split}.txt").write_text(
        "".join(f"{path.as_posix()}\n" for path in targets),
        encoding="utf-8",
    )
    print(split, len(targets), "images")
PY
```

Before running that cell, retain the original split files separately if they are your only copy. The script changes only the Kaggle working copy's `data/splits/*.txt`; it does not edit the local Windows project or the mounted dataset.

Generate a Linux-path YOLO configuration in the Kaggle working project, using the mounted taxonomy in its existing order:

```bash
%%bash
export PROJECT=/kaggle/working/beyond-detection
cd "$PROJECT"
python - <<'PY'
from pathlib import Path
import yaml

root = Path("data/merged").resolve()
split_root = Path("data/splits").resolve()
classes = [line.strip() for line in (root / "classes.txt").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
expected = [
    "person", "rickshaw", "rickshaw van", "auto rickshaw", "truck",
    "pickup truck", "private car", "motorcycle", "bicycle", "bus",
    "micro bus", "covered van", "human hauler", "pothole", "traffic cone",
]
if classes != expected:
    raise ValueError(f"Unexpected 15-class taxonomy/order: {classes}")
config = {
    "path": root.as_posix(),
    "train": (split_root / "train.txt").as_posix(),
    "val": (split_root / "val.txt").as_posix(),
    "test": (split_root / "test.txt").as_posix(),
    "nc": 15,
    "names": classes,
}
Path("models/detector/data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(Path("models/detector/data.yaml").read_text(encoding="utf-8"))
PY
```

`training/train_fwb.py` independently validates all three split files, image paths, and class order. Its validation must pass before it reaches the architecture or transfer gates.

## 4. Stage and verify official pretrained YOLOv8n weights

Use the official Ultralytics v8.4.0 asset already documented for Sprint 3:

```bash
%%bash
export PROJECT=/kaggle/working/beyond-detection
cd "$PROJECT"
wget -O yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt
sha256sum yolov8n.pt
```

The expected SHA-256 recorded in this project is:

`f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`

Do not use the B1 checkpoint as the COCO initialization source. B1 remains a frozen comparison reference only.

Confirm the frozen B1 path and hash:

```bash
%%bash
export PROJECT=/kaggle/working/beyond-detection
test -f "$PROJECT/results/previous_b1/weights/best.pt"
sha256sum "$PROJECT/results/previous_b1/weights/best.pt"
```

The expected hash is `e8fecd7d2597090b815ac408cb1d4c783f908e56b65a588559db8b6b0afd685d`. Stop if either hash differs unexpectedly.

## 5. Fresh-process registration and architecture/transfer gates

First confirm project-local registration in a fresh Python process. This is a small parser test, not the full architecture gate:

```bash
%%bash
export PROJECT=/kaggle/working/beyond-detection
cd "$PROJECT"
python - <<'PY'
from models.detector.fwb_registration import register_fwb
from models.detector.fwb_module import FWB
from ultralytics.nn import tasks

register_fwb()
cfg = {
    "nc": 15,
    "scale": "n",
    "scales": {"n": [1.0, 0.25, 1024]},
    "backbone": [[-1, 1, "Conv", [1024, 1, 1]], [-1, 1, "FWB", [1024]]],
    "head": [[-1, 1, "Classify", [15]]],
}
model, _ = tasks.parse_model(cfg, ch=3, verbose=False)
assert isinstance(model[1], FWB) and model[1].c1 == 256
print("Fresh-process parser registration passed.")
PY
```

Run the complete workflow from the project root after the staged paths and weights have been verified:

```bash
%%bash
export PROJECT=/kaggle/working/beyond-detection
cd "$PROJECT"
python training/train_fwb.py \
  --data models/detector/data.yaml \
  --model models/detector/yolov8n_fwb.yaml \
  --weights yolov8n.pt \
  --epochs 50 --imgsz 640 --batch 16 --device 0 \
  --execute
```

The `--execute` command does not skip the gates. The script must, in this order:

1. Validate all dataset paths and the 15-class taxonomy.
2. Run `training/check_fwb_arch.py`'s GPU 640×640 forward checks for stock YOLOv8n and YOLOv8n+FWB; verify FWB placement and channels; exercise a batch of two; report model parameter counts/GFLOPs; and confirm the B1 hash is unchanged.
3. Load the 80-class COCO checkpoint and verify the layer-index-adjusted state transfer into compatible backbone, neck, and detector regression tensors. It must report actual transferred items/elements, ensure FWB tensors remain initialized locally, and enforce the transfer-coverage check.
4. Persist `results/fwb_optimization_full/preflight_weight_transfer.json`.
5. Only then call Ultralytics training for 50 epochs.

If either gate fails, the command must stop without training. Do not bypass guards, reduce epochs/batch, or call the trainer manually. The actual architecture/transfer results must be inspected and accepted before proceeding. If the environment requires a different dataset path, update the Kaggle working copy's `models/detector/data.yaml`; do not modify the local Windows dataset/configuration.

Expected training outputs are under:

- `results/fwb_optimization_full/weights/best.pt`
- `results/fwb_optimization_full/weights/last.pt`
- `results/fwb_optimization_full/results.csv`
- `results/fwb_optimization_full/args.yaml`
- Ultralytics plots and logs in the same run directory

Check `results/fwb_optimization_full/preflight_status.md` and the results CSV to confirm exactly 50 epoch rows before calling the run complete.

## 6. Held-out evaluation and comparison — only after 50 epochs

Do not evaluate until a valid trained best checkpoint exists and the training record confirms 50 completed epochs:

```bash
%%bash
export PROJECT=/kaggle/working/beyond-detection
cd "$PROJECT"
python evaluation/evaluate_fwb.py
python evaluation/compare_baselines.py
```

The evaluator uses the same test split and converted ground truth as Sprint 5, checks image/category/annotation equality, and applies the existing prediction-validation rule. It does not train or tune on test data. `cache=False` avoids writing a labels cache to the read-only Kaggle image source; writable label files are staged under the Kaggle working project.

Expected evaluation outputs:

- `results/fwb_optimization_full/evaluation/overall_metrics.csv`
- `results/fwb_optimization_full/evaluation/overall_metrics.json`
- `results/fwb_optimization_full/evaluation/per_class_metrics.csv`
- `results/fwb_optimization_full/evaluation/per_class_map50.csv`
- Raw `predictions.json` and the derived `predictions_cocoeval.json`
- Prediction-validation report, COCOeval summary, plots, and distribution files
- Refreshed `results/sprint13_fwb_comparison.csv`
- Refreshed `results/sprint13_fwb_per_class_comparison.csv`

The comparison script refuses incomplete FWB evaluation records and requires actual 750-image held-out test metrics. Keep the original B1 reference files intact.

## 7. Integrity and completion checklist

Before and after running the workflow, verify:

- B1 checkpoint SHA-256 remains the expected frozen hash.
- The mounted source dataset is unchanged; no labels or splits in the input dataset were written.
- Sprint 5 files under `results/b1_baseline_eval/` are unchanged.
- The FWB run is isolated under `results/fwb_optimization_full/`.
- The architecture report records measured full-model parameters/GFLOPs only after the 640×640 checks actually pass.
- The transfer report contains actual transferred tensor/item counts, source hash, class count, and FWB initialization keys.
- Training completed all 50 epochs before describing it as complete.
- Evaluation and comparison metrics were generated from the held-out test outputs, not placeholders.
- No Sprint 14 / INT8 work was started.

## Current remaining steps

1. Stage the project, unchanged B1 checkpoint, merged images/labels/classes, and original split lists in Kaggle.
2. Verify Kaggle GPU, host RAM, free VRAM, package versions, and both checkpoint hashes.
3. Create Linux-path working split lists and `models/detector/data.yaml` as documented.
4. Pass the fresh-process FWB parser check.
5. Run the combined architecture and actual pretrained-transfer gates.
6. If and only if both gates pass, run the exact 50-epoch training configuration.
7. Verify all 50 epochs, evaluate on the unchanged Sprint 5 test split, and generate the actual comparison reports.

Until these steps complete, Sprint 13 remains **prepared but blocked**, and all FWB evaluation numbers remain pending.

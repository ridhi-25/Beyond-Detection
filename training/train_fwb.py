"""Prepare or run the 50-epoch, pretrained YOLOv8n+FWB E1 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.utils.torch_utils import intersect_dicts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.detector.fwb_registration import register_fwb
from training.check_fwb_arch import EXPECTED_CLASSES, _run as run_architecture_check

DEFAULT_DATA = PROJECT_ROOT / "models" / "detector" / "data.yaml"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "detector" / "yolov8n_fwb.yaml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "yolov8n.pt"
RUN_DIR = PROJECT_ROOT / "results" / "fwb_optimization_full"
TRANSFER_REPORT = RUN_DIR / "preflight_weight_transfer.json"
MIN_TRANSFER_NUMEL_RATIO = 0.80


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _names_from_config(config: dict[str, Any]) -> list[str]:
    names = config.get("names")
    if isinstance(names, dict):
        return [names[index] if index in names else names[str(index)] for index in range(config["nc"])]
    if isinstance(names, list):
        return names
    raise ValueError("Dataset YAML must define class names as a list or indexed mapping")


def validate_dataset_config(data_path: Path) -> dict[str, Any]:
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset config missing: {data_path}")
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    names = _names_from_config(config)
    if config.get("nc") != 15 or names != EXPECTED_CLASSES:
        raise ValueError(f"Dataset must use the intended 15-class order; found nc={config.get('nc')}, names={names}")
    splits: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        value = config.get(split)
        if not value:
            raise ValueError(f"Dataset YAML does not define the {split} split")
        split_path = Path(value)
        if not split_path.is_absolute():
            root = Path(config.get("path", data_path.parent))
            if not root.is_absolute():
                root = (data_path.parent / root).resolve()
            split_path = root / split_path
        split_path = split_path.resolve()
        if not split_path.is_file():
            raise FileNotFoundError(f"{split} split list missing: {split_path}")
        image_paths = [
            Path(line.strip())
            for line in split_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        if not image_paths:
            raise ValueError(f"{split} split list is empty: {split_path}")
        missing = [str(path) for path in image_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} {split} image(s) missing; first: {missing[0]}")
        splits[split] = {"path": str(split_path), "image_count": len(image_paths)}
    return {"data_yaml": str(data_path.resolve()), "nc": 15, "names": names, "splits": splits}


def _shift_layer_index(key: str) -> str:
    parts = key.split(".", 2)
    if len(parts) == 3 and parts[0] == "model" and parts[1].isdigit():
        layer_index = int(parts[1])
        if layer_index >= 10:
            return f"model.{layer_index + 1}.{parts[2]}"
    return key


def _completed_epoch_rows(results_csv: Path) -> int:
    if not results_csv.is_file():
        return 0
    with results_csv.open(encoding="utf-8") as stream:
        return max(0, sum(1 for _ in stream) - 1)


def transfer_pretrained_weights(model: YOLO, pretrained_path: Path) -> dict[str, Any]:
    """Transfer shape-compatible pretrained tensors, shifting layers after FWB."""
    if not pretrained_path.is_file():
        raise FileNotFoundError(f"Pretrained COCO weights missing: {pretrained_path}")

    pretrained = YOLO(str(pretrained_path), task="detect")
    if len(pretrained.names) != 80 or int(pretrained.model.model[-1].nc) != 80:
        raise ValueError(
            "Pretrained initialization must be the 80-class official YOLOv8n COCO model"
        )
    source_state = pretrained.model.state_dict()
    target_state = model.model.state_dict()
    shifted_state = {_shift_layer_index(key): value for key, value in source_state.items()}
    transferred = intersect_dicts(shifted_state, target_state)
    if not transferred:
        raise RuntimeError("No compatible pretrained tensors were found for FWB architecture")

    model.model.load_state_dict(transferred, strict=False)
    loaded_state = model.model.state_dict()
    mismatches = [
        key for key, value in transferred.items()
        if not torch.equal(loaded_state[key].detach().cpu(), value.detach().cpu())
    ]
    if mismatches:
        raise RuntimeError(f"Pretrained transfer did not persist for keys: {mismatches[:8]}")

    fwb_keys = [key for key in target_state if key.startswith("model.10.")]
    if not fwb_keys or any(key in transferred for key in fwb_keys):
        raise RuntimeError("FWB parameters were unexpectedly transferred instead of initialized locally")
    detect_regression_keys = [
        key for key in target_state
        if key.startswith("model.23.cv2.") and key in transferred
    ]
    if not detect_regression_keys:
        raise RuntimeError("Compatible pretrained detector regression weights were not transferred")

    transferable_numel = sum(value.numel() for value in transferred.values())
    target_numel = sum(value.numel() for value in target_state.values())
    ratio = transferable_numel / target_numel
    if ratio < MIN_TRANSFER_NUMEL_RATIO:
        raise RuntimeError(
            f"Pretrained transfer is inadequate: {transferable_numel}/{target_numel} "
            f"target elements ({ratio:.1%}); minimum is {MIN_TRANSFER_NUMEL_RATIO:.0%}"
        )

    result = {
        "source": str(pretrained_path.resolve()),
        "source_sha256": sha256(pretrained_path),
        "transfer_method": (
            "Ultralytics intersect_dicts with source layer indices >=10 shifted by one "
            "to account for the inserted FWB at target layer 10"
        ),
        "transferred_state_items": len(transferred),
        "target_state_items": len(target_state),
        "transferred_parameter_and_buffer_numel": transferable_numel,
        "target_parameter_and_buffer_numel": target_numel,
        "transfer_numel_ratio": ratio,
        "fwb_uninitialized_keys": fwb_keys,
        "fwb_initialization": "FWB Conv2d default PyTorch initialization; no COCO counterpart",
        "detector_regression_weight_items_transferred": len(detect_regression_keys),
        "class_count": int(model.model.model[-1].nc),
        "source_model": "YOLOv8n pretrained COCO",
        "ultralytics_version": ultralytics.__version__,
    }
    del pretrained
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--execute", action="store_true", help="Train only after architecture and transfer gates pass.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.epochs, args.imgsz) != (50, 640):
        raise ValueError("E1 is fixed at 50 epochs and imgsz=640; do not silently change the protocol")
    if args.batch != 16:
        raise ValueError("E1 batch size must be 16; resolve memory on the target GPU before changing it")
    data_path, model_path, weights_path = (
        path.expanduser().resolve() for path in (args.data, args.model, args.weights)
    )
    dataset = validate_dataset_config(data_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"FWB model YAML missing: {model_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Pretrained YOLOv8n weights missing: {weights_path}. "
            "Place the official weights locally; this script does not download them."
        )

    if args.execute and not torch.cuda.is_available():
        raise RuntimeError(
            "Refusing the 50-epoch CPU training run: this experiment requires a suitable GPU."
        )
    if args.execute and args.device.lower() == "cpu":
        raise RuntimeError("Refusing the 50-epoch CPU training run; choose a supported GPU device.")

    register_fwb()
    gate_result = run_architecture_check(argparse.Namespace(min_free_ram_gb=3.0))
    if gate_result != 0:
        raise RuntimeError("Mandatory FWB architecture gate did not pass; training was not started")

    model = YOLO(str(model_path), task="detect")
    if int(model.model.model[-1].nc) != 15:
        raise ValueError(f"FWB detector must have 15 classes, found {model.model.model[-1].nc}")
    transfer = transfer_pretrained_weights(model, weights_path)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if (RUN_DIR / "weights" / "best.pt").exists() or (RUN_DIR / "weights" / "last.pt").exists():
        raise FileExistsError(f"Refusing to overwrite an existing FWB training run: {RUN_DIR}")
    preflight = {
        "status": "preflight_passed_training_not_started",
        "dataset": dataset,
        "model_yaml": str(model_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "optimizer": "SGD",
        "lr0": 0.01,
        "seed": 0,
        "deterministic": True,
        "close_mosaic": 10,
        "workers": 2,
        "patience": 100,
        "amp": True,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "training_device": args.device,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available()
            else None
        ),
        "dataset_manifest_sha256": sha256(PROJECT_ROOT / "data" / "merged" / "manifest.csv")
        if (PROJECT_ROOT / "data" / "merged" / "manifest.csv").is_file()
        else None,
        "pretrained_transfer": transfer,
    }
    TRANSFER_REPORT.write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    print(json.dumps(preflight, indent=2))
    if not args.execute:
        print("Preflight passed. Training was not requested; no model training occurred.")
        return 0

    status_path = RUN_DIR / "preflight_status.md"
    status_path.write_text(
        "# E1 FWB Training Run\n\n**Status: TRAINING IN PROGRESS.**\n\n"
        "Started only after the architecture and pretrained-transfer gates passed. "
        "Ultralytics logs, args, results CSV, and checkpoints are being written to this run directory.\n",
        encoding="utf-8",
    )
    started = time.perf_counter()
    try:
        model.train(
            data=str(data_path),
            epochs=50,
            imgsz=640,
            batch=16,
            optimizer="SGD",
            lr0=0.01,
            device=args.device,
            seed=0,
            deterministic=True,
            close_mosaic=10,
            workers=2,
            patience=100,
            amp=True,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            pretrained=False,
            project=str(PROJECT_ROOT / "results"),
            name="fwb_optimization_full",
            exist_ok=True,
            save=True,
            val=True,
            plots=True,
        )
    except Exception as error:
        results_csv = RUN_DIR / "results.csv"
        completed_epochs = _completed_epoch_rows(results_csv)
        status_path.write_text(
            "# E1 FWB Training Run\n\n"
            "**Status: FAILED / INCOMPLETE.**\n\n"
            f"- Epoch rows retained: {completed_epochs}\n"
            f"- Elapsed seconds: {time.perf_counter() - started:.2f}\n"
            f"- Failure: `{type(error).__name__}: {error}`\n"
            "- See the retained Ultralytics run files in this directory for diagnostics.\n",
            encoding="utf-8",
        )
        raise
    results_csv = RUN_DIR / "results.csv"
    completed_epochs = _completed_epoch_rows(results_csv)
    status = "COMPLETE" if completed_epochs == 50 else "INCOMPLETE"
    status_path.write_text(
        "# E1 FWB Training Run\n\n"
        f"**Status: {status}.**\n\n"
        f"- Epoch rows retained: {completed_epochs}/50\n"
        f"- Elapsed seconds: {time.perf_counter() - started:.2f}\n"
        f"- Results CSV: `{results_csv}`\n"
        "- Checkpoints and training arguments are retained in this run directory.\n",
        encoding="utf-8",
    )
    if completed_epochs != 50:
        raise RuntimeError(
            f"Training returned without completing 50 epochs; retained {completed_epochs} rows"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

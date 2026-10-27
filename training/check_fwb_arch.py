"""Validate the YOLOv8n+FWB architecture before any training is permitted."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import psutil
import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.nn import tasks
from ultralytics.utils.torch_utils import get_flops

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.detector.fwb_module import FWB
from models.detector.fwb_registration import register_fwb

FWB_YAML = PROJECT_ROOT / "models" / "detector" / "yolov8n_fwb.yaml"
DATA_YAML = PROJECT_ROOT / "models" / "detector" / "data.yaml"
STOCK_YAML = (
    Path(ultralytics.__file__).resolve().parent
    / "cfg"
    / "models"
    / "v8"
    / "yolov8.yaml"
)
BASELINE_CHECKPOINT = PROJECT_ROOT / "results" / "previous_b1" / "weights" / "best.pt"
ARCH_REPORT = PROJECT_ROOT / "results" / "sprint13_architecture_report.md"
ARCH_METRICS = PROJECT_ROOT / "results" / "sprint13_architecture_metrics.json"
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
    "pothole",
    "traffic cone",
]
MIN_FREE_RAM_BYTES = 3 * 1024**3
MIN_FREE_GPU_MEMORY_BYTES = 3 * 1024**3
# Conservative operational guardrails, not measured minimum requirements of YOLOv8n+FWB.
# Keep them intact unless a separate, evidence-based resource review changes the policy.


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_outputs_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, (list, tuple)):
        return all(tensor_outputs_finite(item) for item in value)
    if isinstance(value, dict):
        return all(tensor_outputs_finite(item) for item in value.values())
    return True


def _write_blocked_report(reason: str, before_hash: str) -> None:
    cuda_available = torch.cuda.is_available()
    gpu_details = (
        f"{torch.cuda.get_device_name(0)}, "
        f"{torch.cuda.mem_get_info(0)[0] / 1024**3:.2f} GiB free VRAM"
        if cuda_available
        else "not available"
    )
    ARCH_METRICS.parent.mkdir(parents=True, exist_ok=True)
    ARCH_METRICS.write_text(
        json.dumps(
            {
                "status": "blocked_before_model_construction",
                "reason": reason,
                "baseline_checkpoint_sha256_before": before_hash,
                "baseline_checkpoint_sha256_after": sha256(BASELINE_CHECKPOINT),
                "cuda_available": cuda_available,
                "gpu": gpu_details,
                "stock_parameters": None,
                "fwb_parameters": None,
                "stock_gflops": None,
                "fwb_gflops": None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    ARCH_REPORT.write_text(
        "# Sprint 13 — FWB Architecture Validation\n\n"
        "## Status\n\n"
        "**BLOCKED — the mandatory 640×640 architecture forward check was not run.**\n\n"
        f"- Blocker: {reason}\n"
        f"- Current free system RAM: {psutil.virtual_memory().available / 1024**3:.2f} GiB\n"
        f"- CUDA available: {cuda_available}; GPU: {gpu_details}.\n"
        f"- Frozen B1 SHA-256 before check: `{before_hash}`\n"
        f"- Frozen B1 SHA-256 after check: `{sha256(BASELINE_CHECKPOINT)}`\n"
        "- No FWB detector was trained and no FWB evaluation was run.\n\n"
        "The independent FWB tensor/unit checks are separate from, and do not replace, "
        "the required full model construction and 640×640 forward pass. Training remains "
        "prohibited until this architecture gate passes on a machine with adequate memory.\n\n"
        "A small parser integration test confirmed FWB channel injection and output-channel "
        "preservation. A stock-YAML comparison test checked the insertion point and absolute "
        "reference shifts. Neither test instantiates the full detector.\n",
        encoding="utf-8",
    )


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _run(args: argparse.Namespace) -> int:
    if not BASELINE_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Frozen baseline checkpoint missing: {BASELINE_CHECKPOINT}")
    before_hash = sha256(BASELINE_CHECKPOINT)
    free_ram = psutil.virtual_memory().available
    if free_ram < int(args.min_free_ram_gb * 1024**3):
        reason = (
            f"Available RAM is below the configured safety gate "
            f"({free_ram / 1024**3:.2f} GiB available; "
            f"{args.min_free_ram_gb:.2f} GiB required)."
        )
        _write_blocked_report(reason, before_hash)
        print(reason, file=sys.stderr)
        return 2
    if not torch.cuda.is_available() and free_ram < MIN_FREE_RAM_BYTES:
        reason = (
            f"CPU architecture validation requires at least "
            f"{MIN_FREE_RAM_BYTES / 1024**3:.1f} GiB free RAM; "
            f"{free_ram / 1024**3:.2f} GiB is available."
        )
        _write_blocked_report(reason, before_hash)
        print(reason, file=sys.stderr)
        return 2
    if torch.cuda.is_available():
        free_vram, _total_vram = torch.cuda.mem_get_info(0)
        if free_vram < MIN_FREE_GPU_MEMORY_BYTES:
            reason = (
                f"Available GPU memory is below the architecture safety gate "
                f"({free_vram / 1024**3:.2f} GiB free; "
                f"{MIN_FREE_GPU_MEMORY_BYTES / 1024**3:.1f} GiB required)."
            )
            _write_blocked_report(reason, before_hash)
            print(reason, file=sys.stderr)
            return 2

    model_config = yaml.safe_load(FWB_YAML.read_text(encoding="utf-8"))
    if model_config.get("nc") != 15 or model_config.get("scale") != "n":
        raise ValueError("FWB YAML must specify YOLOv8n and 15 classes")
    if not STOCK_YAML.is_file() or not DATA_YAML.is_file():
        raise FileNotFoundError("Installed stock YAML or project data.yaml is missing")
    register_fwb()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stock_config = yaml.safe_load(STOCK_YAML.read_text(encoding="utf-8"))
    stock_config["nc"] = 15
    stock_config["scale"] = "n"
    with tempfile.TemporaryDirectory(prefix="beyond_detection_stock_yolo_") as temp_dir:
        stock_yaml = Path(temp_dir) / "yolov8n_15class.yaml"
        stock_yaml.write_text(yaml.safe_dump(stock_config, sort_keys=False), encoding="utf-8")
        stock = YOLO(str(stock_yaml), task="detect")
    stock.model.eval()
    stock.model.to(device)
    stock_parameters = _parameter_count(stock.model)
    stock_gflops = float(get_flops(stock.model, imgsz=640))
    del stock
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    custom = YOLO(str(FWB_YAML), task="detect")
    custom.model.eval()
    custom.model.to(device)
    data_config = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    class_names = data_config.get("names", [])
    if class_names != EXPECTED_CLASSES or data_config.get("nc") != 15:
        raise ValueError(f"Custom model taxonomy mismatch: {class_names}")

    blocks = [module for module in custom.model.modules() if isinstance(module, FWB)]
    if len(blocks) != 1:
        raise ValueError(f"Expected exactly one FWB module, found {len(blocks)}")
    block = blocks[0]
    layer_indexes = [
        module.i
        for module in custom.model.model
        if module is block
    ]
    if len(layer_indexes) != 1 or layer_indexes[0] != 10:
        raise ValueError(f"FWB must be parser layer 10 after SPPF, got {layer_indexes}")
    if custom.model.model[9].type != "ultralytics.nn.modules.block.SPPF":
        raise ValueError(f"Layer 9 must be SPPF, found {custom.model.model[9].type}")
    if custom.model.model[11].f != -1:
        raise ValueError("First neck layer must consume the FWB output")

    observed: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def record_shapes(_module, inputs, output) -> None:
        observed.append((tuple(inputs[0].shape), tuple(output.shape)))

    hook = block.register_forward_hook(record_shapes)
    custom_parameters = _parameter_count(custom.model)
    custom_gflops = float(get_flops(custom.model, imgsz=640))
    observed.clear()
    with torch.inference_mode():
        first = custom.model(torch.zeros(1, 3, 640, 640, device=device))
    if not tensor_outputs_finite(first):
        raise ValueError("640×640 forward output contains non-finite values")
    if not observed or observed[-1][0] != observed[-1][1]:
        raise ValueError(f"FWB changed feature shape: {observed}")
    if observed[-1][0][1] != block.c1:
        raise ValueError(f"FWB input channels {observed[-1][0][1]} != configured {block.c1}")
    if int(custom.model.model[-1].nc) != 15:
        raise ValueError(f"Expected 15 detector classes, got {custom.model.model[-1].nc}")
    del first
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    with torch.inference_mode():
        batch_output = custom.model(torch.zeros(2, 3, 640, 640, device=device))
    if not tensor_outputs_finite(batch_output):
        raise ValueError("Representative batch output contains non-finite values")
    if len(observed) != 2 or observed[-1][0][0] != 2:
        raise ValueError(f"Representative batch did not pass through FWB: {observed}")
    hook.remove()
    after_hash = sha256(BASELINE_CHECKPOINT)
    if before_hash != after_hash:
        raise RuntimeError("Frozen B1 checkpoint SHA-256 changed during architecture validation")

    metrics = {
        "status": "passed",
        "device": device,
        "imgsz": 640,
        "classes": class_names,
        "fwb_layer": layer_indexes[0],
        "fwb_input_shape": list(observed[0][0]),
        "fwb_output_shape": list(observed[0][1]),
        "representative_batch_size": 2,
        "stock_parameters": stock_parameters,
        "fwb_parameters": custom_parameters,
        "parameter_delta": custom_parameters - stock_parameters,
        "stock_gflops": stock_gflops,
        "fwb_gflops": custom_gflops,
        "gflops_delta": custom_gflops - stock_gflops,
        "baseline_checkpoint_sha256_before": before_hash,
        "baseline_checkpoint_sha256_after": after_hash,
        "ultralytics_version": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "gpu_free_memory_bytes": torch.cuda.mem_get_info(0)[0] if device == "cuda" else None,
    }
    ARCH_METRICS.parent.mkdir(parents=True, exist_ok=True)
    ARCH_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    ARCH_REPORT.write_text(
        "# Sprint 13 — FWB Architecture Validation\n\n"
        f"**Status: PASS** on `{device}` using Ultralytics {ultralytics.__version__}.\n\n"
        f"- Model: YOLOv8n + one FWB at layer {layer_indexes[0]} (immediately after SPPF).\n"
        f"- FWB feature shape: `{observed[0][0]}` → `{observed[0][1]}`; representative batch size 2 passed.\n"
        "- Input resolution: 640×640; forward outputs were finite; detector taxonomy: 15 classes.\n"
        f"- Stock 15-class YOLOv8n: {stock_parameters:,} parameters, {stock_gflops:.4f} GFLOPs.\n"
        f"- YOLOv8n+FWB: {custom_parameters:,} parameters, {custom_gflops:.4f} GFLOPs.\n"
        f"- Difference: {custom_parameters - stock_parameters:+,} parameters, "
        f"{custom_gflops - stock_gflops:+.4f} GFLOPs.\n"
        f"- Frozen B1 SHA-256 before/after: `{before_hash}` / `{after_hash}`.\n\n"
        "Full machine-readable measurements: `results/sprint13_architecture_metrics.json`.\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-free-ram-gb", type=float, default=3.0)
    args = parser.parse_args()
    if args.min_free_ram_gb <= 0:
        parser.error("--min-free-ram-gb must be positive")
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Run pretrained MiDaS-small inference on one RGB road-scene image."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "sprint6_depth_map.png"


def load_midas_small(device: str):
    torch.hub.list(
        "rwightman/gen-efficientnet-pytorch",
        trust_repo=True,
        verbose=False,
    )
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    model.to(device)
    model.eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = transforms.small_transform
    return model, transform


def run_depth(image_path: str | Path, out_path: str | Path = DEFAULT_OUTPUT) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    image_path = Path(image_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()
    model, transform = load_midas_small(device)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    if depth_map.shape != img_rgb.shape[:2]:
        raise ValueError(
            f"Depth dimensions {depth_map.shape} do not match image dimensions {img_rgb.shape[:2]}"
        )
    if not np.isfinite(depth_map).all():
        raise ValueError("MiDaS depth map contains non-finite values")

    print(f"Image path: {image_path}")
    print(f"Input dimensions: {img_rgb.shape[1]} x {img_rgb.shape[0]}")
    print(f"Depth map shape: {depth_map.shape}")
    print(f"Depth value range: min={depth_map.min():.2f}, max={depth_map.max():.2f}")
    print(
        "(Note: MiDaS output is RELATIVE inverse depth — higher value = closer object. "
        "These numbers are NOT metres; that's expected at this stage.)"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    depth_vis = axes[1].imshow(depth_map, cmap="inferno")
    axes[1].set_title("MiDaS-small depth (relative, brighter = closer)")
    axes[1].axis("off")

    fig.colorbar(depth_vis, ax=axes[1], fraction=0.046)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise OSError(f"Depth visualization was not saved correctly: {out_path}")
    print(f"Saved visualization to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python models/depth/midas_infer.py <path_to_image>")
        raise SystemExit(1)

    run_depth(sys.argv[1])

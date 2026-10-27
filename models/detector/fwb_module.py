"""Feature Weighting Block used by the Sprint 13 YOLOv8n experiment."""

from __future__ import annotations

import torch
from torch import nn


class FWB(nn.Module):
    """Apply learned channel weights using pooled feature context."""

    def __init__(self, c1: int, c2: int | None = None) -> None:
        super().__init__()
        c2 = c1 if c2 is None else c2
        if c1 != c2:
            raise ValueError(f"FWB must preserve channels: c1={c1}, c2={c2}")
        if not isinstance(c1, int) or isinstance(c1, bool) or c1 < 1:
            raise ValueError(f"c1 must be a positive integer, got {c1!r}")

        self.c1 = c1
        self.c2 = c2
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c1, c1, kernel_size=1)
        self.act1 = nn.LeakyReLU(0.1, inplace=False)
        self.act2 = nn.Hardsigmoid(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.c1:
            raise ValueError(
                f"FWB expected a BCHW tensor with {self.c1} channels, "
                f"got shape {tuple(x.shape)}"
            )
        weights = self.avgpool(x)
        weights = self.conv(weights)
        weights = self.act1(weights)
        weights = self.act2(weights)
        return x * weights

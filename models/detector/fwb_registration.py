"""Project-local registration for FWB with the pinned Ultralytics parser."""

from __future__ import annotations

import inspect

import torch
import ultralytics
from ultralytics.nn import tasks

from models.detector.fwb_module import FWB

SUPPORTED_ULTRALYTICS_VERSION = "8.4.160"


def register_fwb() -> None:
    """Register FWB in the parser's channel-preserving base-module branch."""
    if getattr(tasks, "_beyond_detection_fwb_registered", False):
        return
    if ultralytics.__version__ != SUPPORTED_ULTRALYTICS_VERSION:
        raise RuntimeError(
            "FWB parser adapter was validated against Ultralytics "
            f"{SUPPORTED_ULTRALYTICS_VERSION}, found {ultralytics.__version__}"
        )

    source = inspect.getsource(tasks.parse_model)
    needle = "            Focus,\n"
    if source.count(needle) != 1:
        raise RuntimeError(
            "Could not safely adapt this Ultralytics parse_model source: "
            "the expected base-module insertion point is missing or ambiguous"
        )

    tasks.FWB = FWB
    adapted_source = source.replace(needle, f"{needle}            FWB,\n", 1)
    exec(compile(adapted_source, inspect.getsourcefile(tasks.parse_model) or "<parse_model>", "exec"), tasks.__dict__)
    tasks._beyond_detection_fwb_registered = True

    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is not None:
        add_safe_globals([FWB])

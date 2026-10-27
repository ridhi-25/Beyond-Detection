"""Standalone tests for deterministic spatial zone mapping."""

from __future__ import annotations

if __package__ in (None, ""):
    from zone_mapping import (
        annotate_detection_zone,
        assign_zone,
        in_bottom_third_roi,
    )
else:
    from .zone_mapping import (
        annotate_detection_zone,
        assign_zone,
        in_bottom_third_roi,
    )

FRAME_W = 900
FRAME_H = 600


def test_left_zone() -> None:
    assert assign_zone([50, 100, 150, 200], FRAME_W) == "left"


def test_center_zone() -> None:
    assert assign_zone([400, 100, 500, 200], FRAME_W) == "center"


def test_right_zone() -> None:
    assert assign_zone([750, 100, 850, 200], FRAME_W) == "right"


def test_zone_boundary_exact_edges() -> None:
    assert assign_zone([250, 100, 350, 200], FRAME_W) == "center"
    assert assign_zone([550, 100, 650, 200], FRAME_W) == "right"


def test_box_in_bottom_third_roi() -> None:
    assert in_bottom_third_roi([100, 450, 200, 550], FRAME_H) is True


def test_box_outside_bottom_third_roi() -> None:
    assert in_bottom_third_roi([100, 50, 200, 150], FRAME_H) is False


def test_box_straddling_roi_boundary() -> None:
    assert in_bottom_third_roi([100, 350, 200, 450], FRAME_H) is True


def test_annotate_detection_zone_adds_both_keys() -> None:
    detection = {
        "class_name": "pothole",
        "box_xyxy": [700, 480, 800, 580],
    }

    output = annotate_detection_zone(detection, FRAME_W, FRAME_H)

    assert output["zone"] == "right"
    assert output["in_roi"] is True
    assert output is detection
    assert "zone" in detection
    assert "in_roi" in detection


if __name__ == "__main__":
    tests = [
        value
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(f"\nAll {len(tests)} tests passed.")

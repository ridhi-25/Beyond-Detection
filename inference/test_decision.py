"""Standalone tests for the pure rule-based decision layer."""

from __future__ import annotations

if __package__ in (None, ""):
    from decision import (
        decide_alert,
        get_ahead_objects,
        resolve_direction,
        urgency_word,
    )
else:
    from .decision import (
        decide_alert,
        get_ahead_objects,
        resolve_direction,
        urgency_word,
    )


def make_det(class_name, zone, in_roi, risk):
    return {
        "class_name": class_name,
        "zone": zone,
        "in_roi": in_roi,
        "risk": risk,
    }


def test_no_detections_returns_none():
    assert decide_alert([]) is None


def test_nothing_ahead_returns_none_even_if_high_risk_elsewhere():
    detections = [make_det("truck", "left", True, 0.9)]
    assert decide_alert(detections) is None


def test_ahead_but_below_threshold_returns_none():
    detections = [make_det("bicycle", "center", True, 0.1)]
    assert decide_alert(detections) is None


def test_ahead_above_threshold_triggers_alert():
    detections = [make_det("pothole", "center", True, 0.8)]
    alert = decide_alert(detections)

    assert alert is not None
    assert alert["class_name"] == "pothole"
    assert alert["urgency"] == "danger"


def test_picks_highest_risk_among_multiple_ahead_objects():
    detections = [
        make_det("person", "center", True, 0.4),
        make_det("pothole", "center", True, 0.9),
    ]
    alert = decide_alert(detections)

    assert alert is not None
    assert alert["class_name"] == "pothole"


def test_direction_prefers_lower_risk_side():
    detections = [
        make_det("pothole", "center", True, 0.8),
        make_det("bicycle", "left", False, 0.2),
        make_det("truck", "right", False, 0.7),
    ]
    alert = decide_alert(detections)

    assert alert is not None
    assert alert["direction"] == "step left"


def test_direction_is_stop_when_both_sides_risky():
    detections = [
        make_det("pothole", "center", True, 0.8),
        make_det("truck", "left", False, 0.6),
        make_det("bus", "right", False, 0.65),
    ]
    alert = decide_alert(detections)

    assert alert is not None
    assert alert["direction"] == "stop"


def test_urgency_word_bands():
    assert urgency_word(0.9) == "danger"
    assert urgency_word(0.6) == "warning"
    assert urgency_word(0.4) == "caution"
    assert urgency_word(0.75) == "danger"
    assert urgency_word(0.5) == "warning"


def test_get_ahead_objects_filters_correctly():
    detections = [
        make_det("a", "center", True, 0.5),
        make_det("b", "center", False, 0.5),
        make_det("c", "left", True, 0.9),
    ]
    ahead = get_ahead_objects(detections)

    assert len(ahead) == 1
    assert ahead[0]["class_name"] == "a"


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

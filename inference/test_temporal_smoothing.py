"""Deterministic tests for Sprint 11 temporal alert smoothing."""

from __future__ import annotations

if __package__ in (None, ""):
    from temporal_smoothing import AlertStabilizer
else:
    from .temporal_smoothing import AlertStabilizer


def make_alert(
    class_name: str = "pothole",
    direction: str = "stop",
    risk: float = 0.8,
) -> dict:
    return {
        "class_name": class_name,
        "risk": risk,
        "direction": direction,
        "urgency": "danger",
        "message": f"{class_name}, {direction}, danger",
    }


def test_single_frame_flicker_is_suppressed() -> None:
    stabilizer = AlertStabilizer()
    frames = [None, make_alert(), None, None, None]
    results = [stabilizer.update(alert, float(index)) for index, alert in enumerate(frames)]
    assert results == [None, None, None, None, None]


def test_consistent_alert_speaks_once_it_reaches_threshold_count() -> None:
    stabilizer = AlertStabilizer()
    frames = [make_alert(), make_alert(), make_alert()]
    results = [stabilizer.update(alert, float(index)) for index, alert in enumerate(frames)]
    assert results[0] is None
    assert results[1] is None
    assert results[2] is not None


def test_cooldown_suppresses_immediate_repeat() -> None:
    stabilizer = AlertStabilizer(
        window_size=5,
        min_consistent_frames=1,
        cooldown_seconds=2.0,
    )
    alert = make_alert()
    assert stabilizer.update(alert, 0.0) is alert
    assert stabilizer.update(make_alert(), 0.5) is None


def test_speaks_again_after_cooldown_elapses() -> None:
    stabilizer = AlertStabilizer(
        window_size=5,
        min_consistent_frames=1,
        cooldown_seconds=2.0,
    )
    first = make_alert()
    second = make_alert()
    assert stabilizer.update(first, 0.0) is first
    assert stabilizer.update(second, 3.0) is second


def test_different_hazard_is_not_blocked_by_cooldown_on_previous_one() -> None:
    stabilizer = AlertStabilizer(
        window_size=5,
        min_consistent_frames=1,
        cooldown_seconds=2.0,
    )
    first = make_alert("pothole", "stop")
    second = make_alert("truck", "step left")
    assert stabilizer.update(first, 0.0) is first
    assert stabilizer.update(second, 0.5) is second


def test_no_alert_frames_never_speak() -> None:
    stabilizer = AlertStabilizer()
    results = [stabilizer.update(None, float(index)) for index in range(10)]
    assert all(result is None for result in results)


def test_min_consistent_frames_cannot_exceed_window_size() -> None:
    try:
        AlertStabilizer(window_size=2, min_consistent_frames=3)
    except ValueError:
        return
    raise AssertionError("Expected ValueError when consistency count exceeds window")


def test_invalid_window_size_raises_value_error() -> None:
    for value in (0, -1, 1.5):
        try:
            AlertStabilizer(window_size=value)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for window_size={value}")


def test_invalid_cooldown_raises_value_error() -> None:
    for value in (-0.1, "2", float("nan")):
        try:
            AlertStabilizer(cooldown_seconds=value)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for cooldown_seconds={value}")


def test_same_class_but_different_direction_does_not_satisfy_consistency() -> None:
    stabilizer = AlertStabilizer(window_size=5, min_consistent_frames=3)
    frames = [
        make_alert("pothole", "stop"),
        make_alert("pothole", "step left"),
        make_alert("pothole", "stop"),
    ]
    results = [stabilizer.update(alert, float(index)) for index, alert in enumerate(frames)]
    assert results == [None, None, None]


def test_none_occupies_window_and_does_not_count_toward_consistency() -> None:
    stabilizer = AlertStabilizer(window_size=3, min_consistent_frames=3)
    frames = [make_alert(), make_alert(), None, make_alert()]
    results = [stabilizer.update(alert, float(index)) for index, alert in enumerate(frames)]
    assert results == [None, None, None, None]
    assert list(stabilizer.history) == frames[-3:]


def test_different_class_same_direction_does_not_satisfy_consistency() -> None:
    stabilizer = AlertStabilizer(window_size=5, min_consistent_frames=3)
    frames = [
        make_alert("pothole", "stop"),
        make_alert("truck", "stop"),
        make_alert("pothole", "stop"),
    ]
    results = [stabilizer.update(alert, float(index)) for index, alert in enumerate(frames)]
    assert results == [None, None, None]


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

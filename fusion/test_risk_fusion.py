"""Standalone tests for the pure-Python risk fusion module."""

from __future__ import annotations

from copy import deepcopy

if __package__ in (None, ""):
    from risk_fusion import compute_risk, compute_risks_for_detections, get_tier
else:
    from fusion.risk_fusion import compute_risk, compute_risks_for_detections, get_tier


def test_tier_lookup() -> None:
    assert get_tier("pothole") == "critical"
    assert get_tier("truck") == "high"
    assert get_tier("person") == "low"
    assert get_tier("some_unknown_class") == "high"


def test_close_confident_pothole_is_highest_risk() -> None:
    result = compute_risk("pothole", confidence=0.9, norm_depth=0.95)
    assert result["tier"] == "critical"
    assert result["risk"] > 0.7
    assert result["risk"] == 0.855


def test_distant_low_confidence_object_is_low_risk() -> None:
    result = compute_risk("rickshaw", confidence=0.4, norm_depth=0.1)
    assert result["tier"] == "low"
    assert result["risk"] < 0.05
    assert result["risk"] == 0.012


def test_same_class_closer_object_scores_higher() -> None:
    far = compute_risk("private car", confidence=0.8, norm_depth=0.2)
    near = compute_risk("private car", confidence=0.8, norm_depth=0.9)
    assert near["risk"] > far["risk"]


def test_tier_ordering_holds_at_equal_confidence_and_depth() -> None:
    critical = compute_risk("pothole", confidence=0.7, norm_depth=0.7)
    high = compute_risk("truck", confidence=0.7, norm_depth=0.7)
    low = compute_risk("bicycle", confidence=0.7, norm_depth=0.7)
    assert critical["risk"] > high["risk"] > low["risk"]
    assert critical["risk"] == 0.49
    assert high["risk"] == 0.294
    assert low["risk"] == 0.147


def test_batch_helper_preserves_order_and_count() -> None:
    detections = [
        {"class_name": "person", "confidence": 0.8, "norm_depth": 0.5},
        {"class_name": "pothole", "confidence": 0.6, "norm_depth": 0.9},
    ]
    original = deepcopy(detections)
    output = compute_risks_for_detections(detections)
    assert len(output) == 2
    assert output[0]["class_name"] == "person"
    assert output[1]["class_name"] == "pothole"
    assert detections == original


def test_invalid_confidence_raises() -> None:
    try:
        compute_risk("person", confidence=1.5, norm_depth=0.5)
    except ValueError as error:
        assert str(error) == "confidence out of range: 1.5"
    else:
        raise AssertionError("Expected ValueError for out-of-range confidence")


TESTS = [
    test_tier_lookup,
    test_close_confident_pothole_is_highest_risk,
    test_distant_low_confidence_object_is_low_risk,
    test_same_class_closer_object_scores_higher,
    test_tier_ordering_holds_at_equal_confidence_and_depth,
    test_batch_helper_preserves_order_and_count,
    test_invalid_confidence_raises,
]


if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"PASS: {test.__name__}")
    print("All 7 tests passed.")

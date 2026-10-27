"""Pure-Python first-pass risk scoring from class, confidence, and relative depth."""

TIER_WEIGHTS = {
    "critical": 1.0,
    "high": 0.6,
    "low": 0.3,
}

# These tier assignments are first-pass methodology design choices, not validated
# empirical results; they must be treated as such in any research write-up.
CLASS_TIER = {
    "pothole": "critical",
    "truck": "high",
    "pickup truck": "high",
    "private car": "high",
    "bus": "high",
    "micro bus": "high",
    "covered van": "high",
    "human hauler": "high",
    "motorcycle": "high",
    "auto rickshaw": "high",
    "rickshaw van": "high",
    "person": "low",
    "rickshaw": "low",
    "bicycle": "low",
    "traffic cone": "low",
}


def get_tier(class_name: str) -> str:
    """Return the assigned tier, conservatively defaulting unknown classes to high."""
    return CLASS_TIER.get(class_name, "high")


def compute_risk(class_name: str, confidence: float, norm_depth: float) -> dict:
    """Compute a bounded first-pass score from detection confidence and relative depth."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence out of range: {confidence}")
    if not 0.0 <= norm_depth <= 1.0:
        raise ValueError(f"norm_depth out of range: {norm_depth}")

    tier = get_tier(class_name)
    risk = TIER_WEIGHTS[tier] * confidence * norm_depth
    return {
        "class_name": class_name,
        "confidence": confidence,
        "norm_depth": norm_depth,
        "tier": tier,
        "risk": round(risk, 4),
    }


def compute_risks_for_detections(detections: list[dict]) -> list[dict]:
    """Compute risk results in input order without mutating detection dictionaries."""
    return [
        compute_risk(
            detection["class_name"],
            detection["confidence"],
            detection["norm_depth"],
        )
        for detection in detections
    ]

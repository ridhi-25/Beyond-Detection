"""Pure rule-based alert decisions from risk-scored spatial detections."""

ALERT_THRESHOLD = 0.35

URGENCY_BANDS = [
    (0.75, "danger"),
    (0.5, "warning"),
    (0.0, "caution"),
]


def urgency_word(risk: float) -> str:
    for threshold, word in URGENCY_BANDS:
        if risk >= threshold:
            return word
    return "caution"


def get_ahead_objects(detections: list[dict]) -> list[dict]:
    """Return detections in the center zone and bottom-third ROI."""
    return [
        detection
        for detection in detections
        if detection["zone"] == "center" and detection["in_roi"]
    ]


def resolve_direction(detections: list[dict]) -> str:
    """Choose the lower-risk side, or stop if both sides are risky."""
    left_risks = [
        detection["risk"]
        for detection in detections
        if detection["zone"] == "left"
    ]
    right_risks = [
        detection["risk"]
        for detection in detections
        if detection["zone"] == "right"
    ]

    max_left = max(left_risks, default=0.0)
    max_right = max(right_risks, default=0.0)

    if max_left >= ALERT_THRESHOLD and max_right >= ALERT_THRESHOLD:
        return "stop"

    return "step left" if max_left < max_right else "step right"


def decide_alert(detections: list[dict]) -> dict | None:
    """Return one alert for the riskiest qualifying ahead detection, or silence."""
    ahead = get_ahead_objects(detections)
    if not ahead:
        return None

    top_hazard = max(ahead, key=lambda detection: detection["risk"])
    if top_hazard["risk"] < ALERT_THRESHOLD:
        return None

    direction = resolve_direction(detections)
    urgency = urgency_word(top_hazard["risk"])

    return {
        "class_name": top_hazard["class_name"],
        "risk": top_hazard["risk"],
        "direction": direction,
        "urgency": urgency,
        "message": f"{top_hazard['class_name']}, {direction}, {urgency}",
    }

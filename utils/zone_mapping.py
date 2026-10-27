"""Rule-based horizontal zone and bottom-third ROI mapping."""


def assign_zone(box_xyxy: list[float], frame_width: int) -> str:
    """Classify a box by its horizontal center in original-frame pixels."""
    x1, _, x2, _ = box_xyxy
    center_x = (x1 + x2) / 2
    third = frame_width / 3

    if center_x < third:
        return "left"
    elif center_x < 2 * third:
        return "center"
    else:
        return "right"


def in_bottom_third_roi(box_xyxy: list[float], frame_height: int) -> bool:
    """Return whether a box's vertical center lies in the bottom third."""
    _, y1, _, y2 = box_xyxy
    center_y = (y1 + y2) / 2
    roi_start_y = (2 / 3) * frame_height
    return center_y >= roi_start_y


def annotate_detection_zone(
    detection: dict,
    frame_width: int,
    frame_height: int,
) -> dict:
    """Add zone and bottom-third ROI fields to a detection in place."""
    box = detection["box_xyxy"]
    detection["zone"] = assign_zone(box, frame_width)
    detection["in_roi"] = in_bottom_third_roi(box, frame_height)
    return detection

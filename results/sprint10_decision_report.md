# Sprint 10 — Decision Logic + Alert Threshold

## 1. Objective

Implement a stateless, pure-Python decision layer that consumes the complete frame's risk-scored and spatially tagged detections. It identifies the highest-risk qualifying ahead object, selects urgency and a rule-based direction, or remains silent.

## 2. Decision architecture

The integrated frame flow is:

1. Sprint 7 produces detections, confidence, and normalized relative depth.
2. Sprint 8's existing `compute_risk()` supplies risk and tier.
3. Sprint 9's existing spatial functions supply zone and bottom-third ROI membership.
4. The pipeline collects all detections for the frame and calls `decide_alert()` once.
5. The decision layer prints a single text alert or a silence message.

`inference/decision.py` consumes these values only. It does not load models, compute risk or depth, assign zones, modify detections, or maintain state.

## 3. Alert threshold

`ALERT_THRESHOLD = 0.35`.

The threshold is a **tunable initial configuration / placeholder**. It has not been empirically validated, learned from data, statistically optimized, or established as optimal. It is a starting value to examine and tune in the experimental phase. The alert comparison is inclusive: risk below 0.35 is silent; risk equal to or above 0.35 can trigger an alert when an ahead object exists.

## 4. Urgency bands

| Risk | Urgency |
|---|---|
| `risk >= 0.75` | danger |
| `0.5 <= risk < 0.75` | warning |
| `risk < 0.5` | caution |

No additional urgency levels are defined.

## 5. Ahead-object definition

An ahead object must satisfy both conditions:

- `zone == "center"`
- `in_roi == True`

Risk is not part of this filtering step. Left/right detections are excluded from the ahead list but remain inputs to directional resolution.

## 6. Direction resolution

The decision layer compares the maximum detected risk in the left zone with the maximum detected risk in the right zone, using the full frame detection list. If both side maxima are at least 0.35, the action is `stop`. Otherwise it selects the lower-risk side. Equal side risks resolve to `step right`, as specified.

This directional recommendation is rule-based and depends on the maximum detected risk in the left and right zones. It is a functional rule, not a safety guarantee.

## 7. Silence behavior

The result is `None` (and the integrated pipeline prints silence) when there are no center + bottom-third detections or when the highest-risk ahead detection is below 0.35. The decision is made only after the complete frame's detections are collected.

## 8. Unit-test results

All 9 standalone tests passed:

1. `test_no_detections_returns_none`
2. `test_nothing_ahead_returns_none_even_if_high_risk_elsewhere`
3. `test_ahead_but_below_threshold_returns_none`
4. `test_ahead_above_threshold_triggers_alert`
5. `test_picks_highest_risk_among_multiple_ahead_objects`
6. `test_direction_prefers_lower_risk_side`
7. `test_direction_is_stop_when_both_sides_risky`
8. `test_urgency_word_bands`
9. `test_get_ahead_objects_filters_correctly`

Regression results: Sprint 8 risk-fusion tests passed **7/7**; Sprint 9 zone-mapping tests passed **8/8**.

## 9. Manual decision and threshold checks

- Empty detections: no alert.
- High-risk left-side detection without an ahead detection: no alert.
- Ahead pothole at risk 0.8: selects the pothole with `danger` urgency.
- Ahead risk 0.8, left risk 0.2, right risk 0.7: `step left`.
- Left risk 0.6 and right risk 0.65: `stop`.
- Ahead risk 0.34: silent.
- Ahead risk 0.35: alert.
- Ahead risk 0.36: alert.
- Urgency boundaries: 0.75 is `danger`; 0.5 is `warning`.
- Equal side risks below threshold resolve to `step right`.

## 10. Real-pipeline integration results

The existing Sprint 7 pipeline was minimally extended to collect class, confidence, normalized depth, Sprint 8 risk/tier, Sprint 9 zone/ROI, and bounding box for each detection. `decide_alert()` is called once after all frame detections are collected. The existing YOLO checkpoint, MiDaS model, confidence settings, depth normalization/sampling, risk formula, and annotated image dimensions were not changed.

All runs used CPU inference on existing RSUD20K images; no images or labels were written. Ahead count and highest ahead risk are based strictly on center zone plus bottom-third ROI:

| Image | Detections | Ahead detections | Highest ahead risk | Alert |
|---|---:|---:|---:|---|
| `images/train/train0.jpg` | 7 | 0 | N/A | No — silent |
| `images/train/train1.jpg` | 4 | 1 | 0.427 (pickup truck) | Yes — caution, step left |
| `images/val/val0.jpg` | 8 | 0 | N/A | No — silent |

The observed alert message was:

`pickup truck, step left, caution`

The silent frames printed:

`(silent — nothing above threshold in path)`

The Sprint 7 script writes its annotated visualization to the existing `results/sprint7_yolo_midas_combined.jpg` path; the final file reflects the last tested frame (`val0.jpg`).

These three runs verify functional execution only. They do not validate the threshold, risk mapping, or recommendations empirically.

## 11. Limitations

- The 0.35 alert threshold is not tuned or empirically validated.
- Risk and relative depth are inherited from earlier components; relative depth is not metric distance.
- Direction uses detected side-zone maxima only and does not account for free-space geometry, occlusion, trajectories, or unseen hazards.
- A small number of example frames establishes pipeline behavior, not real-world safety or performance.
- Alerts are text-only.

## 12. Scope and status

- No model training or fine-tuning occurred.
- No dataset or labels were modified.
- No Sprint 8 risk formula or Sprint 9 zone/ROI definition was changed.
- No TTS or audio was added.
- No temporal smoothing, cooldown, tracking, or cross-frame state was added.
- Sprint 11 has **not** been started.

**Sprint 10 complete.**

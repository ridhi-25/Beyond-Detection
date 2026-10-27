# Sprint 8 — Risk Fusion Module

## Objective

Implement and test a pure-Python first-pass risk score:

`risk = tier_weight × confidence × normalized_depth`

Here, confidence is in `[0, 1]`, Sprint 7 normalized relative inverse depth is in `[0, 1]` (`0` farther and `1` closer), and tier weight depends on the detection class. With the configured ranges and weights, the resulting score is bounded in `[0, 1]`.

## Tier weights

| Tier | Weight |
|---|---:|
| critical | 1.0 |
| high | 0.6 |
| low | 0.3 |

## Class-tier assignment

| Class | Tier |
|---|---|
| person | low |
| rickshaw | low |
| rickshaw van | high |
| auto rickshaw | high |
| truck | high |
| pickup truck | high |
| private car | high |
| motorcycle | high |
| bicycle | low |
| bus | high |
| micro bus | high |
| covered van | high |
| human hauler | high |
| pothole | critical |
| traffic cone | low |

This taxonomy matches the merged dataset's existing 15-class ordering; no class names or ordering were changed.

**The class-tier assignment is a first-pass design choice, not a validated empirical result.** It follows the proposed rationale: critical for direct fall/trip hazards not necessarily captured by collision logic; high for motorized vehicles with greater potential collision severity; and low for non-motorized/slower-moving objects or advisory markers. These assignments must be treated as design choices in the paper, not evidence-backed findings.

Unknown classes default conservatively to the `high` tier (weight `0.6`).

## Implementation

`fusion/risk_fusion.py` is a pure-Python module with no YOLO, MiDaS, OpenCV, image, dataset, model, or checkpoint dependencies. `compute_risk()` validates confidence and normalized depth, selects the tier, and returns the inputs, tier, and risk rounded to four decimal places. `compute_risks_for_detections()` preserves input order and count and does not mutate the input dictionaries.

## Unit tests

Pytest is not installed in the project venv, so the documented standalone fallback was used:

`cd fusion; ..\.venv\Scripts\python.exe test_risk_fusion.py`

**Result: 7/7 tests passed.** Coverage includes tier lookup and unknown-class fallback, close/confident pothole, distant/low-confidence rickshaw, monotonicity with depth for a fixed class, tier ordering, batch ordering/count/non-mutation, and invalid confidence.

## Manual numerical verification

- Pothole: `1.0 × 0.9 × 0.95 = 0.855`
- Rickshaw: `0.3 × 0.4 × 0.1 = 0.012`
- Truck: `0.6 × 0.7 × 0.7 = 0.294`
- Bicycle: `0.3 × 0.7 × 0.7 = 0.147`
- Unknown class: defaults to high; `0.6 × 0.8 × 0.8 = 0.384`

The module returns these values to four decimal places.

## Sprint 7 integration

The optional minimal integration was performed in `inference/yolo_midas_combined.py`. After the existing YOLO confidence and Sprint 7 normalized box depth are calculated, it calls `compute_risk()` and prints the tier and risk for observation. The normalized value is clamped only at the call boundary, as permitted by the Sprint 8 brief. YOLO inference, MiDaS inference, depth calculation, box mean sampling, and annotated image labeling remain unchanged. Risk does not filter, sort, classify, or trigger an alert.

The same Sprint 3/6/7 frame was rerun. All 7 detections received the following observational risk outputs:

| Class | Confidence | Normalized depth | Tier | Risk |
|---|---:|---:|---|---:|
| rickshaw | 0.94 | 0.74 | low | 0.208 |
| bicycle | 0.91 | 0.24 | low | 0.065 |
| auto rickshaw | 0.89 | 0.17 | high | 0.091 |
| private car | 0.83 | 0.15 | high | 0.075 |
| rickshaw | 0.79 | 0.19 | low | 0.046 |
| rickshaw | 0.76 | 0.17 | low | 0.039 |
| motorcycle | 0.64 | 0.22 | high | 0.085 |

These outputs verify implementation behavior only. The tiers and resulting risk values are not empirically validated and must not be interpreted as validated danger assessments.

## Limitations and scope

- Tier weights and class assignments are first-pass design choices, not empirical results.
- Input depth is relative inverse depth; no metric distance was computed.
- No threat classification, alert logic, risk threshold, or navigation behavior was added.
- No model training or fine-tuning occurred.
- No dataset or split was modified.
- Sprint 9 has **not** been started.

## Status

**Sprint 8 complete.** The independent module tests pass, the manual calculations match, and the optional Sprint 7 integration was exercised on the established sample.

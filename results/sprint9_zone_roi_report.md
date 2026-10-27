# Sprint 9 — Left/Center/Right Zones + Bottom-Third ROI

## 1. Objective

Implement and test two independent, deterministic spatial rules for detections: horizontal left/center/right zone assignment and whether a bounding-box center lies in the bottom-third ahead ROI. The rules use original-frame pixel coordinates and are not learned models.

## 2. Horizontal zone methodology

For a box `[x1, y1, x2, y2]`, `center_x = (x1 + x2) / 2`. With the original frame width `W`:

- left: `center_x < W / 3`
- center: `W / 3 <= center_x < 2 * W / 3`
- right: `center_x >= 2 * W / 3`

The exact first boundary belongs to center; the exact second boundary belongs to right.

## 3. Bottom-third ROI methodology

For the same box, `center_y = (y1 + y2) / 2`. With original frame height `H`, the box is inside the ahead ROI when:

`center_y >= (2 / 3) * H`

The exact ROI boundary is included.

## 4. Unit-test results

All 8 standalone tests passed:

1. `test_left_zone`
2. `test_center_zone`
3. `test_right_zone`
4. `test_zone_boundary_exact_edges`
5. `test_box_in_bottom_third_roi`
6. `test_box_outside_bottom_third_roi`
7. `test_box_straddling_roi_boundary`
8. `test_annotate_detection_zone_adds_both_keys`

The helper mutates the supplied detection dictionary in place to add `zone` and `in_roi`.

## 5. Manual sanity checks

| Input/check | Result |
|---|---|
| `assign_zone([50, 100, 150, 200], 900)` | `left` |
| `assign_zone([400, 100, 500, 200], 900)` | `center` |
| `assign_zone([750, 100, 850, 200], 900)` | `right` |
| `assign_zone([250, 100, 350, 200], 900)`; center x = 300 | `center` |
| `assign_zone([550, 100, 650, 200], 900)`; center x = 600 | `right` |
| `in_bottom_third_roi([100, 450, 200, 550], 600)` | `True` |
| `in_bottom_third_roi([100, 50, 200, 150], 600)` | `False` |
| `in_bottom_third_roi([100, 350, 200, 450], 600)`; center y = 400 | `True` |

## 6. Sprint 7 integration results

The minimal print-only integration was run on:

`data/raw/rsud20k_download/verified_download/extracted/rsud20k/images/train/train0.jpg`

The original frame was 1920 × 1080. Coordinates came from the YOLO result's `[x1, y1, x2, y2]` boxes and were classified using the original `img_bgr.shape` dimensions. All 7 detections completed successfully:

| Detection class | Zone | In bottom-third ROI |
|---|---|---|
| rickshaw | left | True |
| bicycle | left | True |
| auto rickshaw | center | False |
| private car | center | False |
| rickshaw | center | False |
| rickshaw | center | False |
| motorcycle | left | True |

No right-zone detection appeared in this sample; the right-zone behavior is covered by the isolated unit and manual tests. The established YOLO, MiDaS, depth-sampling, and risk outputs completed as before, and the existing Sprint 7 annotated output was saved. Zone/ROI values are observational only.

## 7. Limitations

- These are deterministic geometric rules, not learned or empirically validated models.
- ROI membership depends only on the bounding-box vertical center, not box size, depth, or apparent proximity.
- The sample contained no right-zone detections, so that zone was verified with dummy-box tests rather than a real detection.

## 8. Scope and status

- No model training or fine-tuning occurred.
- No dataset or labels were modified.
- The Sprint 8 risk formula was not changed.
- No alert, threat classification, filtering, or decision logic was added.
- No metric distance estimation or temporal smoothing was added.
- Sprint 10 has **not** been started.

**Sprint 9 complete.**

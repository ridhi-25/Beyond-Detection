# Sprint 11 — Temporal Smoothing / Cooldown

## 1. Objective

Add a small stateful wrapper around the frozen Sprint 10 `decide_alert()` output to suppress single-frame alert flicker, require repeated recent alert decisions, and suppress repeated identical messages during a cooldown. Sprint 10 decision generation itself remains unchanged.

## 2. Motivation

Sprint 10 makes an independent decision for each frame. In a future continuous stream, intermittent one-frame decisions could flicker, and identical messages could be emitted repeatedly. Sprint 11 verifies the specified history/cooldown behavior with deterministic unit tests; it does not establish that real-world flicker or false alarms are reduced.

## 3. Temporal smoothing methodology

`inference/temporal_smoothing.py` implements `AlertStabilizer`. Each call to `update(raw_alert, current_time)` appends the supplied alert or `None` to a bounded recent history. The candidate is the current frame's alert. It is eligible only when at least the configured count of entries in the current window have the same `class_name` and `direction` as that candidate.

Consistency counts matching decisions anywhere in the current window; frames do **not** need to be consecutive. `None` entries remain in the deque and never count as matching alerts. There is no risk, depth, or confidence averaging.

## 4. Window size

Default `window_size = 5`. The five-frame window is a starting design parameter and has not been empirically optimized.

## 5. Minimum consistency count

Default `min_consistent_frames = 3`. This is a **3-of-5 recent-window count**, not three consecutive frames. Three-of-five is a starting design parameter and has not been empirically optimized.

The constructor rejects non-positive/non-integer window sizes, non-positive/non-integer consistency counts, and consistency counts larger than the window.

## 6. Cooldown duration

Default `cooldown_seconds = 2.0`. This two-second cooldown is a starting design parameter and has not been empirically optimized.

Cooldown compares the exact `message` string to the last returned message. The same message is suppressed while elapsed caller-supplied time is less than the configured duration. A different message is not blocked by the previous message's cooldown. Time is supplied by the caller; `update()` does not call a wall clock.

## 7. Consistency definition

Consistency uses exactly `class_name + direction`. Same class and direction count, even if other alert fields differ. A different class or a different direction does not count toward the current candidate. No track ID or object identity is inferred.

## 8. Cooldown definition

Cooldown applies only when the candidate's exact message equals `last_spoken_message`. A new hazard/message can be returned during the prior message's cooldown. There is no urgency-escalation override: a changed urgency is only treated as a new message if the actual message string differs; no special urgency comparison or override is added.

## 9. Why decision-output smoothing is used

The wrapper operates on Sprint 10's complete alert decision rather than smoothing box coordinates or object-level risk. This is the intended small-scope method and avoids duplicating or altering Sprint 8 risk fusion, Sprint 9 spatial logic, or Sprint 10 decision logic.

## 10. Why object tracking is deliberately not used

The input has no object identity. This sprint deliberately uses only the decision fields `class_name` and `direction`; it performs no object tracking, re-identification, association, or cross-frame object state.

## 11. Unit-test results

All **12** standalone tests passed:

1. `test_single_frame_flicker_is_suppressed`
2. `test_consistent_alert_speaks_once_it_reaches_threshold_count`
3. `test_cooldown_suppresses_immediate_repeat`
4. `test_speaks_again_after_cooldown_elapses`
5. `test_different_hazard_is_not_blocked_by_cooldown_on_previous_one`
6. `test_no_alert_frames_never_speak`
7. `test_min_consistent_frames_cannot_exceed_window_size`
8. `test_invalid_window_size_raises_value_error`
9. `test_invalid_cooldown_raises_value_error`
10. `test_same_class_but_different_direction_does_not_satisfy_consistency`
11. `test_none_occupies_window_and_does_not_count_toward_consistency`
12. `test_different_class_same_direction_does_not_satisfy_consistency`

The tests use supplied timestamps/fake clocks; they are deterministic and do not sleep.

## 12. Edge-case results

- Isolated flicker surrounded by `None`: suppressed.
- Three matching class/direction alerts in the window: returned when the third matching entry arrives.
- Identical message at 0.5 seconds after return: suppressed; at 3 seconds: returned again.
- Different hazard/message during the earlier message's cooldown: returned.
- `None` frames are stored and displace older entries as the deque advances.
- Matching alerts separated by `None` can still count if they remain within the current window; the implementation does not require consecutiveness.
- Impossible window/consistency settings and a negative or non-numeric cooldown raise `ValueError`.

## 13. Regression results

- Sprint 8 risk-fusion tests: **7/7 passed**.
- Sprint 9 zone/ROI tests: **8/8 passed**.
- Sprint 10 decision tests: **9/9 passed**.
- Sprint 11 temporal-smoothing tests: **12/12 passed**.

`inference/decision.py` was not modified. Its SHA-256 matched the value recorded before Sprint 11 implementation.

## 14. Limitations

- The 5-frame window, 3-of-5 minimum, and 2-second cooldown are starting design parameters; none has been empirically optimized.
- Tests establish deterministic temporal logic only. No real sequential/live video was collected, so reduced real-world flicker or false alarms have not been demonstrated.
- A class/direction pair is not object identity; similar hazards may be counted together, while one hazard whose class/direction changes may not be.
- Caller timestamps must be monotonic for cooldown timing to have the intended interpretation.
- There is no urgency override, tracking, camera input, audio, TTS, or cross-frame object association.

## 15. Future enhancement: urgency escalation

An explicit urgency-escalation policy could be considered in a separately specified future experiment. Sprint 11 does not automatically bypass cooldown for a more urgent alert. Only the exact message equality rule currently determines whether cooldown applies.

## 16. Relationship to Sprint 12

Sprint 11 provides a tested stateful component but does not integrate it with the combined inference loop. Sprint 12 remains responsible for any live webcam loop and TTS integration. No webcam was opened or camera code added, and no TTS/audio was used.

**Sprint 11 complete. Sprint 12 has not been started.**

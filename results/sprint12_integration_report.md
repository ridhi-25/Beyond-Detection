# Sprint 12 — TTS Output and Full Pipeline Integration

## Objective

Integrate the frozen B1 detector and MiDaS-small depth estimate with the existing risk fusion, zone/ROI mapping, Sprint 10 decision, and Sprint 11 temporal stabilizer. Validate the complete video pipeline and non-blocking file-based TTS without requiring a physical webcam or speaker.

## Environment and model

- Project Python: 3.12.3 virtual environment.
- PyTorch: 2.14.0+cpu; CUDA unavailable; CPU inference used.
- Ultralytics: 8.4.160.
- OpenCV: 5.0.0.
- Detector: existing frozen B1 checkpoint `results/previous_b1/weights/best.pt`; checkpoint taxonomy was checked against the expected 15 classes at load time.
- Depth model: MiDaS-small, loaded through the existing Sprint 7 helper.
- No training or fine-tuning was performed.

## Pipeline

The integrated order is video/webcam capture → B1 YOLO detection → MiDaS-small depth map → per-box depth sampling and normalization → existing Sprint 8 risk score → existing Sprint 9 zone/ROI fields → Sprint 10 alert decision → Sprint 11 stabilization → asynchronous TTS.

The implementation reuses `compute_risk()` and zone/ROI behavior through the existing `score_and_tag_detection()` helper, `decide_alert()`, and `AlertStabilizer`. Sprint 8–11 methodology and source files were not changed. The stabilizer configuration is `window_size=5`, `min_consistent_frames=3`, and `cooldown_seconds=2.0`. `CAPTURE_INTERVAL_SEC=0.5`. Integer camera sources, including `VIDEO_SOURCE=0`, are supported; no webcam was opened during validation.

`TTSEngine` sends playback messages to a background worker. `FileTTSEngine` queues messages without blocking frame processing and synthesizes distinct WAV files during flush. TTS enqueue latency, rather than the asynchronous worker's later synthesis duration, is recorded in the frame's `tts_time_s`.

## Recorded input and validation

- Input: `results/sprint12_test_input.mp4`
- Video properties: 15 frames, 2 FPS, 1920 × 1080.
- Selected frames: all 15; at 2 FPS the recording naturally supplies one frame per 0.5-second capture interval.
- Device: CPU.
- TTS mode: file-based; no local audio playback required.
- Output log: `results/sprint12_frame_log.csv`.
- Annotated frames: `results/sprint12_annotated/frame_00000.jpg` through `frame_00014.jpg`.
- Audio outputs: three WAV files in `results/tts_outputs/`, referenced by their exact paths in the CSV.

All 15 frames completed with no failed depth samples. The pipeline produced 95 detections and 3 stabilized alerts; each stabilized alert has a non-empty WAV output. The frame log contains detection counts, raw and stable alert JSON, YOLO duration, MiDaS duration, fusion/zone/decision/smoothing duration, TTS enqueue latency, total processing duration, measured per-frame processing FPS, device, source, and detection summaries.

Mean measured per-frame processing FPS was **4.6523** (mean total processing time **0.2379 s**). Mean YOLO time was **0.0755 s**, mean MiDaS time **0.1390 s**, mean fusion/zone/decision/smoothing time **0.0090 s**, and mean TTS enqueue latency **0.0000074 s**. These are measurements of this brief CPU integration run only, not a benchmark or performance guarantee.

## Files created or updated

- Created `inference/tts_output.py`.
- Created `inference/live_pipeline.py`.
- Updated `results/sprint12_frame_log.csv`.
- Created annotated frame images in `results/sprint12_annotated/`.
- Created three run-specific file-TTS WAV outputs in `results/tts_outputs/`.
- Created this report.
- Updated `README.md` and `results/experiment_log.csv`.
- Reused the already existing `results/sprint12_test_input.mp4` and `preprocessing/make_test_video.py`; neither was regenerated.

## Scope and limitations

- No training, fine-tuning, dataset or label modification, or checkpoint modification occurred.
- The physical webcam path (`VIDEO_SOURCE=0`) and real audible playback were implemented but not tested because the recorded-video/file-TTS path sufficed for validation.
- MiDaS output is relative depth, not metric distance.
- Risk tiers and alert decisions retain their earlier first-pass methodology; this run does not validate accuracy, safety, or threat detection.
- Sprint 13 has not started.

## Validation status

The recorded CPU pipeline completed all 15 frames, generated the per-frame CSV, annotated output frames, and three non-empty WAV files. All Sprint 8–11 regression tests passed (7/7, 8/8, 9/9, and 12/12 respectively); all three Sprint 12 TTS tests passed. **Sprint 12 integration validation: COMPLETE.**

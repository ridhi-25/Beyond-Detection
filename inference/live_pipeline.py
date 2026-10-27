"""Run the frozen B1 + MiDaS + risk/decision pipeline on video or a webcam."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.temporal_smoothing import AlertStabilizer
from inference.tts_output import FileTTSEngine, TTSEngine
from inference.decision import decide_alert
from inference.yolo_midas_combined import (
    EXPECTED_CLASSES,
    get_depth_map,
    load_midas_small,
    sample_box_depth,
    score_and_tag_detection,
)


CAPTURE_INTERVAL_SEC = 0.5
CHECKPOINT = PROJECT_ROOT / "results" / "previous_b1" / "weights" / "best.pt"
LOG_PATH = PROJECT_ROOT / "results" / "sprint12_frame_log.csv"
ANNOTATED_DIR = PROJECT_ROOT / "results" / "sprint12_annotated"
TTS_OUTPUT_DIR = PROJECT_ROOT / "results" / "tts_outputs"
LOG_FIELDS = [
    "frame_idx",
    "timestamp",
    "n_detections",
    "n_valid_detections",
    "raw_alert",
    "stable_alert",
    "yolo_time_s",
    "midas_time_s",
    "fusion_zone_decision_time_s",
    "tts_time_s",
    "total_time_s",
    "actual_fps",
    "tts_file",
    "device",
    "input_source",
    "processed_detections",
]


def _parse_source(source: str) -> str | int:
    if source.isdecimal():
        return int(source)
    return str(Path(source).expanduser().resolve())


def _load_detector(checkpoint: Path, device: str) -> YOLO:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Verified B1 checkpoint does not exist: {checkpoint}")
    model = YOLO(str(checkpoint))
    names = model.names
    class_names = [names[index] for index in range(len(names))]
    if class_names != EXPECTED_CLASSES:
        raise ValueError(f"B1 class taxonomy mismatch: {class_names}")
    model.to(device)
    return model


def run_pipeline(
    source: str | int,
    *,
    tts_mode: str = "file",
    device: str | None = None,
    max_frames: int | None = None,
    checkpoint: Path = CHECKPOINT,
    log_path: Path = LOG_PATH,
    annotated_dir: Path = ANNOTATED_DIR,
    tts_output_dir: Path = TTS_OUTPUT_DIR,
) -> dict[str, Any]:
    """Process sampled video frames and return run summary metadata."""
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    if tts_mode not in {"file", "play"}:
        raise ValueError("tts_mode must be 'file' or 'play'")
    source = _parse_source(str(source)) if isinstance(source, str) else source
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    log_path = Path(log_path).expanduser().resolve()
    annotated_dir = Path(annotated_dir).expanduser().resolve()
    tts_output_dir = Path(tts_output_dir).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open video source: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    is_file = not isinstance(source, int)
    if not np.isfinite(fps) or fps <= 0:
        fps = 0.0

    detector = _load_detector(checkpoint, device)
    midas_model, midas_transform = load_midas_small(device)
    stabilizer = AlertStabilizer(
        window_size=5,
        min_consistent_frames=3,
        cooldown_seconds=2.0,
    )
    tts_engine = (
        FileTTSEngine(tts_output_dir)
        if tts_mode == "file"
        else TTSEngine()
    )

    rows: list[dict[str, Any]] = []
    frames_read = 0
    next_capture_time = 0.0
    wall_start = time.perf_counter()
    try:
        with log_path.open("w", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
            writer.writeheader()

            while max_frames is None or len(rows) < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_idx = frames_read
                frames_read += 1

                if is_file and fps > 0:
                    timestamp = frame_idx / fps
                else:
                    timestamp = time.perf_counter() - wall_start
                if timestamp + 1e-9 < next_capture_time:
                    continue
                next_capture_time = timestamp + CAPTURE_INTERVAL_SEC

                process_start = time.perf_counter()
                frame_height, frame_width = frame.shape[:2]
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                stage_start = time.perf_counter()
                result = detector.predict(frame, device=device, verbose=False)[0]
                yolo_time = time.perf_counter() - stage_start

                stage_start = time.perf_counter()
                depth_map = get_depth_map(
                    img_rgb,
                    midas_model,
                    midas_transform,
                    device,
                )
                midas_time = time.perf_counter() - stage_start

                stage_start = time.perf_counter()
                depth_min = float(depth_map.min())
                depth_max = float(depth_map.max())
                boxes = result.boxes
                detection_count = 0 if boxes is None else len(boxes)
                processed: list[dict[str, Any]] = []
                annotated = frame.copy()

                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        xyxy = [float(value) for value in box.xyxy[0].tolist()]
                        class_name = detector.names[class_id]
                        raw_depth = sample_box_depth(depth_map, xyxy)
                        if raw_depth is None or not np.isfinite(raw_depth):
                            raise ValueError(
                                f"Could not obtain valid MiDaS depth for "
                                f"{class_name} in frame {frame_idx}"
                            )

                        tagged = score_and_tag_detection(
                            class_name,
                            confidence,
                            xyxy,
                            raw_depth,
                            depth_min,
                            depth_max,
                            frame_width,
                            frame_height,
                        )
                        processed.append(tagged)
                        x1, y1, x2, y2 = [int(value) for value in xyxy]
                        x1 = min(max(x1, 0), frame_width - 1)
                        y1 = min(max(y1, 0), frame_height - 1)
                        x2 = min(max(x2, 0), frame_width - 1)
                        y2 = min(max(y2, 0), frame_height - 1)
                        label = (
                            f"{class_name} {confidence:.2f} "
                            f"risk={tagged['risk']:.2f} {tagged['zone']}"
                        )
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            annotated,
                            label,
                            (x1, max(y1 - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 0),
                            2,
                        )

                raw_alert = decide_alert(processed)
                stable_alert = stabilizer.update(raw_alert, float(timestamp))
                fusion_time = time.perf_counter() - stage_start

                tts_time = 0.0
                tts_file = ""
                if stable_alert is not None:
                    tts_stage_start = time.perf_counter()
                    tts_engine.speak(stable_alert["message"])
                    tts_time = time.perf_counter() - tts_stage_start

                if stable_alert is not None:
                    cv2.putText(
                        annotated,
                        f"ALERT: {stable_alert['message']}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                    )

                annotated_path = annotated_dir / f"frame_{len(rows):05d}.jpg"
                if not cv2.imwrite(str(annotated_path), annotated):
                    raise OSError(f"Could not save annotated frame: {annotated_path}")
                total_time = time.perf_counter() - process_start
                actual_fps = 1.0 / total_time if total_time > 0 else 0.0

                row = {
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                    "n_detections": detection_count,
                    "n_valid_detections": len(processed),
                    "raw_alert": json.dumps(raw_alert, ensure_ascii=False),
                    "stable_alert": json.dumps(stable_alert, ensure_ascii=False),
                    "yolo_time_s": yolo_time,
                    "midas_time_s": midas_time,
                    "fusion_zone_decision_time_s": fusion_time,
                    "tts_time_s": tts_time,
                    "total_time_s": total_time,
                    "actual_fps": actual_fps,
                    "tts_file": tts_file,
                    "device": device,
                    "input_source": str(source),
                    "processed_detections": json.dumps(
                        [
                            {
                                "class_name": detection["class_name"],
                                "confidence": detection["confidence"],
                                "norm_depth": detection["norm_depth"],
                                "risk": detection["risk"],
                                "zone": detection["zone"],
                                "in_roi": detection["in_roi"],
                            }
                            for detection in processed
                        ],
                        ensure_ascii=False,
                    ),
                }
                writer.writerow(row)
                log_file.flush()
                rows.append(row)
                print(
                    f"frame={frame_idx} t={timestamp:.2f}s "
                    f"detections={detection_count} "
                    f"raw_alert={raw_alert is not None} "
                    f"stable_alert={stable_alert is not None} "
                    f"total={total_time:.3f}s fps={actual_fps:.2f}"
                )

        tts_engine.flush()
        if tts_mode == "file":
            generated_files = tts_engine.audio_files
            alert_rows = [
                row
                for row in rows
                if row["stable_alert"] and row["stable_alert"] != "null"
            ]
            if len(generated_files) != len(alert_rows):
                raise RuntimeError("TTS output count does not match stable alert count")
            for generated, row in zip(generated_files, alert_rows):
                row["tts_file"] = str(generated)
            if rows:
                with log_path.open("w", newline="", encoding="utf-8") as log_file:
                    writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
    finally:
        capture.release()
        tts_engine.close()

    return {
        "source": source,
        "device": device,
        "video_fps": fps,
        "frames_read": frames_read,
        "frames_processed": len(rows),
        "detections": sum(int(row["n_detections"]) for row in rows),
        "stable_alerts": sum(bool(row["stable_alert"]) and row["stable_alert"] != "null" for row in rows),
        "tts_files": [str(path) for path in tts_engine.audio_files],
        "log_path": str(log_path),
        "annotated_dir": str(annotated_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=os.environ.get("VIDEO_SOURCE", str(PROJECT_ROOT / "results" / "sprint12_test_input.mp4")),
        help="Video path or camera index; set VIDEO_SOURCE=0 for webcam input.",
    )
    parser.add_argument("--tts", choices=("file", "play"), default="file")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    summary = run_pipeline(
        args.source,
        tts_mode=args.tts,
        device=args.device,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
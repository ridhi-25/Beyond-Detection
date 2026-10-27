# Beyond-Detection

Short description

Beyond-Detection is a computer-vision research project focusing on fusing YOLOv8 object detection with MiDaS depth estimation for road-scene analysis (RSUD20K dataset). The project includes preprocessing, fusion, evaluation, and inference components. Heavy model training will be performed on GPU-enabled environments (e.g., Kaggle).

Current Sprint

Sprint 13 — FWB architecture and optimization experiment (prepared; blocked before training)

Sprint 1 — Environment Setup (completed; setup details retained below)

Local machine limitations

- 8 GB RAM
- No NVIDIA CUDA GPU available locally
- Local machine is for development, preprocessing, inference/testing and analysis
- Heavy training will be performed on Kaggle (GPU)

Python

- Python 3.12.x (system Python used to create project-local venv)

Environment setup (Windows PowerShell)

1. Create the project-local virtual environment (already done in this sprint):
   python -m venv .venv

2. Activate the virtual environment in PowerShell:
   .\.venv\Scripts\Activate.ps1

3. Install dependencies (from requirements.txt):
   python -m pip install -r requirements.txt

Verification

- To verify Python inside venv:
  python --version

- To verify PyTorch and CUDA availability in venv:
  python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

Project structure

C:\beyond-detection\
├── .venv\
├── data\
│   ├── raw\
│   ├── processed\
│   └── samples\
├── models\
├── preprocessing\
├── training\
├── fusion\
├── evaluation\
├── inference\
├── utils\
├── results\
│   ├── figures\
│   ├── metrics\
│   └── predictions\
├── checkpoints\
├── scripts\
├── configs\
├── tests\
├── requirements.txt
├── .gitignore
└── README.md

Sprint 1 note: RSUD20K was intentionally not included during environment setup. It was downloaded and verified in Sprint 2 as described below.

Sprint 2 — Dataset Verification (summary)

- RSUD20K source: `hasibzunair/rsud20k-bangladesh-road-scene-understanding`.
- Downloaded archive: `data/raw/rsud20k_download/verified_download/rsud20k-bangladesh-road-scene-understanding.zip` (10.97 GB); extracted dataset root: `data/raw/rsud20k_download/verified_download/extracted/rsud20k` (10.99 GB).
- The dataset contains 20,334 JPG images and 20,334 YOLO TXT labels: train 18,681, validation 1,004, test 649. All images match labels; no missing, orphan, or empty labels were found.
- `classes.txt` defines 13 classes, IDs 0–12, in the methodology's expected order. All 130,000 annotation rows have valid five-field YOLO syntax and normalized coordinates.
- Verification found boxes extending outside normalized image boundaries in 3,399 label files. No annotations were changed; decide how to handle these before later label preprocessing.
- The original Wang/Baidu dataset (ID 198589) was not found locally. No substitute was downloaded; substitute selection requires explicit approval.
- The verifier is `preprocessing/verify_dataset.py`; the detailed report is `results/dataset_verification.md`.

Sprint 3 — Basic Pretrained YOLOv8n Inference

- Ran the untouched pretrained COCO YOLOv8n model (`yolov8n.pt`) on CPU with PyTorch 2.14.0+cpu and Ultralytics 8.4.160. CUDA is unavailable as expected.
- Tested one RSUD20K train image (`train0.jpg`) and two validation images (`val0.jpg`, `val1.jpg`). All were 1920 × 1080; inference succeeded with 8, 8, and 7 detections, respectively.
- Annotated images: `results/sprint3/image_1_annotated.jpg`, `results/sprint3/image_2_annotated.jpg`, and `results/sprint3/image_3_annotated.jpg`.
- The official `yolov8n.pt` COCO weights were downloaded from the Ultralytics v8.4.0 release. No custom, baseline, or merged-model weights were used; no training or dataset modifications occurred.
- This was a pipeline check only, not an accuracy evaluation. COCO does not include all RSUD20K-specific classes.
- Full detections and confidences: `results/sprint3_pretrained_report.md`.

Sprint 4 — Dataset Merge and B1 Baseline Preparation

- Inspected the supplied `datas.zip` and `B1_baseline_50epochs.zip` without altering either original archive. The road-obstacle archive contains `Annotation/train.json`, `Annotation/eval.json`, and 6,276 JPGs under `is/`; the JSONs contain COCO `images`, `annotations`, and `categories`.
- Baidu source classes include `pothole` and `trafficcone` (with category IDs reordered between train and eval). Conversion selects by exact category name, ignores Baidu `person` and `car`, assigns pothole to class 13 and traffic cone to class 14, and skips invalid/out-of-bounds target boxes rather than modifying them.
- Created `data/merged/classes.txt` with the fixed 15-class taxonomy. RSUD20K IDs 0–12 and original source files were preserved; merged copies are separated and prefixed.
- Generated `data/merged/images/`, `data/merged/labels/`, `data/merged/manifest.csv`, and deterministic split lists under `data/splits/`. RSUD20K allocations remain train/val/test; Baidu train-source items use a seeded 90/10 train/validation split and Baidu eval-source items are assigned to test.
- Merged counts: 21,223 images/labels and 132,733 annotation rows; train 19,390, validation 1,083, test 750. Detailed verification is in `results/sprint4_dataset_report.md`.
- Prepared `models/detector/data.yaml` and `training/train_detector.py` for plain YOLOv8n B1 (50 epochs, 640 image size, batch 16, SGD, lr0 0.01). Ultralytics validated the dataset configuration. No local training was run because this machine is CPU-only with approximately 8 GB RAM.
- The previous B1 archive was imported read-only to `results/previous_b1/`; its checkpoint/configuration and 50-row metrics CSV are documented in `results/sprint4_previous_b1_report.md`. Its recorded epoch-50 mAP50 is 0.90954 and mAP50–95 is 0.72882. The historical data files were not in that archive, so exact dataset-content equivalence is unknown. The historical row is marked `previous_verified_B1` in `results/experiment_log.csv`.
- Baidu verification and conversion reports: `results/sprint4_baidu_verification.md`, `results/sprint4_baidu_conversion_report.md`; Sprint 4 completion summary: `results/sprint4_completion_report.md`.
- The B1 artifact passed the final audit and was ready for the held-out Sprint 5 test evaluation.

### B1 baseline final audit

- The original 50-epoch B1 baseline was completed previously. Its archived checkpoint/results were imported into `results/previous_b1/` and audited against the archived `results.csv`, `args.yaml`, checkpoint metadata, and the current 15-class taxonomy.
- The audit verified all 50 consecutive epochs, finite metrics, loadable best/last checkpoints, YOLOv8n, and the intended 15 class names/configuration. Epoch-50 metrics: mAP50 0.90954, mAP50–95 0.72882, precision 0.88449, recall 0.83755.
- No retraining or inference was performed during the audit. Historical dataset byte-for-byte identity with the current reconstruction is not claimed; the user confirms the previous run used the merged RSUD20K + Baidu dataset.
- Status: **B1 VERIFIED — READY FOR SPRINT 5** at the end of the Sprint 4 audit.
- Full verification details: `results/sprint4_b1_final_audit.md`.

### Sprint 5 — B1 held-out test evaluation

- Evaluated the audited `results/previous_b1/weights/best.pt` checkpoint on the held-out `test` split only (750 images, 4,132 annotations), using CPU inference at 640 px. No training or test-split edits were performed.
- Ultralytics test metrics: precision 0.900697, recall 0.863312, mAP50 0.928556, and mAP50–95 0.776626. COCOeval metrics and object-size breakdown are documented separately; 67 zero-area raw detections were excluded only from the derived COCOeval input, while the unmodified raw predictions were retained.
- The full report, per-class metrics, test distribution, predictions, and plots are in `results/b1_baseline_eval/`; the Sprint 5 report is `results/sprint5_baseline_evaluation_report.md`.
- This was an evaluation-only sprint; no training was performed.

### Sprint 6 — MiDaS-small standalone depth estimation

- Ran the pretrained MiDaS-small model through `torch.hub.load()` on the same RSUD20K train image used for Sprint 3 (`train0.jpg`), using CPU inference and the model's `small_transform`.
- Saved the original/depth comparison to `results/sprint6_depth_map.png`; the 1080 × 1920 depth map contains finite, spatially varying relative inverse-depth values.
- MiDaS output is relative inverse depth, not distance in metres. No training, fine-tuning, YOLO, fusion, or dataset changes were performed.
- `timm==1.0.30` is installed in the project virtual environment and pinned in `requirements.txt`. Details: `results/sprint6_depth_report.md`.
- At the completion of Sprint 6, Sprint 7 had not yet started.

### Sprint 7 — YOLOv8n + MiDaS-small combined inference

- Ran the verified B1 checkpoint `results/previous_b1/weights/best.pt` and pretrained MiDaS-small on the same `train0.jpg` frame used in Sprints 3 and 6.
- Sampled the simple mean relative inverse-depth value inside each YOLO detection box and normalized it to 0–1 for display. Seven detections were produced and all received valid depth samples.
- The annotated output is `results/sprint7_yolo_midas_combined.jpg`; per-detection values and verification details are in `results/sprint7_yolo_midas_report.md`.
- This is component association only: no fusion formula, risk score, metric distance, or training was performed. No dataset files were modified.
- Sprint 8 has not been started.

### Sprint 8 — Risk fusion module

- Added the pure-Python score `risk = tier_weight × confidence × normalized_depth` with weights `critical=1.0`, `high=0.6`, and `low=0.3`; unknown classes default conservatively to `high`.
- The 15-class tier mapping is a first-pass methodology design choice, not an empirically validated result.
- All 7 standalone tests passed. The optional Sprint 7 integration was performed and prints tier/risk per detection only; it does not filter detections, generate alerts, or alter YOLO/MiDaS depth calculations.
- No training, dataset modification, metric distance estimation, or Sprint 9 work occurred. Details and measured integration outputs: `results/sprint8_risk_fusion_report.md`.

### Sprint 9 — Left/center/right zones and bottom-third ROI

- Added deterministic spatial rules based on each detection's bounding-box center in original-image pixel coordinates: horizontal zones are left, center, and right; the ahead ROI is the bottom third of the frame.
- Boundary conventions: x = frame width / 3 belongs to center; x = 2 × frame width / 3 belongs to right; y = 2 × frame height / 3 is inside the ROI.
- All 8 standalone tests passed, including exact horizontal and vertical boundary checks.
- Added print-only zone/ROI reporting to Sprint 7 combined inference. On the established 1920 × 1080 sample, 7 detections were processed: 3 left/in-ROI and 4 center/outside-ROI; no right-zone detection appeared in this sample.
- No detection, depth, or risk calculations were changed; no training, dataset modification, metric-distance estimation, or alert/decision logic was added. Sprint 10 has not been started. Details: `results/sprint9_zone_roi_report.md`.

### Sprint 10 — Decision logic and alert threshold

- Added a pure decision layer that filters for center-zone + bottom-third-ROI detections, selects the highest-risk ahead hazard, stays silent if none qualifies, and otherwise produces a text alert.
- `ALERT_THRESHOLD = 0.35` is a tunable starting configuration, not empirically validated, learned, statistically optimized, or claimed optimal. Risk equal to the threshold triggers an alert.
- Urgency bands are caution (<0.5), warning (0.5–<0.75), and danger (>=0.75). Direction is rule-based on the maximum detected risks in the left/right zones; if both are at least 0.35 it returns stop, otherwise it chooses the lower-risk side (ties choose step right).
- All 9 decision tests passed, along with the Sprint 8 (7/7) and Sprint 9 (8/8) regression suites. Manual threshold checks passed.
- Integrated the decision layer after collecting the full frame's risk-scored and spatially tagged detections. Three existing RSUD20K images ran successfully: two were silent; `train1.jpg` produced `pickup truck, step left, caution`. These are functional checks, not empirical validation.
- No training, dataset modification, TTS, temporal smoothing, cooldown, or Sprint 11 work occurred. Details: `results/sprint10_decision_report.md`.

### Sprint 11 — Temporal smoothing / cooldown

- Added a stateful `AlertStabilizer` wrapper for Sprint 10 alert outputs; Sprint 10 `decision.py` remains unchanged.
- Defaults: five-frame history, three matching class+direction alert decisions within that window, and a two-second cooldown for the exact same message. These are starting design parameters, not empirically optimized; matching frames need not be consecutive, and `None` frames remain in history.
- Different messages can bypass a previous message's cooldown. No risk averaging, urgency override, object tracking, or re-identification is implemented.
- All 12 deterministic temporal tests passed; Sprint 8/9/10 regressions passed 7/7, 8/8, and 9/9.
- No webcam, camera loop, TTS/audio, or live-stream testing was performed. Sprint 12 has not been started. Details: `results/sprint11_temporal_smoothing_report.md`.

### Alert Validation Experiment — 2026-09-28 (descriptive check; not a new sprint)

- Ran the existing Sprint 8–10 pipeline on 30 deterministically selected, evenly spaced images from the local RSUD20K test split using CPU, the verified B1 checkpoint `results/previous_b1/weights/best.pt`, and MiDaS-small.
- The run produced 183 detections, 23 ahead detections, 3 caution alerts, and 27 silent images. All selected source-image hashes and the B1 checkpoint hash were unchanged; no implementation issues were automatically detected.
- All 29 automatically flagged frames and all three alert frames received manual visual review. One possible perception/free-space concern was noted: `test0.jpg` recommends step right although a large right-side vehicle appears unboxed. This is not an identified decision-code bug or a safety determination.
- The experiment is descriptive, not accuracy or safety validation; it does not validate metric distance, establish 0.35 as optimal, or establish safe direction recommendations. No threshold/methodology change, training, dataset modification, or human-subject testing occurred.
- Regression suites passed: Sprint 8 7/7, Sprint 9 8/8, Sprint 10 9/9. Report, CSVs, metrics, review notes, reproducibility script, and annotated outputs are under `results/alert_validation/`; see `results/alert_validation/alert_validation_summary.md`.

### Sprint 12 — TTS output and full pipeline integration

- Added `inference/live_pipeline.py` to connect the frozen B1 checkpoint, MiDaS-small, depth sampling, existing risk fusion and zone/ROI mapping, Sprint 10 decisions, and Sprint 11 alert stabilization. The existing Sprints 8–11 implementations and methodology were reused without modification.
- Added `inference/tts_output.py` with background-thread `TTSEngine` playback and non-playing `FileTTSEngine` WAV synthesis. TTS is enqueued asynchronously; the recorded-video validation does not require a speaker or webcam.
- Validated on the existing 15-frame, 2 FPS recording `results/sprint12_test_input.mp4` on CPU with `CAPTURE_INTERVAL_SEC = 0.5` and the existing 5-frame / 3-consistent / 2-second stabilizer settings. All 15 frames were processed, yielding 95 detections and 3 stabilized alerts with three WAV files. Average processing throughput was approximately 4.65 FPS; this is an integration measurement on this clip, not a performance or safety claim.
- Per-frame YOLO, MiDaS, fusion/zone/decision/smoothing, TTS enqueue, total processing time, measured FPS, detections, alerts, and generated audio paths are recorded in `results/sprint12_frame_log.csv`. Annotated frames are in `results/sprint12_annotated/`; the full integration details are in `results/sprint12_integration_report.md`.
- Use `python inference/live_pipeline.py --source results/sprint12_test_input.mp4 --tts file` for recorded-video/file-TTS validation. For a future local webcam run, set `$env:VIDEO_SOURCE = "0"` and use `--tts play`; the webcam and audible playback were not tested in Sprint 12.
- No model training, dataset/label modification, or methodology changes were made. MiDaS relative depth is not metric distance, and alerts are not validated safety or threat determinations. Sprint 13 work was not part of Sprint 12.

### Sprint 13 — FWB optimization E1 (prepared; blocked)

- Implemented the Feature Weighting Block, a project-local Ultralytics 8.4.160 parser adapter, a YOLOv8n+FWB YAML, architecture/training/evaluation/comparison workflows, and independent module tests. FWB is configured immediately after SPPF; later absolute layer references are shifted and the neck receives the reweighted P5 feature.
- The independent FWB tests passed 6/6, pretrained layer-remap tests passed 3/3, and Sprint 8–12 regressions/TTS tests passed 39/39. Full 640×640 detector architecture validation was **blocked before construction** because 0.58–0.97 GiB RAM was available (3.00 GiB gate); CUDA is unavailable.
- No pretrained transfer, model training, or FWB evaluation occurred. The 50-epoch CPU run was not attempted. The B1 reference is still the Sprint 5 held-out test result; its checkpoint and dataset were not modified.
- B1-only values in `results/sprint13_fwb_comparison.csv` are recorded reference metrics; FWB and delta fields remain blank/pending, not fabricated. Full status and hardware requirements: `results/sprint13_fwb_report.md`; architecture gate: `results/sprint13_architecture_report.md`.
- Next action: use the reproducible [Kaggle GPU guide](./results/sprint13_gpu_execution_guide.md), validate the full architecture and pretrained transfer, then run the prepared 50-epoch experiment. The 3 GiB threshold is retained as a conservative guard, not claimed as an empirically measured minimum. Sprint 14 / INT8 has not started.

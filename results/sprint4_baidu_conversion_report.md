# Sprint 4 — Baidu COCO Conversion Report

## Source

- Source directory: `C:\beyond-detection\data\raw\baidu_road_obstacle`
- Converted output: `C:\beyond-detection\data\merged`
- Source image records: 6276
- Source annotation records: 78969
- Source category annotation counts: {'person': 56915, 'car': 19159, 'trafficcone': 1176, 'pothole': 1719}
- Source category names (exact as found): {'person': 'person', 'car': 'car', 'trafficcone': 'trafficcone', 'pothole': 'pothole'}

## Conversion

- Retained source classes: exact category names `pothole` and `trafficcone` (also accepts `traffic cone` if present).
- Mapping: `pothole` → 13; `trafficcone` / `traffic cone` → 14.
- COCO `person` and `car` annotations were ignored.
- Source pothole annotations: 1719
- Source traffic-cone annotations: 1176
- Valid retained pothole annotations converted: 1582
- Valid retained traffic-cone annotations converted: 1151
- Converted images with at least one valid retained annotation: 889
- Skipped images with no valid retained annotation: 5387
- Missing or unreadable source images: 0
- Invalid target annotations skipped: 162

## Per-file source summary

### `eval.json`

- Image records: 570
- Annotation records: 6717
- Categories: {1: 'person', 2: 'car', 3: 'trafficcone', 4: 'pothole'}
- Valid retained annotations converted: {'trafficcone': 138, 'pothole': 189}
- Invalid annotations skipped: 25
- Converted images: 101
- Skipped images: 469
- Missing or unreadable images: 0

### `train.json`

- Image records: 5706
- Annotation records: 72252
- Categories: {1: 'car', 2: 'person', 3: 'trafficcone', 4: 'pothole'}
- Valid retained annotations converted: {'trafficcone': 1013, 'pothole': 1393}
- Invalid annotations skipped: 137
- Converted images: 788
- Skipped images: 4918
- Missing or unreadable images: 0

Only copied images and generated YOLO labels were written. The original archive and extracted COCO data were not modified.

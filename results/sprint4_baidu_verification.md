# Sprint 4 — Baidu COCO Dataset Verification

## Source artifact

- User-provided archive: `C:\Users\ridhi\Downloads\datas.zip`
- Archive size: 532,365,807 bytes.
- ZIP integrity: verified; 6,280 entries, 549,885,668 uncompressed bytes.
- Original ZIP retained unchanged.
- Extracted workspace copy: `data/raw/baidu_road_obstacle/`
- Actual archive structure:

```text
baidu_road_obstacle/
├── Annotation/
│   ├── eval.json
│   └── train.json
└── is/
    ├── 0.jpg
    ├── 1.jpg
    └── ... (6,276 JPG files total)
```

This archive contains the road-obstacle data represented in the brief: COCO annotations include `pothole` and `trafficcone`. It also includes `person` and `car`; those categories are not imported into the unified classes. No other dataset directory was present in the archive.

## COCO JSON verification

Both annotation files are valid JSON objects with keys `images`, `type`, `annotations`, and `categories`. Both use COCO-style image records (`id`, `file_name`, `width`, `height`) and annotation records (`segmentation`, `area`, `iscrowd`, `ignore`, `image_id`, `bbox`, `category_id`, `id`). All referenced image filenames were present in `is/`; train and eval filenames do not overlap.

| File | Images | Annotations | Category IDs and exact names |
|---|---:|---:|---|
| `Annotation/train.json` | 5,706 | 72,252 | 1 car; 2 person; 3 trafficcone; 4 pothole |
| `Annotation/eval.json` | 570 | 6,717 | 1 person; 2 car; 3 trafficcone; 4 pothole |
| **Total** | **6,276** | **78,969** |  |

The category ID order differs between JSON files. Conversion therefore selects categories by exact name, not by category ID.

| Category name | Train annotations | Eval annotations | Total |
|---|---:|---:|---:|
| car | 17,556 | 1,603 | 19,159 |
| person | 52,153 | 4,762 | 56,915 |
| trafficcone | 1,032 | 144 | 1,176 |
| pothole | 1,511 | 208 | 1,719 |
| **Total** | **72,252** | **6,717** | **78,969** |

Target categories occur in 815 train-source images and 105 eval-source images. Sample verified image record: `{"id": 1, "file_name": "0.jpg", "width": 1920, "height": 1080}` from train. Sample verified annotation: image ID 1, category ID 1, bbox `[564, 375, 35, 64]` (train category ID 1 is `car`). Eval sample image `1064.jpg` is 960×540.

### Bounding-box observations

COCO boxes were checked against their associated JSON image dimensions. Some source boxes are outside image bounds or have invalid size: 2,181 rows in train and 208 in eval over all categories. In the target categories, 162 invalid boxes were excluded from conversion (137 train-source, 25 eval-source). No source JSON or image was modified, and boxes were not clamped.

## Conclusion

The user-provided archive contains the intended road-obstacle annotation set with useful `pothole` and `trafficcone` categories. The full archive was safely extracted without altering the original ZIP. Only valid target-category boxes were selected for conversion; person and car rows were excluded from Baidu outputs.

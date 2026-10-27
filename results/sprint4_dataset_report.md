# Sprint 4 — Merged Dataset Report

## Dataset counts

- RSUD20K images/labels copied: 20,334
- Valid Baidu images/labels converted: 889
- Merged images: 21,223
- Merged label files: 21,223
- Merged annotation rows: 132,733

## Split counts

| Split | Total images | RSUD20K | Baidu |
|---|---:|---:|---:|
| Train | 19,390 | 18,681 | 709 |
| Validation | 1,083 | 1,004 | 79 |
| Test | 750 | 649 | 101 |

- Baidu train-source assignment: deterministic shuffle (seed 42); 90% train / 10% validation.
- Baidu eval-source images are assigned to test; RSUD20K's original split allocation is preserved.

## Class distribution (annotation rows)

| ID | Class | Rows |
|---:|---|---:|
| 0 | person | 35,645 |
| 1 | rickshaw | 25,410 |
| 2 | rickshaw van | 3,050 |
| 3 | auto rickshaw | 13,327 |
| 4 | truck | 1,600 |
| 5 | pickup truck | 1,174 |
| 6 | private car | 20,080 |
| 7 | motorcycle | 16,170 |
| 8 | bicycle | 2,983 |
| 9 | bus | 6,029 |
| 10 | micro bus | 3,323 |
| 11 | covered van | 692 |
| 12 | human hauler | 517 |
| 13 | pothole | 1,582 |
| 14 | traffic cone | 1,151 |

## Validation

- Missing image files: 0
- Images without manifest entries: 0
- Orphan label files: 0
- Duplicate output filenames: 0
- Malformed label rows/files: 0
- Invalid class IDs: {}
- Invalid YOLO coordinates: {}
- Source boundary-overrun incidences retained unchanged: {'horizontal': 2648, 'vertical': 1153}
- Split overlap: []
- Split list sample images opened successfully: ['C:\\beyond-detection\\data\\merged\\images\\baidu_5500.jpg', 'C:\\beyond-detection\\data\\merged\\images\\baidu_5510.jpg', 'C:\\beyond-detection\\data\\merged\\images\\baidu_5505.jpg']
- Annotation IDs 0–12 from RSUD20K were preserved; source RSUD20K images and labels remain unchanged.
- Only Baidu pothole and traffic-cone annotations were converted; source person/car categories were ignored.
- Invalid Baidu boxes were skipped, not clamped or written.

## Baidu conversion detail

- Source image records: 6,276
- Source annotation records: 78,969
- Converted target annotations: {'trafficcone': 1151, 'pothole': 1582}
- Invalid Baidu annotations skipped: 162
- Converted Baidu images: 889
- Skipped Baidu images: 5387
- Missing/unreadable Baidu images: 0

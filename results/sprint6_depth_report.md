# Sprint 6 — MiDaS-small Standalone Depth Estimation

## Model

MiDaS-small, pretrained and loaded through `torch.hub.load("intel-isl/MiDaS", "MiDaS_small")`, using the corresponding `transforms.small_transform`.

## Input image

`C:\beyond-detection\data\raw\rsud20k_download\verified_download\extracted\rsud20k\images\train\train0.jpg` — the primary image used in Sprint 3.

## Device

CPU (`torch.cuda.is_available()` returned `False`).

## Input dimensions

Width × height: **1920 × 1080**.

## Output depth dimensions

Height × width: **1080 × 1920**.

## Depth value range

- Minimum: **-11.01**
- Maximum: **366.29**
- All depth values were finite.

## Model interpretation

MiDaS-small produces relative inverse depth. Higher values generally indicate closer regions; lower values generally indicate farther regions. Values are model-relative and do not represent metres or other metric distance units.

## Output

`results/sprint6_depth_map.png` — side-by-side original RGB image and MiDaS-small depth visualization using the requested `inferno` color map.

## Verification

- Model loaded successfully from the official MiDaS Torch Hub repository.
- MiDaS-small's `small_transform` was used.
- Inference completed on CPU.
- The prediction was resized to the original RGB image dimensions.
- The 1080 × 1920 depth array had finite, non-constant values.
- The PNG was opened and decoded successfully; visual inspection confirmed the original image and non-uniform depth map are present. Foreground/near road regions are generally brighter than the distant central scene, consistent with relative inverse-depth interpretation.

## Important limitation

MiDaS-small output at this stage is **relative inverse depth** and is **not metric distance in metres**. No object-distance estimation was performed.

## Training

No training or fine-tuning was performed.

## Dataset

No dataset images, labels, or split files were modified.

## Other scope checks

No YOLO, depth fusion, risk fusion, tracking, threat detection, KITTI data, or Sprint 7 work was used or added.

## Environment

- Python: 3.12.3
- PyTorch: 2.14.0+cpu
- `timm`: 1.0.30
- `timm==1.0.30` is pinned in `requirements.txt`.

PyTorch Hub prompted for repository trust in the initial non-interactive attempt. The run script now explicitly trusts the requested MiDaS repository and preauthorizes its EfficientNet encoder repository, which MiDaS-small loads internally. Model source code and pretrained weights were downloaded to the user-level Torch Hub cache, not into the dataset. The legacy upstream model emitted `torch.jit.script` and `timm.models.layers` deprecation warnings; inference completed successfully.

## Sprint 6 status

**Complete.** Standalone pretrained MiDaS-small CPU inference and output verification succeeded. Sprint 7 has not been started.

"""
Verify local environment for Beyond-Detection (Sprint 1).
This script prints versions and availability of key packages and does not download models or large files.
"""
import platform
import os

print("--- Environment verification for Beyond-Detection ---")
print(f"Python: {platform.python_version()} ({platform.system()} {platform.release()})")
print(f"Current working directory: {os.getcwd()}")

# Torch
try:
    import torch
    print(f"torch: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA available: {cuda_avail}")
    if cuda_avail:
        try:
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"Could not get GPU name: {e}")
except Exception as e:
    print(f"torch import error: {e}")

# Ultralytics (YOLOv8)
try:
    import ultralytics
    print(f"ultralytics: {ultralytics.__version__}")
    # Import YOLO class but do not instantiate models or download weights
    try:
        from ultralytics import YOLO  # noqa: F401
        print("YOLO class import: OK")
    except Exception as e:
        print(f"YOLO class import error: {e}")
except Exception as e:
    print(f"ultralytics import error: {e}")

# OpenCV
try:
    import cv2
    print(f"opencv-python: {cv2.__version__}")
except Exception as e:
    print(f"cv2 import error: {e}")

# NumPy
try:
    import numpy as np
    print(f"numpy: {np.__version__}")
except Exception as e:
    print(f"numpy import error: {e}")

# Pandas
try:
    import pandas as pd
    print(f"pandas: {pd.__version__}")
except Exception as e:
    print(f"pandas import error: {e}")

# Matplotlib
try:
    import matplotlib
    print(f"matplotlib: {matplotlib.__version__}")
except Exception as e:
    print(f"matplotlib import error: {e}")

# PIL / Pillow
try:
    from PIL import Image
    import PIL
    print(f"Pillow: {PIL.__version__}")
except Exception as e:
    print(f"Pillow import error: {e}")

# PyYAML
try:
    import yaml
    print(f"PyYAML: {yaml.__version__}")
except Exception as e:
    print(f"yaml import error: {e}")

# SciPy
try:
    import scipy
    print(f"scipy: {scipy.__version__}")
except Exception as e:
    print(f"scipy import error: {e}")

print("--- End of verification ---")

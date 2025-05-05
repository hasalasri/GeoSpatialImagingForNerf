# IMU‑Assisted Structure‑from‑Motion  
`sfm_mobile.py`

---

## Overview
A lightweight Python pipeline that reconstructs a **sparse 3‑D point cloud** and camera poses from a set of overlapping photographs *optionally* enhanced with per‑image IMU orientation/position data.  
The code produces COLMAP‑compatible `cameras.txt`, `images.txt`, and `points3D.txt` files so you can continue with dense reconstruction or NeRF training.

---

## 🛠️ Requirements
| Package | Tested Version |
|---------|---------------|
| Python  | ≥ 3.8 |
| NumPy   | 1.26 |
| SciPy   | 1.12 |
| OpenCV (main modules + contrib) | 4.10 |
| tqdm *(optional CLI progress)* | 4.66 |


## installation
```bash
# 1) create and activate a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 2) upgrade pip & install core dependencies
python -m pip install --upgrade pip
python -m pip install numpy scipy tqdm

# 3) install OpenCV with contrib modules (for SIFT)
python -m pip install --extra-index-url https://artifacts.opencv.org/opencv-python/ \
                      opencv-contrib-python

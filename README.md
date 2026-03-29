# 🎯 python-realtime-object-tracker

[![Build](https://github.com/danielcruzdev/python-realtime-object-tracker/actions/workflows/build.yml/badge.svg)](https://github.com/danielcruzdev/python-realtime-object-tracker/actions/workflows/build.yml)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)

Real-time multi-object detection and tracking via webcam, combining **YOLO26** (detection) with the **SORT** algorithm (tracking) and Kalman Filter.

---

## 📸 Demo

> Each object receives a **unique and persistent ID**, colored bounding box, class label, centroid, and **trajectory trail** in real time.

```
[Webcam]  →  [YOLO26: detection]  →  [SORT + Kalman: tracking]  →  [OpenCV visualization]
```

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🔍 YOLO26 Detection | Detects 80 classes (people, cars, animals, etc.) |
| 🏷️ SORT Tracking | Unique and persistent IDs per object via Kalman Filter |
| 🌈 Colored IDs | Each track_id has a unique, reproducible color |
| 〰️ Trajectory Trail | Line showing the path traveled by each object |
| 📊 Real-time HUD | FPS, total active tracks, and per-class count |
| ⚙️ Confidence Tuning | `+` / `-` keys change the YOLO threshold in real time |
| 👁️ Display Toggles | Show/hide centroids (`C`) and labels (`L`) |
| 📷 Screenshot | Save current frame as PNG with the `S` key |

---

## 🛠️ Prerequisites

### Python

Recommended version: **Python 3.10 or 3.11**

> Python 3.12+ may have compatibility issues with some dependencies. Use 3.10 or 3.11 to ensure compatibility.

Download at: https://www.python.org/downloads/

During installation on Windows, check the **"Add Python to PATH"** option.

Verify the installation:
```bash
python --version
```

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/danielcruzdev/python-realtime-object-tracker.git
cd python-realtime-object-tracker
```

### 2. (Recommended) Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> On the first run, YOLO26 will automatically download the `yolo26n.pt` model (~5 MB) if it does not exist in the folder.

---

## ▶️ Running the Project

```bash
# Default (yolo26n, camera 0)
python main.py

# Choose model
python main.py --model yolo26s.pt
python main.py --model yolo26m.pt

# Choose camera and model
python main.py --model yolo26s.pt --camera 1

# Set initial confidence
python main.py --model yolo26n.pt --conf 0.6

# Show all options
python main.py --help
```

A window will open with the real-time webcam feed showing detected and tracked objects.

---

## ⌨️ Controls

| Key | Action |
|---|---|
| `Q` or `ESC` | Quit the program |
| `C` | Show / hide centroids |
| `L` | Show / hide labels (class + ID) |
| `+` or `=` | Increase minimum confidence (+5%) |
| `-` | Decrease minimum confidence (-5%) |
| `S` | Save a screenshot of the current frame |

---

## ⚙️ Configuration

Edit the constants at the top of `main.py` to customize the behavior:

```python
CAMERA_INDEX   = 0      # Webcam index (0 = system default)
YOLO_MODEL     = "yolo26n.pt"  # Model: n (fast) → s → m → l → x (accurate)
CONF_THRESHOLD = 0.4    # Initial minimum confidence (0.0 to 1.0)
IOU_THRESHOLD  = 0.45   # IoU for YOLO internal NMS
MAX_AGE        = 5      # Max frames without update before removing a track
MIN_HITS       = 3      # Minimum detections to confirm a new track
SORT_IOU       = 0.3    # Minimum IoU to associate a detection with an existing track
TRAIL_LENGTH   = 40     # Number of positions stored in the trajectory trail
```

### YOLO model selection

| Model | Speed | Accuracy | Recommended for |
|---|---|---|---|
| `yolo26n.pt` | ⚡⚡⚡ | ★★☆ | Webcam / CPU |
| `yolo26s.pt` | ⚡⚡ | ★★★ | Modern CPU / GPU |
| `yolo26m.pt` | ⚡ | ★★★★ | GPU |
| `yolo26l.pt` | 🐢 | ★★★★★ | Dedicated GPU |

---

## 📂 Project Structure

```
python-realtime-object-tracker/
├── tracker/
│   ├── __init__.py   # Exports the Sort class
│   └── sort.py       # SORT algorithm with Kalman Filter
├── models/           # .pt weights (git-ignored, downloaded automatically)
│   └── yolo26n.pt
├── screenshots/      # Screenshots saved with the S key (git-ignored)
├── main.py           # Main loop: capture, detection, tracking, and HUD
├── requirements.txt  # Project dependencies
├── .gitignore
└── README.md
```

---

## 🧠 How It Works

### 1. Detection — YOLO26
Each webcam frame is processed by **YOLO26**, which returns bounding boxes `[x1, y1, x2, y2]` with the class and confidence score for each detected object.

### 2. Tracking — SORT + Kalman Filter
**SORT** (*Simple Online and Realtime Tracking*) receives the YOLO detections and:
- Uses the **Hungarian Algorithm** to associate previous detections with new ones (based on IoU)
- Maintains a **Kalman Filter** per object to predict its position even when not detected
- Assigns **unique and stable IDs** per object across frames

### 3. Visualization — OpenCV
Each rendered track includes:
- Bounding box colored by ID
- Label `ID X | class`
- Centroid (center point)
- Trajectory trail (last N positions)
- HUD with FPS, total active tracks, and per-class count

---

## 📚 References

- [YOLO26 — Ultralytics](https://docs.ultralytics.com/models/yolo26/)
- [SORT — Bewley et al., 2016](https://arxiv.org/abs/1602.00763)
- [FilterPy — Kalman Filter in Python](https://github.com/rlabbe/filterpy)
- [OpenCV](https://opencv.org/)

---

## 📋 System Requirements

- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 12+
- **Python**: 3.10 or 3.11
- **RAM**: minimum 4 GB (8 GB recommended)
- **Webcam**: any OpenCV-compatible camera
- **GPU** *(optional)*: CUDA-compatible for YOLO acceleration

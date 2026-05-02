# Self-Driving Car Perception — CV Project

Multi-task real-time driving perception pipeline using YOLOPv2 (lane + drivable area
+ object detection) and Depth Anything V2 Small (monocular depth), with voice
instructions via Kokoro TTS.

---

## Repository structure

```
sdcar-perception/
├── demo.py                       ← main entry point, run this
├── requirements.txt
├── scripts/
│   └── download_models.py        ← run once to fetch + export all ONNX models
├── src/
│   └── pipeline/
│       ├── preprocessor.py       ← resize + normalize frames
│       ├── detector.py           ← YOLOPv2 ONNX wrapper
│       ├── depth_estimator.py    ← Depth Anything V2 + background thread
│       ├── fusion.py             ← combines outputs into SceneState
│       ├── navigation.py         ← rule engine + roast generator
│       ├── tts_engine.py         ← Kokoro TTS, non-blocking queue
│       └── visualizer.py         ← draws overlays onto the frame
├── models/                       ← ONNX weights land here (gitignored)
└── data/
    └── samples/                  ← put test videos here
```

---

## Setup (one time)

```bash
# 1. Clone
git clone https://github.com/abdullah-mansoor3/self-driving-car-perception
cd sdcar-perception

# 2. Install python 3.11
sudo apt update
sudo apt install python3.11 python3.11-venv


# 3. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download + export + quantize all models  (~15 min, ~1.5 GB download)
python scripts/download_models.py
```

---

## Run the demo

```bash
# Webcam
python demo.py --source 0

# Video file
python demo.py --source data/samples/dashcam.mp4

# Save output to file
python demo.py --source 0 --save output.avi

# Silent mode (no TTS)
python demo.py --source 0 --no-tts
```

Press `q` in the window to quit.

---

## What you see on screen

| Overlay | Meaning |
|---|---|
| Green tint | Drivable area detected by YOLOPv2 |
| Teal lines | Lane lines detected by YOLOPv2 |
| Yellow box | Detected obstacle (not in your lane) |
| Red box | Detected obstacle **in your lane** |
| Depth thumbnail | Top-right corner, magma colormap — bright = close |
| Arrow at bottom | Lane deviation direction and magnitude |
| Banner at bottom | Current navigation event (orange = warn, red = critical) |

---

## Expected performance on Core i5 / 8 GB RAM

| Component | Per-frame cost |
|---|---|
| YOLOPv2 INT8 (640×384) | ~90–120 ms |
| Depth Anything V2 (every 3rd frame, amortized) | ~50 ms |
| Fusion + navigation | ~3 ms |
| **Total** | **~100–130 ms → 7–10 FPS** |

TTS runs in a background thread — zero impact on frame rate.

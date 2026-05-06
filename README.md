# Self-Driving Car Perception — CV Project

Multi-task real-time driving perception pipeline using YOLOPv2 (lane + drivable area
+ object detection) and Depth Anything V2 Small (monocular depth), with voice
instructions via Kokoro TTS.

---

## Repository structure

```
sdcar-perception/
├── demo.py                        ← main entry point, run this
├── requirements.txt
├── scripts/
│   └── download_models.py         ← run once to fetch + export all ONNX models
├── src/
│   └── pipeline/
│       ├── preprocessor.py        ← resize + normalize frames
│       ├── detector.py            ← YOLOPv2 ONNX wrapper
│       ├── depth_estimator.py     ← Depth Anything V2 + background thread
│       ├── fusion.py              ← combines outputs into SceneState
│       ├── navigation.py          ← rule engine + roast generator
│       ├── tts_engine.py          ← Kokoro TTS, non-blocking queue
│       └── visualizer.py          ← draws overlays onto the frame
├── models/                        ← ONNX weights land here (gitignored)
└── data/
    └── samples/                   ← put test videos here
```

---

## Setup (one time)

```bash
# 1. Clone
git clone https://github.com/abdullah-mansoor3/self-driving-car-perception
cd sdcar-perception

# 2. Install Python 3.11
sudo apt update
sudo apt install python3.11 python3.11-venv
sudo apt install portaudio19-dev python3-dev

# 3. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download + export + quantize all models  (~15 min, ~200 MB download)
python scripts/download_models.py

```

### What the download script does

| Step | Action |
|---|---|
| 1 | Downloads YOLOPv2 as a **pre-exported ONNX** directly from GitHub releases — no `.pt` file, no PyTorch export needed |
| 2 | Downloads Depth Anything V2 Small `.pth` checkpoint from HuggingFace and exports it to ONNX using the legacy `torch.onnx` exporter |
| 3 | INT8-quantizes both ONNX models (~30% speed gain on CPU) |

> **Note — why not the official YOLOPv2 `.pt` weights?**
> The CAIC-AD Google Drive link in the official repo is dead (returns 404).
> We use a community ONNX export (Kazuhito00, MIT licence) instead.
> Input size for this ONNX is **640×640** (square).

> **Note — why 392×630 for Depth Anything V2?**
> The DINOv2 backbone divides the image into 14×14 patches, so both height and
> width must be exact multiples of 14. The commonly suggested 384×640 is not
> (`384 / 14 = 27.4`). The closest valid size is `392×630` (28×14 and 45×14).

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

# Print stage timings every ~120 frames
python demo.py --source data/samples/dashcam.mp4 --profile
```

Press `q` in the window to quit.

## GPU/CPU execution policy

The pipeline now tries ONNX Runtime `CUDAExecutionProvider` first, and falls back
to `CPUExecutionProvider` automatically if CUDA is unavailable.

CPU fallback is optimized with:
- `ORT_ENABLE_ALL` graph optimizations
- tuned thread counts for detector/depth sessions
- reduced per-frame preprocessing work (depth input is only built when needed)
- cached YOLO decode grids + cached anchor outputs

## CARLA preprocessing (Kaggle archive -> training subset)

`notebooks/01_generate_carla_data.ipynb` is optional if you already have CARLA data.
Use this script to build a compact, balanced synthetic dataset with `domain=0`:

```bash
python scripts/preprocess_carla_dataset.py \
  --archive /data/archive \
  --target-samples 2400 \
  --model models/yolopv2_int8.onnx
```

This writes:
- `data/carla/images/*.jpg`
- `data/carla/labels/*.txt`
- `data/carla/masks/*.png`
- `data/carla/manifest.csv`

The manifest is directly compatible with `notebooks/03_finetune_yolopv2.ipynb`.

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

| Component | Input size | Per-frame cost |
|---|---|---|
| YOLOPv2 INT8 | 640×640 | ~90–120 ms |
| Depth Anything V2 Small (every 3rd frame, amortized) | 392×630 | ~50 ms |
| Fusion + navigation | — | ~3 ms |
| **Total** | | **~100–130 ms → 7–10 FPS** |

TTS runs in a background thread — zero impact on frame rate.

---

## Troubleshooting

**`AssertionError: Input image height X is not a multiple of patch height 14`**
The depth model input dimensions must each be divisible by 14. If you re-export
with a custom size, make sure both H and W satisfy `dim % 14 == 0`.

**`torch.onnx export fails with torch.export error`**
Add `dynamo=False` to your `torch.onnx.export()` call. PyTorch 2.x defaults to
the new `torch.export`-based exporter which fails on dynamic ViT ops in
Depth Anything V2.

**YOLOPv2 download fails (404)**
The `wget` URL points to a GitHub release asset. If it becomes unavailable,
manually download `YOLOPv2.onnx` from
`https://github.com/Kazuhito00/YOLOPv2-ONNX-Sample/releases` and place it in
`models/yolopv2.onnx`.

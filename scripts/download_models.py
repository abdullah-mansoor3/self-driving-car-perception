"""
download_models.py
──────────────────
Run once before starting the live demo:
    python scripts/download_models.py

What this does:
  1. Downloads YOLOPv2 as a pre-exported ONNX directly from GitHub releases
     (no .pt weights, no PyTorch export step needed)
  2. Downloads MiDaS Small ONNX from a Hugging Face ONNX export
  3. INT8-quantizes both ONNX models for faster CPU inference

All final files land in models/.
"""

import sys
import subprocess
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run(cmd: str):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}")
        sys.exit(1)


def already_exists(path: Path, label: str, min_bytes: int = 1) -> bool:
    if path.exists():
        if path.stat().st_size < min_bytes:
            print(f"[WARN] {label} at {path} looks incomplete; re-downloading")
            path.unlink()
            return False
        print(f"[SKIP] {label} already exists at {path}")
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — YOLOPv2
#
# The official CAIC-AD Google Drive link is dead.
# We download the pre-exported ONNX directly from the Kazuhito00 community
# inference repo (MIT licence), which ships the ONNX in its GitHub releases.
# The community export has dynamic H/W axes; the runtime preprocessor feeds
# 320x320 and the finetune notebook exports the final model at 320x320.
# ─────────────────────────────────────────────────────────────────────────────

YOLOPV2_ONNX_URL = (
    "https://github.com/Kazuhito00/YOLOPv2-ONNX-Sample"
    "/releases/download/v0.0.0/YOLOPv2.onnx"
)


def download_yolopv2():
    onnx_path = MODELS_DIR / "yolopv2.onnx"
    if already_exists(onnx_path, "YOLOPv2 ONNX"):
        return

    print("\n[YOLOPv2] Downloading pre-exported ONNX (~38 MB)...")
    run(f'wget -O "{onnx_path}" "{YOLOPV2_ONNX_URL}"')
    print(f"[OK] YOLOPv2 ONNX saved to {onnx_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — MiDaS Small
#
# The official MiDaS v2.1 GitHub release provides the small model as .pt and
# OpenVINO, but not this ONNX asset. Use a public ONNX export of the same
# midas_v21_small_256 model. Input size is 256x256 RGB.
# ─────────────────────────────────────────────────────────────────────────────

MIDAS_SMALL_ONNX_URL = (
    "https://huggingface.co/julienkay/sentis-MiDaS/resolve/main/"
    "onnx/midas_v21_small_256.onnx"
)


def download_midas_small():
    onnx_path = MODELS_DIR / "midas_small.onnx"
    if already_exists(onnx_path, "MiDaS Small ONNX", min_bytes=10 * 1024 * 1024):
        return

    print("\n[Depth] Downloading MiDaS Small ONNX (~66 MB)...")
    run(f'wget -O "{onnx_path}" "{MIDAS_SMALL_ONNX_URL}"')
    print(f"[OK] MiDaS Small ONNX saved to {onnx_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 -- INT8 quantization (halves model size, ~30% faster on CPU)
# ─────────────────────────────────────────────────────────────────────────────

QUANT_SCRIPT = """
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("{src}", "{dst}", weight_type=QuantType.QInt8)
print("Quantized: {dst}")
"""


def quantize(model_name: str):
    src = MODELS_DIR / f"{model_name}.onnx"
    dst = MODELS_DIR / f"{model_name}_int8.onnx"
    if already_exists(dst, f"{model_name} INT8"):
        return
    tmp = ROOT / "_quant_tmp.py"
    tmp.write_text(QUANT_SCRIPT.format(src=src, dst=dst))
    run(f'python "{tmp}"')
    tmp.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Self-driving perception -- model download + export")
    print("=" * 60)

    print("\n[1/4] YOLOPv2 -- direct ONNX download")
    download_yolopv2()

    print("\n[2/4] MiDaS Small -- direct ONNX download")
    download_midas_small()

    print("\n[3/4] INT8 quantization -- YOLOPv2")
    quantize("yolopv2")

    print("\n[4/4] INT8 quantization -- MiDaS Small")
    quantize("midas_small")

    print("\n" + "=" * 60)
    print("  All models ready. Run:  python demo.py --source 0")
    print("=" * 60)

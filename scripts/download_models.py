"""
download_models.py
──────────────────
Run once before starting the live demo:
    python scripts/download_models.py

What this does:
  1. Downloads YOLOPv2 as a pre-exported ONNX directly from GitHub releases
     (no .pt weights, no PyTorch export step needed)
  2. Downloads Depth Anything V2 Small checkpoint from HuggingFace and
     exports it to ONNX using the legacy torch.onnx exporter
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


def already_exists(path: Path, label: str) -> bool:
    if path.exists():
        print(f"[SKIP] {label} already exists at {path}")
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — YOLOPv2
#
# The official CAIC-AD Google Drive link is dead.
# We download the pre-exported ONNX directly from the Kazuhito00 community
# inference repo (MIT licence), which ships the ONNX in its GitHub releases.
# Input size for this export: 640x640 (square).
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
# Step 2 — Depth Anything V2 Small
#
# DINOv2 patch embeddings require H and W to be exact multiples of 14.
#   384 / 14 = 27.43  <- crashes
#   392 / 14 = 28.00  <- OK
#   640 / 14 = 45.71  <- crashes
#   630 / 14 = 45.00  <- OK
#
# Export uses dynamo=False to force the legacy torch.onnx exporter.
# PyTorch 2.x defaults to torch.export which fails on dynamic ViT ops.
# ─────────────────────────────────────────────────────────────────────────────

DEPTH_EXPORT_SCRIPT = """
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path("{repo}").resolve()))
from depth_anything_v2.dpt import DepthAnythingV2

model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
ckpt  = torch.load("{ckpt}", map_location="cpu")
model.load_state_dict(ckpt)
model.eval()

# 392 = 28x14,  630 = 45x14  -- both divisible by DINOv2 patch size (14)
dummy = torch.zeros(1, 3, 392, 630)

torch.onnx.export(
    model,
    dummy,
    "{out}",
    input_names=["image"],
    output_names=["depth"],
    opset_version=16,
    dynamo=False,             # legacy exporter -- avoids torch.export failures
    do_constant_folding=True,
)
print("[OK] Depth Anything V2 Small ONNX exported.")
"""


def download_depth_anything():
    onnx_path = MODELS_DIR / "depth_anything_v2_small.onnx"
    if already_exists(onnx_path, "Depth Anything V2 Small ONNX"):
        return

    ckpt_path = MODELS_DIR / "depth_anything_v2_vits.pth"
    repo_dir  = ROOT / "_depth_anything_repo"

    # Clone repo
    if not repo_dir.exists():
        run(f"git clone https://github.com/DepthAnything/Depth-Anything-V2 {repo_dir}")
        run(f"pip install -r {repo_dir}/requirements.txt -q")

    # Download checkpoint from HuggingFace
    if not ckpt_path.exists():
        print("\n[Depth] Downloading Depth Anything V2 Small checkpoint (~100 MB)...")
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="depth-anything/Depth-Anything-V2-Small",
            filename="depth_anything_v2_vits.pth",
            local_dir=str(MODELS_DIR),
        )
        print(f"[Depth] Checkpoint saved to {downloaded}")

    # Write tmp export script and run it
    export_code = DEPTH_EXPORT_SCRIPT.format(
        repo=repo_dir,
        ckpt=ckpt_path,
        out=onnx_path,
    )
    tmp_script = ROOT / "_export_depth_tmp.py"
    tmp_script.write_text(export_code)
    run(f'python "{tmp_script}"')
    tmp_script.unlink()
    print(f"[OK] Depth ONNX saved to {onnx_path}")


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

    print("\n[2/4] Depth Anything V2 Small -- checkpoint + ONNX export")
    download_depth_anything()

    print("\n[3/4] INT8 quantization -- YOLOPv2")
    quantize("yolopv2")

    print("\n[4/4] INT8 quantization -- Depth Anything V2 Small")
    quantize("depth_anything_v2_small")

    print("\n" + "=" * 60)
    print("  All models ready. Run:  python demo.py --source 0")
    print("=" * 60)

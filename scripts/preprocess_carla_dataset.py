"""
Preprocess CARLA archive into a compact YOLOP-style training set.

Inputs:
  - CARLA archive folders under /data/archive or ./data/archive
Outputs:
  - data/carla/images/*.jpg
  - data/carla/labels/*.txt      (YOLO labels from teacher detector)
  - data/carla/masks/*.png       (2-channel da/lane mask pack)
  - data/carla/manifest.csv      (domain=0 for discriminator head)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import csv
import sys
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import YOLOPv2, preprocess_yolo


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess CARLA archive for YOLOP finetuning")
    p.add_argument("--archive", default=None, help="Path to CARLA archive root")
    p.add_argument("--out", default="data/carla", help="Output dataset root")
    p.add_argument("--target-samples", type=int, default=2400,
                   help="Total CARLA samples to keep after downsampling")
    p.add_argument("--conf", type=float, default=0.20, help="Teacher confidence threshold")
    p.add_argument(
        "--iou",
        type=float,
        default=0.65,
        help="NMS IoU threshold (higher keeps more overlapping boxes)",
    )
    p.add_argument("--model", default="models/yolopv2_int8.onnx", help="Teacher ONNX path")
    p.add_argument(
        "--chunk-gb",
        type=float,
        default=25.0,
        help="Process archive in chunks of this many GB of source image files",
    )
    return p.parse_args()


def resolve_archive(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(Path("/data/archive"))
    candidates.append(Path("data/archive"))
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("CARLA archive not found. Checked: /data/archive and data/archive")


def list_image_dirs(archive_root: Path) -> list[Path]:
    return sorted(archive_root.rglob("vehicle.tesla.model3.master/image_2"))


def iter_image_chunks(image_dirs: list[Path], chunk_bytes: int):
    """
    Yield lists of image paths where cumulative source-file size is <= chunk_bytes.
    """
    chunk = []
    cur_bytes = 0
    for d in image_dirs:
        for img in sorted(d.glob("*.png")):
            try:
                size = img.stat().st_size
            except OSError:
                continue
            if chunk and (cur_bytes + size) > chunk_bytes:
                yield chunk, cur_bytes
                chunk = []
                cur_bytes = 0
            chunk.append(img)
            cur_bytes += size
    if chunk:
        yield chunk, cur_bytes


def write_yolo_labels(path: Path, dets: list[dict], w: int, h: int):
    lines = []
    for d in dets:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0
        lines.append(
            f"{int(d['cls'])} {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f}"
        )
    path.write_text("\n".join(lines))


def pack_masks(da_mask: np.ndarray, lane_mask: np.ndarray) -> np.ndarray:
    # channel 0: drivable, channel 1: lane
    out = np.zeros((da_mask.shape[0], da_mask.shape[1], 3), dtype=np.uint8)
    out[..., 0] = (da_mask > 0).astype(np.uint8) * 255
    out[..., 1] = (lane_mask > 0).astype(np.uint8) * 255
    return out


def main():
    args = parse_args()
    archive_root = resolve_archive(args.archive)
    out_root = Path(args.out)
    images_dir = out_root / "images"
    labels_dir = out_root / "labels"
    masks_dir = out_root / "masks"
    for p in (images_dir, labels_dir, masks_dir):
        p.mkdir(parents=True, exist_ok=True)

    img_dirs = list_image_dirs(archive_root)
    if not img_dirs:
        raise RuntimeError(f"No image_2 folders found under {archive_root}")

    chunk_bytes = int(max(1.0, args.chunk_gb) * (1024 ** 3))
    chunks = list(iter_image_chunks(img_dirs, chunk_bytes))
    if not chunks:
        raise RuntimeError("No CARLA images found")

    total_images = sum(len(c[0]) for c in chunks)
    stride = max(1, int(np.ceil(total_images / max(1, args.target_samples))))

    teacher = YOLOPv2(args.model, conf_thresh=args.conf, iou_thresh=args.iou)
    rows = []
    kept = 0
    seen = 0
    sampled_total = 0
    for chunk_idx, (chunk_paths, chunk_size) in enumerate(chunks, start=1):
        print(
            f"[CARLA] chunk {chunk_idx}/{len(chunks)}: "
            f"{len(chunk_paths)} images, {chunk_size / (1024 ** 3):.2f} GB"
        )
        for img_path in chunk_paths:
            if seen % stride != 0:
                seen += 1
                continue
            if sampled_total >= args.target_samples:
                break

            frame = cv2.imread(str(img_path))
            seen += 1
            if frame is None:
                continue
            sampled_total += 1
            h, w = frame.shape[:2]
            x, orig_shape = preprocess_yolo(frame)
            dets, da_mask, lane_mask = teacher.infer(x, orig_shape)
            if not dets:
                continue

            stem = f"carla_{sampled_total:07d}"
            out_img = images_dir / f"{stem}.jpg"
            out_lbl = labels_dir / f"{stem}.txt"
            out_msk = masks_dir / f"{stem}.png"

            cv2.imwrite(str(out_img), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            write_yolo_labels(out_lbl, dets, w, h)
            cv2.imwrite(str(out_msk), pack_masks(da_mask, lane_mask))

            rows.append(
                {
                    "image_path": str(out_img),
                    "label_path": str(out_lbl),
                    "mask_path": str(out_msk),
                    "domain": 0,
                    "source": "carla",
                }
            )
            kept += 1
            if kept % 250 == 0:
                print(f"[CARLA] kept {kept}/{args.target_samples}")
        if sampled_total >= args.target_samples:
            break

    manifest = out_root / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "label_path", "mask_path", "domain", "source"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] archive: {archive_root}")
    print(f"[DONE] sampled: {sampled_total}  kept: {len(rows)}")
    print(f"[DONE] manifest: {manifest}")


if __name__ == "__main__":
    main()

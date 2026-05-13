"""
src/pipeline/preprocessor.py
─────────────────────────────
Resize and normalize a raw BGR frame (from cv2) into the float32
tensors that both ONNX models expect.
"""

import cv2
import numpy as np
from typing import Any

# Both models trained / fine-tuned with ImageNet normalization
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

YOLO_W, YOLO_H   = 320, 320   # default YOLOPv2 export target
DEPTH_W, DEPTH_H = 256, 256   # MiDaS Small ONNX input


def preprocess_yolo(
    bgr_frame: np.ndarray,
    input_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    orig_shape = bgr_frame.shape[:2]
    yolo_w, yolo_h = input_size if input_size is not None else (YOLO_W, YOLO_H)
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    orig_h, orig_w = orig_shape
    scale = min(yolo_w / orig_w, yolo_h / orig_h)
    resized_w = max(1, int(round(orig_w * scale)))
    resized_h = max(1, int(round(orig_h * scale)))
    pad_x = (yolo_w - resized_w) // 2
    pad_y = (yolo_h - resized_h) // 2

    resized = cv2.resize(rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    yolo_resized = np.full((yolo_h, yolo_w, 3), 114, dtype=np.uint8)
    yolo_resized[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized

    yolo_norm = (yolo_resized.astype(np.float32) / 255.0 - _MEAN) / _STD
    yolo_chw = yolo_norm.transpose(2, 0, 1)
    meta = {
        "orig_shape": orig_shape,
        "input_size": (yolo_w, yolo_h),
        "scale": scale,
        "pad": (pad_x, pad_y),
        "resized_shape": (resized_w, resized_h),
    }
    return np.expand_dims(yolo_chw, axis=0), meta


def preprocess_depth(
    bgr_frame: np.ndarray,
    depth_size: tuple[int, int] | None = None,
) -> np.ndarray:
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    if depth_size is None:
        depth_w, depth_h = DEPTH_W, DEPTH_H
    else:
        depth_w, depth_h = depth_size
    depth_resized = cv2.resize(rgb, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)
    depth_norm = (depth_resized.astype(np.float32) / 255.0 - _MEAN) / _STD
    depth_chw = depth_norm.transpose(2, 0, 1)
    return np.expand_dims(depth_chw, axis=0)


def preprocess(
    bgr_frame: np.ndarray,
    depth_size: tuple[int, int] | None = None,
    yolo_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple]:
    """
    Parameters
    ----------
    bgr_frame : H x W x 3  uint8  (direct cv2 read)

    Returns
    -------
    yolo_tensor  : (1, 3, H, W) float32 — feed to YOLOPv2
    depth_tensor : (1, 3, 256, 256) float32 — feed to MiDaS Small
    orig_shape   : (orig_H, orig_W) — needed to scale detections back
    """
    orig_shape = bgr_frame.shape[:2]

    yolo_tensor, orig_shape = preprocess_yolo(bgr_frame, input_size=yolo_size)
    depth_tensor = preprocess_depth(bgr_frame, depth_size=depth_size)
    return yolo_tensor, depth_tensor, orig_shape

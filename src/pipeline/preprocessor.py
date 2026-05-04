"""
src/pipeline/preprocessor.py
─────────────────────────────
Resize and normalize a raw BGR frame (from cv2) into the float32
tensors that both ONNX models expect.
"""

import cv2
import numpy as np

# Both models trained / fine-tuned with ImageNet normalization
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

YOLO_W, YOLO_H   = 640, 640   # YOLOPv2 fixed input (Kazuhito00 ONNX export)
DEPTH_W, DEPTH_H = 630, 392   # Depth Anything V2 Small (both divisible by 14)


def preprocess(
    bgr_frame: np.ndarray,
    depth_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple]:
    """
    Parameters
    ----------
    bgr_frame : H x W x 3  uint8  (direct cv2 read)

    Returns
    -------
    yolo_tensor  : (1, 3, 640, 640) float32 — feed to YOLOPv2
    depth_tensor : (1, 3, H, W) float32 — feed to Depth Anything
    orig_shape   : (orig_H, orig_W) — needed to scale detections back
    """
    orig_shape = bgr_frame.shape[:2]

    # Convert BGR → RGB
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    # Resize for YOLO (640x640)
    yolo_resized = cv2.resize(rgb, (YOLO_W, YOLO_H), interpolation=cv2.INTER_LINEAR)
    yolo_norm = (yolo_resized.astype(np.float32) / 255.0 - _MEAN) / _STD
    yolo_chw = yolo_norm.transpose(2, 0, 1)
    yolo_tensor = np.expand_dims(yolo_chw, axis=0)

    # Resize for Depth (default 392x630)
    if depth_size is None:
        depth_w, depth_h = DEPTH_W, DEPTH_H
    else:
        depth_w, depth_h = depth_size
    depth_resized = cv2.resize(rgb, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)
    depth_norm = (depth_resized.astype(np.float32) / 255.0 - _MEAN) / _STD
    depth_chw = depth_norm.transpose(2, 0, 1)
    depth_tensor = np.expand_dims(depth_chw, axis=0)

    return yolo_tensor, depth_tensor, orig_shape

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

YOLO_W, YOLO_H   = 640, 384   # YOLOPv2 fixed input
DEPTH_W, DEPTH_H = 640, 384   # Depth Anything V2 Small — same for simplicity


def preprocess(bgr_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple]:
    """
    Parameters
    ----------
    bgr_frame : H x W x 3  uint8  (direct cv2 read)

    Returns
    -------
    yolo_tensor  : (1, 3, 384, 640) float32 — feed to YOLOPv2
    depth_tensor : (1, 3, 384, 640) float32 — feed to Depth Anything
    orig_shape   : (orig_H, orig_W) — needed to scale detections back
    """
    orig_shape = bgr_frame.shape[:2]

    # Convert BGR → RGB
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    # Resize
    resized = cv2.resize(rgb, (YOLO_W, YOLO_H), interpolation=cv2.INTER_LINEAR)

    # Normalize and convert to float32
    normalized = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD

    # HWC → CHW → add batch dim
    chw = normalized.transpose(2, 0, 1)
    tensor = np.expand_dims(chw, axis=0)  # (1, 3, H, W)

    # Both models take the same tensor here; depth model may later use a
    # different resolution if you swap it out, so we keep them separate.
    return tensor.copy(), tensor.copy(), orig_shape

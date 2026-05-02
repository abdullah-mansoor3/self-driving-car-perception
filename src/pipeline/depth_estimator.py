"""
src/pipeline/depth_estimator.py
────────────────────────────────
Depth Anything V2 Small — ONNX wrapper.

Runs in a background thread every N frames (default 3) and exposes
a cached depth map so the main pipeline never blocks waiting for it.
"""

import threading
import numpy as np
import onnxruntime as ort
import cv2


class DepthEstimator:
    def __init__(self, onnx_path: str, skip_frames: int = 3):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session     = ort.InferenceSession(onnx_path, sess_options=opts,
                                                providers=["CPUExecutionProvider"])
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.skip_frames = skip_frames

        # Shared state
        self._depth_map  = None          # latest normalized depth (H, W) float32
        self._lock       = threading.Lock()
        self._frame_idx  = 0
        self._thread     = None

        print(f"[Depth] Loaded from {onnx_path}  (skip_frames={skip_frames})")

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, tensor: np.ndarray):
        """
        Call once per frame from the main loop.
        Launches a background thread every `skip_frames` frames.
        Never blocks the caller.
        """
        self._frame_idx += 1
        if self._frame_idx % self.skip_frames == 0:
            # Only spawn a new thread if the previous one finished
            if self._thread is None or not self._thread.is_alive():
                t = threading.Thread(
                    target=self._infer,
                    args=(tensor.copy(),),   # copy so the main loop can reuse the array
                    daemon=True,
                )
                t.start()
                self._thread = t

    def get_depth_map(self) -> np.ndarray | None:
        """
        Returns the latest available depth map (H, W) float32 in [0, 1].
        Returns None until the first inference completes (~first 3 frames).
        """
        with self._lock:
            return self._depth_map.copy() if self._depth_map is not None else None

    def get_depth_at_box(self, x1: int, y1: int, x2: int, y2: int) -> float | None:
        """
        Returns the median depth in the bottom 30% of a bounding box.
        This is the road-contact region of the obstacle — more reliable
        than the full box which includes sky/background at the top.

        Returns a float in [0, 1] where 0 = closest, 1 = farthest,
        or None if no depth map is available yet.
        """
        depth = self.get_depth_map()
        if depth is None:
            return None

        h_box = y2 - y1
        y_contact = y2 - max(1, int(h_box * 0.30))   # bottom 30% start
        region = depth[y_contact:y2, x1:x2]

        if region.size == 0:
            return None
        return float(np.median(region))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _infer(self, tensor: np.ndarray):
        _, _, model_h, model_w = tensor.shape
        raw = self.session.run([self.output_name], {self.input_name: tensor})[0]
        # raw: (1, H, W) or (H, W) depending on export
        depth = raw[0] if raw.ndim == 3 else raw

        # Normalize to [0, 1]
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)

        with self._lock:
            self._depth_map = depth.astype(np.float32)

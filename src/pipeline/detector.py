"""
src/pipeline/detector.py
─────────────────────────
YOLOPv2 ONNX wrapper.

Outputs three things per forward pass:
    • boxes     — list of dicts {cls, label, conf, x1, y1, x2, y2}  (in orig px)
    • da_mask   — (H, W) uint8  0=background  1=drivable area
    • lane_mask — (H, W) uint8  0=background  1=lane line
"""

import numpy as np
import onnxruntime as ort
import cv2
import os

# Class labels as defined in BDD100K (YOLOPv2 training set)
LABELS = [
    "car", "truck", "bus", "person", "rider",
    "bicycle", "motorcycle", "traffic light", "traffic sign", "train",
]


def _sigmoid(arr: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-1.0 * arr))


def _make_grid(nx: int, ny: int) -> np.ndarray:
    xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
    return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype("float32")


def _xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Pure-numpy NMS. Returns indices of kept boxes."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


def _decode_yolop(pred_layers: list[np.ndarray], anchor_grid: list[np.ndarray]) -> np.ndarray:
    """Decode YOLOPv2 ONNX head outputs into a (1, N, 85) tensor."""
    z = []
    strides = [8, 16, 32]
    for i in range(3):
        pred = pred_layers[i]
        bs, _, ny, nx = pred.shape
        pred = pred.reshape(bs, 3, 85, ny, nx).transpose(0, 1, 3, 4, 2)
        y = _sigmoid(pred)
        grid = _make_grid(nx, ny)
        y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + grid) * strides[i]
        y[..., 2:4] = (y[..., 2:4] * 2.0) ** 2 * anchor_grid[i]
        z.append(y.reshape(bs, -1, 85))
    return np.concatenate(z, axis=1)


def _non_max_suppression(prediction: np.ndarray, conf_thres: float, iou_thres: float) -> list[np.ndarray]:
    """Runs class-agnostic NMS. Returns [xyxy, conf, cls] per image."""
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres

    output = [np.zeros((0, 6), dtype=np.float32)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]
        conf = np.max(x[:, 5:], axis=1, keepdims=True)
        cls_ids = np.argmax(x[:, 5:], axis=1).reshape(-1, 1).astype(np.float32)
        x = np.concatenate((x[:, :4], conf, cls_ids), axis=1)
        x = x[conf[:, 0] > conf_thres]
        if not x.shape[0]:
            continue

        boxes = _xywh2xyxy(x[:, :4])
        scores = x[:, 4]
        keep = _nms(boxes, scores, iou_thres)
        if keep:
            output[xi] = np.concatenate(
                (boxes[keep], scores[keep, None], x[keep, 5:6]),
                axis=1,
            )

    return output


class YOLOPv2:
    def __init__(
        self,
        onnx_path: str,
        conf_thresh: float = 0.45,
        iou_thresh: float = 0.5,
        input_size: tuple[int, int] | None = None,
    ):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(8, max(1, (os.cpu_count() or 4) - 1))
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ort.get_available_providers()
        provider_chain = []
        if "CUDAExecutionProvider" in providers:
            provider_chain.append("CUDAExecutionProvider")
        provider_chain.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=opts,
            providers=provider_chain,
        )
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh
        self._grid_cache: dict[tuple[int, int], np.ndarray] = {}
        self._anchor_cache: list[np.ndarray] | None = None

        # Grab input/output names from the ONNX graph
        model_input      = self.session.get_inputs()[0]
        self.input_name  = model_input.name
        self.input_size  = input_size or self._parse_input_size(model_input.shape)
        outputs          = self.session.get_outputs()
        output_names     = [o.name for o in outputs]

        self.det_name    = None
        self.anchor_names = []
        seg_names = []
        self.da_name     = None
        self.ll_name     = None
        self.det_is_decoded = False

        for o in outputs:
            if len(o.shape) == 3 and o.shape[-1] == 6:
                self.det_name = o.name
                self.det_is_decoded = True
            elif o.type.startswith("seq"):
                self.det_name = o.name
            elif len(o.shape) == 5 and o.shape[-1] == 2:
                self.anchor_names.append(o.name)
            elif len(o.shape) == 4 and o.shape[1] in (1, 2):
                seg_names.append(o.name)

        if len(seg_names) >= 2:
            self.da_name = seg_names[-2]
            self.ll_name = seg_names[-1]
        elif len(seg_names) == 1:
            self.da_name = seg_names[0]
            self.ll_name = seg_names[0]

        # Fallbacks for unexpected export ordering
        if self.det_name is None and output_names:
            self.det_name = output_names[0]
        if len(self.anchor_names) < 3 and len(output_names) >= 4:
            self.anchor_names = output_names[1:4]
        if self.da_name is None and len(output_names) >= 2:
            self.da_name = output_names[-2]
        if self.ll_name is None and len(output_names) >= 1:
            self.ll_name = output_names[-1]

        print(f"[YOLOPv2] Loaded from {onnx_path}")
        print(f"          Providers: {self.session.get_providers()}")
        print(f"          Input:   {self.input_name} {self.input_size[0]}x{self.input_size[1]}")
        print(f"          Outputs: {output_names}")

    @staticmethod
    def _parse_input_size(shape: list) -> tuple[int, int]:
        if len(shape) != 4:
            raise ValueError(
                f"YOLOPv2 ONNX input must be NCHW, got shape {shape}"
            )
        if not isinstance(shape[2], int) or not isinstance(shape[3], int):
            return 320, 320
        return shape[3], shape[2]

    def _decode_with_cache(self, pred_layers: list[np.ndarray], anchor_grid: list[np.ndarray]) -> np.ndarray:
        z = []
        strides = [8, 16, 32]
        for i in range(3):
            pred = pred_layers[i]
            bs, _, ny, nx = pred.shape
            pred = pred.reshape(bs, 3, 85, ny, nx).transpose(0, 1, 3, 4, 2)
            y = _sigmoid(pred)
            key = (ny, nx)
            if key not in self._grid_cache:
                self._grid_cache[key] = _make_grid(nx, ny)
            grid = self._grid_cache[key]
            y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + grid) * strides[i]
            y[..., 2:4] = (y[..., 2:4] * 2.0) ** 2 * anchor_grid[i]
            z.append(y.reshape(bs, -1, 85))
        return np.concatenate(z, axis=1)

    def _postprocess_decoded_det(self, pred: np.ndarray) -> np.ndarray:
        """Post-process exported [cx, cy, w, h, objectness, class_score/class_id]."""
        pred = pred[0] if pred.ndim == 3 else pred
        if pred.size == 0:
            return np.zeros((0, 6), dtype=np.float32)

        obj = pred[:, 4]
        cls_or_score = pred[:, 5]
        score_like = np.nanmin(cls_or_score) >= 0.0 and np.nanmax(cls_or_score) <= 1.0
        if score_like:
            scores = obj * cls_or_score
            cls_ids = np.zeros_like(scores, dtype=np.float32)
        else:
            scores = obj
            cls_ids = cls_or_score.astype(np.float32)

        keep = scores > self.conf_thresh
        if not np.any(keep):
            return np.zeros((0, 6), dtype=np.float32)

        boxes = _xywh2xyxy(pred[keep, :4])
        scores = scores[keep]
        cls_ids = cls_ids[keep]
        keep_idx = _nms(boxes, scores, self.iou_thresh)
        if not keep_idx:
            return np.zeros((0, 6), dtype=np.float32)

        return np.concatenate(
            (boxes[keep_idx], scores[keep_idx, None], cls_ids[keep_idx, None]),
            axis=1,
        ).astype(np.float32)

    @staticmethod
    def _drivable_mask(raw: np.ndarray) -> np.ndarray:
        raw = raw[0]
        if raw.shape[0] == 1:
            return (raw[0] > 0.5).astype(np.uint8)
        return (raw[0] < 0.5).astype(np.uint8)

    @staticmethod
    def _lane_mask(raw: np.ndarray) -> np.ndarray:
        raw = raw[0]
        if raw.shape[0] == 1:
            return (raw[0] > 0.5).astype(np.uint8)
        return (raw[1] > 0.5).astype(np.uint8)

    def infer(
        self,
        tensor: np.ndarray,
        orig_shape: tuple | dict,
    ) -> tuple[list[dict], np.ndarray, np.ndarray]:

        _, _, model_h, model_w = tensor.shape
        if isinstance(orig_shape, dict):
            orig_h, orig_w = orig_shape["orig_shape"]
            scale = float(orig_shape.get("scale", 1.0))
            pad_x, pad_y = orig_shape.get("pad", (0, 0))
            resized_w, resized_h = orig_shape.get("resized_shape", (model_w, model_h))
        else:
            orig_h, orig_w = orig_shape
            scale = min(model_w / orig_w, model_h / orig_h)
            resized_w = max(1, int(round(orig_w * scale)))
            resized_h = max(1, int(round(orig_h * scale)))
            pad_x = (model_w - resized_w) // 2
            pad_y = (model_h - resized_h) // 2

        # ── Forward pass ──────────────────────────────────────────────────
        if self.det_is_decoded:
            det_raw, da_raw, ll_raw = self.session.run(
                [self.det_name, self.da_name, self.ll_name],
                {self.input_name: tensor},
            )
        elif self._anchor_cache is None:
            det_raw, a0, a1, a2, da_raw, ll_raw = self.session.run(
                [self.det_name, *self.anchor_names[:3], self.da_name, self.ll_name],
                {self.input_name: tensor},
            )
            self._anchor_cache = [a0, a1, a2]
        else:
            det_raw, da_raw, ll_raw = self.session.run(
                [self.det_name, self.da_name, self.ll_name],
                {self.input_name: tensor},
            )
            a0, a1, a2 = self._anchor_cache
        # det_raw: list of 3 tensors (1, 255, ny, nx)
        # a0/a1/a2: anchor grids
        # da_raw: (1, 2, H, W)
        # ll_raw: (1, 1, H, W)

        # ── Segmentation masks ─────────────────────────────────────────────
        da_mask   = self._drivable_mask(da_raw)
        lane_mask = self._lane_mask(ll_raw)

        da_mask = da_mask[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w]
        lane_mask = lane_mask[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w]

        # Scale masks back to original frame size after removing letterbox padding.
        da_mask   = cv2.resize(da_mask,   (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        lane_mask = cv2.resize(lane_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # ── Detection post-processing ──────────────────────────────────────
        if self.det_is_decoded:
            dets = self._postprocess_decoded_det(det_raw)
        else:
            pred = self._decode_with_cache(det_raw, [a0, a1, a2])
            nms_out = _non_max_suppression(pred, self.conf_thresh, self.iou_thresh)
            dets = nms_out[0] if len(nms_out) else np.zeros((0, 6), dtype=np.float32)

        if dets.size == 0:
            return [], da_mask, lane_mask

        # Undo letterbox padding and scale to original frame size.
        dets[:, [0, 2]] = (dets[:, [0, 2]] - pad_x) / scale
        dets[:, [1, 3]] = (dets[:, [1, 3]] - pad_y) / scale
        dets[:, [0, 2]] = np.clip(dets[:, [0, 2]], 0, orig_w - 1)
        dets[:, [1, 3]] = np.clip(dets[:, [1, 3]], 0, orig_h - 1)

        detections = []
        for x1, y1, x2, y2, conf, cls_id in dets:
            cls_id = int(cls_id)
            label = LABELS[cls_id] if cls_id < len(LABELS) else f"class_{cls_id}"
            detections.append({
                "cls":   cls_id,
                "label": label,
                "conf":  float(conf),
                "x1":    int(x1),
                "y1":    int(y1),
                "x2":    int(x2),
                "y2":    int(y2),
            })

        return detections, da_mask, lane_mask

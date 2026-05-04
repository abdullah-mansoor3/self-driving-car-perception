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
    def __init__(self, onnx_path: str, conf_thresh: float = 0.45, iou_thresh: float = 0.5):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4          # use 4 of your i5 cores
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session     = ort.InferenceSession(onnx_path, sess_options=opts,
                                                providers=["CPUExecutionProvider"])
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh

        # Grab input/output names from the ONNX graph
        self.input_name  = self.session.get_inputs()[0].name
        outputs          = self.session.get_outputs()
        output_names     = [o.name for o in outputs]

        self.det_name    = None
        self.anchor_names = []
        self.da_name     = None
        self.ll_name     = None

        for o in outputs:
            if o.type.startswith("seq"):
                self.det_name = o.name
            elif len(o.shape) == 5 and o.shape[-1] == 2:
                self.anchor_names.append(o.name)
            elif len(o.shape) == 4 and o.shape[1] == 2:
                self.da_name = o.name
            elif len(o.shape) == 4 and o.shape[1] == 1:
                self.ll_name = o.name

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
        print(f"          Input:   {self.input_name}")
        print(f"          Outputs: {output_names}")

    def infer(
        self,
        tensor: np.ndarray,           # (1, 3, 640, 640) float32
        orig_shape: tuple,            # (orig_H, orig_W)
    ) -> tuple[list[dict], np.ndarray, np.ndarray]:

        _, _, model_h, model_w = tensor.shape
        orig_h, orig_w         = orig_shape

        # ── Forward pass ──────────────────────────────────────────────────
        det_raw, a0, a1, a2, da_raw, ll_raw = self.session.run(
            [self.det_name, *self.anchor_names[:3], self.da_name, self.ll_name],
            {self.input_name: tensor},
        )
        # det_raw: list of 3 tensors (1, 255, ny, nx)
        # a0/a1/a2: anchor grids
        # da_raw: (1, 2, H, W)
        # ll_raw: (1, 1, H, W)

        # ── Segmentation masks ─────────────────────────────────────────────
        da_mask   = (1.0 - da_raw[0][0] > 0.5).astype(np.uint8)
        lane_mask = (ll_raw[0][0] > 0.5).astype(np.uint8)

        # Scale masks back to original frame size
        da_mask   = cv2.resize(da_mask,   (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        lane_mask = cv2.resize(lane_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # ── Detection post-processing ──────────────────────────────────────
        pred = _decode_yolop(det_raw, [a0, a1, a2])
        nms_out = _non_max_suppression(pred, self.conf_thresh, self.iou_thresh)
        dets = nms_out[0] if len(nms_out) else np.zeros((0, 6), dtype=np.float32)

        if dets.size == 0:
            return [], da_mask, lane_mask

        # Scale to original frame size
        sx = orig_w / model_w
        sy = orig_h / model_h
        dets[:, [0, 2]] *= sx
        dets[:, [1, 3]] *= sy

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

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
        output_names     = [o.name for o in self.session.get_outputs()]
        # YOLOPv2 exports: [det_out, da_seg_out, ll_seg_out]
        self.det_name    = output_names[0]
        self.da_name     = output_names[1]
        self.ll_name     = output_names[2]

        print(f"[YOLOPv2] Loaded from {onnx_path}")
        print(f"          Input:   {self.input_name}")
        print(f"          Outputs: {output_names}")

    def infer(
        self,
        tensor: np.ndarray,           # (1, 3, 384, 640) float32
        orig_shape: tuple,            # (orig_H, orig_W)
    ) -> tuple[list[dict], np.ndarray, np.ndarray]:

        _, _, model_h, model_w = tensor.shape
        orig_h, orig_w         = orig_shape

        # ── Forward pass ──────────────────────────────────────────────────
        det_raw, da_raw, ll_raw = self.session.run(
            [self.det_name, self.da_name, self.ll_name],
            {self.input_name: tensor},
        )
        # det_raw : (1, N, 6)   — [cx, cy, w, h, obj_conf, cls_conf...]
        #           OR (1, N, 5+num_classes) depending on export
        # da_raw  : (1, 2, H, W)
        # ll_raw  : (1, 2, H, W)

        # ── Segmentation masks ─────────────────────────────────────────────
        da_mask   = da_raw[0].argmax(axis=0).astype(np.uint8)   # (H, W)
        lane_mask = ll_raw[0].argmax(axis=0).astype(np.uint8)   # (H, W)

        # Scale masks back to original frame size
        da_mask   = cv2.resize(da_mask,   (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        lane_mask = cv2.resize(lane_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # ── Detection post-processing ──────────────────────────────────────
        preds = det_raw[0]   # (N, 6+)  or (N, 5+num_classes)

        # Separate objectness from class scores
        if preds.shape[1] == 6:
            # Format: [cx, cy, w, h, conf, cls_id]  (already argmaxed by exporter)
            obj_conf = preds[:, 4]
            cls_ids  = preds[:, 5].astype(int)
            scores   = obj_conf
        else:
            # Format: [cx, cy, w, h, obj_conf, cls0, cls1, ...]
            obj_conf   = preds[:, 4]
            cls_confs  = preds[:, 5:]
            cls_ids    = cls_confs.argmax(axis=1)
            scores     = obj_conf * cls_confs.max(axis=1)

        # Confidence threshold filter
        mask = scores >= self.conf_thresh
        preds, scores, cls_ids = preds[mask], scores[mask], cls_ids[mask]

        if len(preds) == 0:
            return [], da_mask, lane_mask

        # cx, cy, w, h → x1, y1, x2, y2 (in model-input coords)
        cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Scale to original frame size
        sx = orig_w / model_w
        sy = orig_h / model_h
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy

        boxes_arr = np.stack([x1, y1, x2, y2], axis=1)
        keep      = _nms(boxes_arr, scores, self.iou_thresh)

        detections = []
        for i in keep:
            cls_id = int(cls_ids[i])
            detections.append({
                "cls":   cls_id,
                "label": LABELS[cls_id] if cls_id < len(LABELS) else "unknown",
                "conf":  float(scores[i]),
                "x1":    int(x1[i]),
                "y1":    int(y1[i]),
                "x2":    int(x2[i]),
                "y2":    int(y2[i]),
            })

        return detections, da_mask, lane_mask

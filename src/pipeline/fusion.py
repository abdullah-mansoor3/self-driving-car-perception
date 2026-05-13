"""
src/pipeline/fusion.py
───────────────────────
Combines YOLOPv2 outputs with the depth map to build a
SceneState — the single object passed to navigation logic.
"""

from dataclasses import dataclass, field
import warnings
import numpy as np

try:
    from numpy.exceptions import RankWarning
except ImportError:  # pragma: no cover - older numpy
    RankWarning = np.RankWarning


@dataclass
class Obstacle:
    label:      str
    conf:       float
    x1: int; y1: int; x2: int; y2: int
    depth:      float           # [0=closest … 1=farthest]  None → unknown
    in_ego_lane: bool = False   # True if obstacle overlaps drivable area


@dataclass
class SceneState:
    obstacles:       list[Obstacle] = field(default_factory=list)
    lane_deviation:  float = 0.0      # pixels: positive = drifting right
    drivable_clear:  bool  = True     # is the ego lane ahead clear?
    lane_detected:   bool  = False    # were lane lines found at all?
    lane_lines:      list[list[tuple[int, int]]] = field(default_factory=list)
    closest_obstacle: Obstacle | None = None


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _box_center_x(obs: Obstacle) -> int:
    return (obs.x1 + obs.x2) // 2


def _fit_lane_side(
    xs: np.ndarray,
    ys: np.ndarray,
    h: int,
    w: int,
) -> tuple[np.ndarray | None, list[tuple[int, int]]]:
    if len(xs) < 80:
        return None, []

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RankWarning)
            coeff = np.polyfit(ys.astype(np.float32), xs.astype(np.float32), deg=2)
    except np.linalg.LinAlgError:
        return None, []

    y_eval = np.linspace(int(h * 0.52), int(h * 0.96), 32)
    x_eval = np.polyval(coeff, y_eval)
    valid = (x_eval >= 0) & (x_eval < w)
    points = [
        (int(x), int(y))
        for x, y in zip(x_eval[valid], y_eval[valid])
    ]
    if len(points) < 8:
        return None, []
    return coeff, points


def _lane_deviation(lane_mask: np.ndarray) -> tuple[float, bool, list[list[tuple[int, int]]]]:
    """
    Fits left/right lane curves from mask pixels and computes how far the
    frame midpoint is from the inferred lane center.

    Returns (deviation_px, lane_detected, fitted_lane_lines).
    """
    h, w = lane_mask.shape
    ys, xs = np.where(lane_mask > 0)
    road_roi = ys >= int(h * 0.45)
    xs = xs[road_roi]
    ys = ys[road_roi]

    if len(xs) < 160:
        return 0.0, False, []

    center_x = w // 2
    left = xs < center_x
    right = xs >= center_x
    left_fit, left_points = _fit_lane_side(xs[left], ys[left], h, w)
    right_fit, right_points = _fit_lane_side(xs[right], ys[right], h, w)

    lane_lines = []
    if left_points:
        lane_lines.append(left_points)
    if right_points:
        lane_lines.append(right_points)

    if left_fit is None or right_fit is None:
        return 0.0, bool(lane_lines), lane_lines

    scan_y = int(h * 0.72)
    left_x = float(np.polyval(left_fit, scan_y))
    right_x = float(np.polyval(right_fit, scan_y))
    lane_width = right_x - left_x
    if lane_width < w * 0.12:
        return 0.0, False, lane_lines

    lane_center = (left_x + right_x) / 2.0
    frame_center = w / 2.0
    deviation = float(frame_center - lane_center)    # + = car drifting right
    return deviation, True, lane_lines


def _is_in_ego_lane(obs: Obstacle, da_mask: np.ndarray) -> bool:
    """
    Samples the drivable-area mask in the bottom half of the bounding box.
    If ≥ 40% of sampled pixels are drivable → obstacle is in our lane.
    """
    cx = (obs.x1 + obs.x2) // 2
    mid_y = (obs.y1 + obs.y2) // 2
    bot_y = obs.y2

    h, w = da_mask.shape
    mid_y = min(mid_y, h - 1)
    bot_y = min(bot_y, h - 1)
    cx    = min(max(cx, 0), w - 1)

    region = da_mask[mid_y:bot_y, max(0, cx-20):min(w, cx+20)]
    if region.size == 0:
        return False
    return float(region.mean()) >= 0.4


# ── Main fuser ────────────────────────────────────────────────────────────────

class Fuser:
    def __init__(
        self,
        max_obstacle_depth: float | None = None,
        keep_unknown_depth: bool = True,
    ):
        self.max_obstacle_depth = max_obstacle_depth
        self.keep_unknown_depth = keep_unknown_depth

    def fuse(
        self,
        detections: list[dict],
        da_mask:    np.ndarray,
        lane_mask:  np.ndarray,
        depth_fn,                  # callable(x1,y1,x2,y2) → float|None
    ) -> SceneState:

        deviation, lane_det, lane_lines = _lane_deviation(lane_mask)

        obstacles = []
        for d in detections:
            depth = depth_fn(d["x1"], d["y1"], d["x2"], d["y2"])
            if depth is None and not self.keep_unknown_depth:
                continue
            if (
                depth is not None
                and self.max_obstacle_depth is not None
                and depth > self.max_obstacle_depth
            ):
                continue

            obs = Obstacle(
                label=d["label"],
                conf=d["conf"],
                x1=d["x1"], y1=d["y1"],
                x2=d["x2"], y2=d["y2"],
                depth=depth if depth is not None else 1.0,
                in_ego_lane=False,
            )
            obs.in_ego_lane = _is_in_ego_lane(obs, da_mask)
            obstacles.append(obs)

        # Sort by depth ascending (closest first)
        obstacles.sort(key=lambda o: o.depth)
        closest_ego = next((o for o in obstacles if o.in_ego_lane), None)

        # Drivable area is clear if no ego-lane obstacle is within depth 0.35
        drivable_clear = (
            closest_ego is None or closest_ego.depth > 0.35
        )

        return SceneState(
            obstacles=obstacles,
            lane_deviation=deviation,
            drivable_clear=drivable_clear,
            lane_detected=lane_det,
            lane_lines=lane_lines,
            closest_obstacle=closest_ego,
        )

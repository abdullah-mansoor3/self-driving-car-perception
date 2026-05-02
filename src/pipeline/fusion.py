"""
src/pipeline/fusion.py
───────────────────────
Combines YOLOPv2 outputs with the depth map to build a
SceneState — the single object passed to navigation logic.
"""

from dataclasses import dataclass, field
import numpy as np


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
    closest_obstacle: Obstacle | None = None


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _box_center_x(obs: Obstacle) -> int:
    return (obs.x1 + obs.x2) // 2


def _lane_deviation(lane_mask: np.ndarray) -> tuple[float, bool]:
    """
    Finds the left and right lane line positions at 2/3 height of the frame,
    computes how far the center of the car (frame midpoint) is from the
    lane center.

    Returns (deviation_px, lane_detected).
    """
    h, w = lane_mask.shape
    scan_y = int(h * 0.66)
    row    = lane_mask[scan_y, :]

    lane_x = np.where(row > 0)[0]
    if len(lane_x) < 2:
        return 0.0, False

    left_x  = int(lane_x[:len(lane_x)//2].mean())   if len(lane_x) > 0 else 0
    right_x = int(lane_x[len(lane_x)//2:].mean())   if len(lane_x) > 0 else w

    lane_center = (left_x + right_x) // 2
    frame_center = w // 2
    deviation = float(frame_center - lane_center)    # + = car drifting right
    return deviation, True


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
    def fuse(
        self,
        detections: list[dict],
        da_mask:    np.ndarray,
        lane_mask:  np.ndarray,
        depth_fn,                  # callable(x1,y1,x2,y2) → float|None
    ) -> SceneState:

        deviation, lane_det = _lane_deviation(lane_mask)

        obstacles = []
        for d in detections:
            depth = depth_fn(d["x1"], d["y1"], d["x2"], d["y2"])
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
            closest_obstacle=closest_ego,
        )

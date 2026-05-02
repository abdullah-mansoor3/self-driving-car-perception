"""
src/pipeline/visualizer.py
───────────────────────────
Draws the perception output onto the BGR frame for the live demo window.
"""

import cv2
import numpy as np

from .fusion import SceneState, Obstacle
from .navigation import NavigationEvent, Severity

# Color palette (BGR)
COLOR_LANE      = (0,   255, 180)
COLOR_DRIVABLE  = (0,   200, 60)
COLOR_BOX       = (255, 180, 0)
COLOR_BOX_EGO   = (0,   60,  255)    # red — obstacle in our lane
COLOR_HUD_BG    = (0,   0,   0)
COLOR_WARN      = (0,   165, 255)    # orange
COLOR_CRITICAL  = (0,   0,   220)    # red


def draw(
    frame:        np.ndarray,
    da_mask:      np.ndarray,
    lane_mask:    np.ndarray,
    depth_map:    np.ndarray | None,
    state:        SceneState,
    event:        NavigationEvent | None,
) -> np.ndarray:

    out = frame.copy()

    # ── Drivable area overlay (green tint) ────────────────────────────────
    if da_mask is not None:
        da_color = np.zeros_like(out)
        da_color[da_mask == 1] = COLOR_DRIVABLE
        out = cv2.addWeighted(out, 1.0, da_color, 0.25, 0)

    # ── Lane lines overlay ────────────────────────────────────────────────
    if lane_mask is not None:
        lane_color = np.zeros_like(out)
        lane_color[lane_mask == 1] = COLOR_LANE
        out = cv2.addWeighted(out, 1.0, lane_color, 0.6, 0)

    # ── Depth map thumbnail (top-right corner) ────────────────────────────
    if depth_map is not None:
        h, w = out.shape[:2]
        thumb_w, thumb_h = 160, 96
        depth_u8 = (depth_map * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_MAGMA)
        depth_thumb = cv2.resize(depth_colored, (thumb_w, thumb_h))
        out[10:10+thumb_h, w-thumb_w-10:w-10] = depth_thumb
        cv2.putText(out, "depth", (w-thumb_w-10, 10+thumb_h+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # ── Bounding boxes ────────────────────────────────────────────────────
    for obs in state.obstacles:
        color     = COLOR_BOX_EGO if obs.in_ego_lane else COLOR_BOX
        thickness = 2 if obs.in_ego_lane else 1
        cv2.rectangle(out, (obs.x1, obs.y1), (obs.x2, obs.y2), color, thickness)

        depth_str = f"{obs.depth:.2f}" if obs.depth is not None else "??"
        label_str = f"{obs.label} {obs.conf:.0%}  d={depth_str}"
        cv2.putText(out, label_str, (obs.x1, obs.y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # ── Lane deviation indicator ──────────────────────────────────────────
    h, w = out.shape[:2]
    cx = w // 2
    dev = int(state.lane_deviation)
    arrow_x = cx - dev   # subtract because positive dev = car drifted right
    cv2.arrowedLine(out, (cx, h-20), (arrow_x, h-20), COLOR_LANE, 2, tipLength=0.3)
    cv2.putText(out, f"dev {dev:+d}px", (cx-40, h-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_LANE, 1)

    # ── Event HUD banner (bottom of frame) ───────────────────────────────
    if event is not None:
        if event.severity == Severity.CRITICAL:
            hud_color = COLOR_CRITICAL
        elif event.severity == Severity.WARN:
            hud_color = COLOR_WARN
        else:
            hud_color = (180, 180, 180)

        banner_h = 36
        cv2.rectangle(out, (0, h-banner_h), (w, h), COLOR_HUD_BG, -1)
        cv2.putText(out, event.instruction, (10, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 1, cv2.LINE_AA)

    return out

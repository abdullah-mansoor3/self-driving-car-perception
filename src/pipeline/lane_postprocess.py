"""
Runtime lane-mask post-processing.

This keeps dashed lane markings alive across adjacent processed frames and
removes small speckles before downstream lane geometry is estimated.
"""

from collections import deque

import cv2
import numpy as np


class LanePostProcessor:
    def __init__(self, history: int = 10, min_votes: int = 2):
        self.history = max(1, int(history))
        self.min_votes = max(1, int(min_votes))
        self._masks: deque[np.ndarray] = deque(maxlen=self.history)
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 17))
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def update(self, lane_mask: np.ndarray) -> np.ndarray:
        mask = (lane_mask > 0).astype(np.uint8)
        self._masks.append(mask)

        if len(self._masks) == 1:
            combined = mask
        else:
            votes = np.zeros_like(mask, dtype=np.uint8)
            for old_mask in self._masks:
                votes += old_mask
            threshold = min(self.min_votes, len(self._masks))
            combined = (votes >= threshold).astype(np.uint8)

        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self._close_kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, self._open_kernel)
        return combined.astype(np.uint8)

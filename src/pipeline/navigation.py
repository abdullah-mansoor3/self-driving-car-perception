"""
src/pipeline/navigation.py
───────────────────────────
Rule-based navigation logic.

Takes a SceneState, maintains rolling event counters,
and produces a NavigationEvent when something noteworthy happens.
The event carries a spoken instruction (including roasts).
"""

import random
import time
from dataclasses import dataclass
from enum import Enum, auto
from collections import deque

from .fusion import SceneState


class Severity(Enum):
    INFO     = auto()
    WARN     = auto()
    CRITICAL = auto()


@dataclass
class NavigationEvent:
    severity:    Severity
    instruction: str          # the text to speak aloud
    tag:         str          # machine-readable event type


# ── Thresholds ────────────────────────────────────────────────────────────────

LANE_DEVIATION_WARN     = 60    # pixels
LANE_DEVIATION_CRITICAL = 130   # pixels
DEPTH_WARN              = 0.30  # normalized (closer = lower value)
DEPTH_CRITICAL          = 0.15
ROAST_AFTER_N_EVENTS    = 3     # roast kicks in after N repeated violations


# ── Roast template banks ──────────────────────────────────────────────────────

_ROASTS_LANE = [
    "You've drifted out of your lane AGAIN. Are you actually trying to drive?",
    "The lane is the white lines. Both of them. Stay between them. This is not optional.",
    "I've seen shopping trolleys with better lane discipline than you.",
    "Your lane is that way. No, the other way. Actually, just pull over.",
    "Congratulations on discovering the shoulder. Would you like to live there?",
]

_ROASTS_TAILGATE = [
    "Do you think that car is your best friend? Give it some space.",
    "If you get any closer you'll need to buy that car dinner first.",
    "You're basically in the back seat of the car ahead. Back off.",
    "Braking distance is a concept. Please learn it.",
    "I see you've chosen violence. Slow down.",
]

_ROASTS_OBSTACLE = [
    "There is a very large object in front of you. It would prefer not to be hit.",
    "BRAKE. Or don't. I'm just a computer. But also: BRAKE.",
    "The obstacle is stationary. You are not. This is a problem.",
]


# ── Navigation engine ─────────────────────────────────────────────────────────

class Navigator:
    def __init__(self, cooldown_s: float = 2.5):
        self._cooldown_s  = cooldown_s
        self._last_spoken: dict[str, float] = {}   # tag → timestamp
        self._event_counts: dict[str, int]  = {}   # tag → consecutive count

    def process(self, state: SceneState) -> NavigationEvent | None:
        """
        Evaluate the scene and return a NavigationEvent if something
        needs to be said, or None if everything is fine / in cooldown.
        """
        events: list[NavigationEvent] = []

        # ── 1. Obstacle in ego lane ────────────────────────────────────────
        if state.closest_obstacle and state.closest_obstacle.in_ego_lane:
            d = state.closest_obstacle.depth
            label = state.closest_obstacle.label

            if d <= DEPTH_CRITICAL:
                events.append(NavigationEvent(
                    severity=Severity.CRITICAL,
                    instruction=f"BRAKE NOW — {label} directly ahead!",
                    tag="obstacle_critical",
                ))
            elif d <= DEPTH_WARN:
                events.append(NavigationEvent(
                    severity=Severity.WARN,
                    instruction=f"Slow down, {label} ahead. Keep your distance.",
                    tag="obstacle_warn",
                ))

        # ── 2. Lane departure ──────────────────────────────────────────────
        if state.lane_detected:
            dev = abs(state.lane_deviation)
            direction = "right" if state.lane_deviation > 0 else "left"

            if dev >= LANE_DEVIATION_CRITICAL:
                events.append(NavigationEvent(
                    severity=Severity.CRITICAL,
                    instruction=f"STEER {direction.upper()} — you are leaving the lane!",
                    tag="lane_critical",
                ))
            elif dev >= LANE_DEVIATION_WARN:
                events.append(NavigationEvent(
                    severity=Severity.WARN,
                    instruction=f"Drift detected — steer slightly {direction}.",
                    tag="lane_warn",
                ))

        # ── 3. No lane lines detected at all ──────────────────────────────
        elif not state.lane_detected:
            events.append(NavigationEvent(
                severity=Severity.INFO,
                instruction="Lane markings unclear — proceed with caution.",
                tag="no_lane",
            ))

        # ── 4. Pick highest-severity event, apply cooldown + roast ────────
        if not events:
            return None

        events.sort(key=lambda e: e.severity.value, reverse=True)
        chosen = events[0]

        if not self._can_speak(chosen.tag):
            return None

        # Increment consecutive counter and optionally roast
        n = self._event_counts.get(chosen.tag, 0) + 1
        self._event_counts[chosen.tag] = n

        if n >= ROAST_AFTER_N_EVENTS:
            chosen.instruction = self._pick_roast(chosen.tag)
            self._event_counts[chosen.tag] = 0   # reset after roast

        self._last_spoken[chosen.tag] = time.time()
        return chosen

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _can_speak(self, tag: str) -> bool:
        last = self._last_spoken.get(tag, 0)
        return (time.time() - last) >= self._cooldown_s

    @staticmethod
    def _pick_roast(tag: str) -> str:
        if "lane" in tag:
            return random.choice(_ROASTS_LANE)
        if "obstacle" in tag:
            return random.choice(_ROASTS_TAILGATE if "warn" in tag else _ROASTS_OBSTACLE)
        return "Pay attention. Please."

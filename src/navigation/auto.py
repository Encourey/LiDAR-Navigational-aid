"""
navigation/auto.py
──────────────────
Automatic mode switching between indoor and outdoor navigation.

Logic:
  Computes average distance of all scan points each frame.
  Maintains a rolling count of consecutive frames agreeing on the new mode.
  Only switches after AUTO_SWITCH_FRAMES consecutive frames agree —
  prevents a single noisy scan from triggering YOLO cold-start.

  Hysteresis (0.8 multiplier) prevents rapid mode flapping at boundaries.

Typical values:
  Indoor room   : avg ~1-3m  → indoor mode
  Outdoor street: avg ~4-8m  → outdoor mode
"""

import numpy as np
from src import config as cfg
from src.navigation.indoor import IndoorNavigator
from src.navigation.outdoor import OutdoorNavigator


class AutoNavigator:
    """
    Wraps IndoorNavigator and OutdoorNavigator.
    Automatically selects the appropriate mode based on scan context.
    Requires AUTO_SWITCH_FRAMES consecutive agreeing frames before switching.
    """

    def __init__(self, model_path=None):
        self._indoor         = IndoorNavigator()
        self._outdoor        = OutdoorNavigator(model_path)
        self._mode           = "indoor"   # safer cold-start default
        self._pending_mode   = None       # candidate mode being confirmed
        self._pending_count  = 0          # consecutive frames in candidate mode

    @property
    def mode(self):
        return self._mode

    def check(self, pts_array):
        """
        Auto-select mode, run appropriate navigator, return alerts.
        pts_array: Nx2 numpy array of (x, y) in metres.
        Returns (mode_string, alerts_list).
        """
        if len(pts_array) > 0:
            dists    = np.sqrt(pts_array[:, 0]**2 + pts_array[:, 1]**2)
            avg_dist = float(np.mean(dists))
            self._update_mode(avg_dist)

        if self._mode == "outdoor":
            alerts = self._outdoor.check(pts_array)
        else:
            alerts = self._indoor.check(pts_array)

        return self._mode, alerts

    def _update_mode(self, avg_dist):
        """
        Vote on mode switch. Only commit after AUTO_SWITCH_FRAMES
        consecutive frames voting for the same new mode.
        """
        if self._mode == "indoor" and avg_dist > cfg.AUTO_SWITCH_DIST:
            candidate = "outdoor"
        elif self._mode == "outdoor" and avg_dist < cfg.AUTO_SWITCH_DIST * 0.8:
            candidate = "indoor"
        else:
            # Current mode is still appropriate — reset any pending switch
            self._pending_mode  = None
            self._pending_count = 0
            return

        if candidate == self._pending_mode:
            self._pending_count += 1
        else:
            # New candidate — start fresh count
            self._pending_mode  = candidate
            self._pending_count = 1

        if self._pending_count >= cfg.AUTO_SWITCH_FRAMES:
            print(f"[AutoNav] → {candidate.upper()} mode  "
                  f"(avg dist {avg_dist:.1f}m, "
                  f"{self._pending_count} frames)")
            self._mode          = candidate
            self._pending_mode  = None
	    self._pending_count = 0

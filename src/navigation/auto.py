"""
navigation/auto.py
──────────────────
Automatic mode switching between indoor and outdoor navigation.

Logic:
  Computes average distance of all scan points each frame.
  If avg distance > AUTO_SWITCH_DIST → switch to outdoor (YOLO) mode.
  If avg distance < AUTO_SWITCH_DIST * 0.8 → switch to indoor (zone) mode.
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
    """

    def __init__(self, model_path=None):
        self._indoor  = IndoorNavigator()
        self._outdoor = OutdoorNavigator(model_path)
        # Start in indoor mode (safer default — avoids cold YOLO inference
        # on first frame when environment is unknown)
        self._mode    = "indoor"

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
            dists    = np.sqrt(pts_array[:,0]**2 + pts_array[:,1]**2)
            avg_dist = float(np.mean(dists))

            prev = self._mode
            if avg_dist > cfg.AUTO_SWITCH_DIST and self._mode == "indoor":
                self._mode = "outdoor"
                print(f"[AutoNav] → OUTDOOR mode  (avg dist {avg_dist:.1f}m)")
            elif avg_dist < cfg.AUTO_SWITCH_DIST * 0.8 and self._mode == "outdoor":
                self._mode = "indoor"
                print(f"[AutoNav] → INDOOR mode   (avg dist {avg_dist:.1f}m)")

        if self._mode == "outdoor":
            alerts = self._outdoor.check(pts_array)
        else:
            alerts = self._indoor.check(pts_array)

        return self._mode, alerts

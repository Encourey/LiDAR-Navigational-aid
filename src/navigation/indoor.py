"""
navigation/indoor.py
────────────────────
Zone-based proximity detection for indoor navigation.
Requires NO machine learning — pure geometry on LiDAR point distances.
Works on any surface, any object, any room without retraining.

Zone layout (top-down view, sensor at centre):
  ┌─────────────────────────────────┐
  │         hard-left  hard-right   │
  │   left  front-left │ front-right  right │
  │         front                   │
  │           ● (sensor)            │
  └─────────────────────────────────┘

Each zone spans an angular range and a distance threshold.
If point density in a zone exceeds INDOOR_DENSITY_THRESH,
an alert is raised with the zone direction and urgency.
"""

import numpy as np
from src import config as cfg


# Zone definitions: (direction_label, angle_min, angle_max)
# Angles are from np.arctan2(y, x) — range -180 to +180 degrees
ZONES = [
    ("ahead",      -30,    30),
    ("left",        30,    90),
    ("right",      -90,   -30),
    ("hard-left",   90,   150),
    ("hard-right", -150,  -90),
]


class IndoorNavigator:
    """Detects proximity obstacles using LiDAR point density per zone."""

    def check(self, pts_array):
        """
        Check all zones for obstacle proximity.
        pts_array: Nx2 numpy array of (x, y) in metres.
        Returns list of (urgency, "obstacle", direction, point_count) tuples.
        """
        if len(pts_array) == 0:
            return []

        x    = pts_array[:, 0]
        y    = pts_array[:, 1]
        dist = np.sqrt(x**2 + y**2)
        ang  = np.degrees(np.arctan2(y, x))   # -180 to +180

        alerts = []
        for direction, a_min, a_max in ZONES:
            in_zone = (dist < cfg.INDOOR_WARN_DIST) & (ang >= a_min) & (ang < a_max)
            count   = int(np.sum(in_zone))

            if count >= cfg.INDOOR_DENSITY_THRESH:
                zone_dists = dist[in_zone]
                closest    = float(np.min(zone_dists))
                # WARNING if within half the warning distance
                urgency = "WARNING" if closest < cfg.INDOOR_WARN_DIST * 0.5 \
                          else "caution"
                alerts.append((urgency, "obstacle", direction, count))

        # Sort: WARNING first
        alerts.sort(key=lambda a: a[0] != "WARNING")
        return alerts

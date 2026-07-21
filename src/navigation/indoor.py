"""
navigation/indoor.py
────────────────────
Zone-based proximity detection for indoor navigation.
Requires NO machine learning — pure geometry on LiDAR point distances.

Zone layout (top-down view, sensor at centre, 0° = forward = +X):

         hard-left (-150 to -90)    hard-right (90 to 150)
              left (-90 to -30)     right (30 to 90)
                      ahead (-30 to 30)
                         ● sensor

Angles from np.arctan2(y, x) — range -180° to +180°
  0°   = right (+Y direction in image)
  90°  = forward (+X direction)
  -90° = backward (-X direction)
  ±180° = left (-Y direction)

Note: arctan2(y,x) with x=forward, y=left means:
  ahead zone  = high positive X, small Y → angles near 0° won't work
  Use atan2(y, x) correctly:
    ahead      → x large positive, y small → angle near 0°... 

Actually rewritten to use bearing from +X axis properly.
"""

import numpy as np
from src import config as cfg

# Zone definitions: (label, angle_min, angle_max)
# arctan2(y, x): 0° = +X (forward), 90° = +Y (left), -90° = -Y (right)
ZONES = [
    ("ahead",      -25,    25),
    ("left",        25,    70),
    ("right",      -70,   -25),
    ("hard-left",   70,   100),
    ("hard-right", -100,  -70),
]

class IndoorNavigator:
    """Detects proximity obstacles using LiDAR point density per zone."""

    def check(self, pts_array):
        """
        Check all zones for obstacle proximity.
        pts_array: Nx2 numpy array of (x, y) in metres.
          x = forward, y = left (positive = left, negative = right)
        Returns list of (urgency, "obstacle", direction, closest_dist) tuples.
        """
        if len(pts_array) == 0:
            return []

        x    = pts_array[:, 0]
        y    = pts_array[:, 1]
        dist = np.sqrt(x**2 + y**2)
        ang  = np.degrees(np.arctan2(y, x))   # 0°=forward, 90°=left, -90°=right

        alerts = []
        for direction, a_min, a_max in ZONES:
            in_zone = (dist < cfg.INDOOR_WARN_DIST) & (ang >= a_min) & (ang < a_max)
            count   = int(np.sum(in_zone))
            if count >= cfg.INDOOR_DENSITY_THRESH:
                zone_dists = dist[in_zone]
                zone_dists = zone_dists[zone_dists > 0.05]  # filter zero/noise
                if len(zone_dists) == 0:
                    continue
                closest = float(np.min(zone_dists)) + cfg.LIDAR_DISTANCE_OFFSET
                urgency = "WARNING" if closest < cfg.INDOOR_WARN_DIST * 0.5 \
                          else "caution"
                alerts.append((urgency, "obstacle", direction, closest))

        alerts.sort(key=lambda a: a[0] != "WARNING")
        return alerts


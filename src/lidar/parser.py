"""
lidar/parser.py
───────────────
Parses YDLIDAR X3 YB scan packets into numpy point arrays.

The X3 delivers packets continuously. Each packet covers a segment of the
360° sweep. A full rotation is complete when angle_end wraps past 0°
(i.e. angle_end < prev_end and prev_end > 300).

We collect packets until we detect this wrap, guaranteeing a complete 360°
scan every call — no partial scans from fixed-duration collection.

Angle interpolation:
  Each packet gives angle_start and angle_end, and N distance samples
  between them. We linearly interpolate the angle for each sample.

Coordinate system:
  x = forward (positive away from sensor, 0° direction)
  y = left    (negative = right)
"""

import numpy as np
from src import config as cfg


class LidarParser:
    """Converts YDLIDAR X3 packet stream into numpy scan arrays."""

    def __init__(self, reader):
        self._reader = reader

    def collect_scan(self):
        """
        Collect one complete 360° scan.
        Reads packets until the angle wraps (end < start crossing 0°),
        indicating a full rotation has been received.

        Filters by MAX_DIST_MM and discards zero-distance (invalid) readings.
        Returns Nx2 numpy array of (x, y) in metres.
        """
        current   = []   # points accumulating for this rotation
        prev_end  = None
        max_pkts  = 400
        result    = []   # will hold the first complete rotation

        for _ in range(max_pkts):
            pkt = self._reader.read_packet()
            if pkt is None:
                continue

            a_start = pkt["angle_start"]
            a_end   = pkt["angle_end"]
            dists   = pkt["distances"]
            n       = pkt["num_points"]

            if n == 0:
                continue

            wrap_inside  = a_end < a_start and a_start > 300
            wrap_between = prev_end is not None and prev_end > 300 and a_start < 60

            if wrap_inside or wrap_between:
                # Snapshot completed rotation if it has enough points
                if len(current) > 100:
                    result = list(current)
                    break   # done — return this complete rotation
                # Too few points — rotation was partial, discard and restart
                current = []

            self._add_points(current, a_start, a_end, dists, n)
            prev_end = a_end

        if not result:
            result = current   # fallback if wrap never triggered

        return np.array(result, dtype=np.float32) if result else np.empty((0, 2))

    def _add_points(self, points, a_start, a_end, dists, n):
        """
        Interpolate angles across the packet and convert to (x, y).
        Appends (x, y) tuples to `points`. Skips zero/out-of-range distances.
        """
        if a_end < a_start:
            a_end += 360.0

        angles = np.linspace(a_start, a_end, n) if n > 1 else np.array([a_start])

        for angle, dist_mm in zip(angles, dists):
            if dist_mm == 0 or dist_mm > cfg.MAX_DIST_MM:
                continue
            angle_rad = np.deg2rad((angle - cfg.LIDAR_ANGLE_OFFSET) % 360)
            dist_m    = dist_mm / 1000.0
            # YDLIDAR X3 angle 0° = forward, increases clockwise
            # x = forward, y = left (positive left)
            points.append(( dist_m * np.sin(angle_rad),
                            dist_m * np.cos(angle_rad)))


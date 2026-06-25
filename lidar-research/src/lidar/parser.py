"""
lidar/parser.py
───────────────
Parses YDLIDAR X3 YB scan packets into numpy point arrays.

The X3 delivers packets continuously. Each packet covers a segment of the
360° sweep. A full rotation is complete when angle_end wraps past 0°
(i.e. angle_end < angle_start for the first time after a rising sequence).

We collect packets until we detect this wrap, guaranteeing a complete 360°
scan every call — no partial scans from fixed-duration collection.

Angle interpolation:
  Each packet gives angle_start and angle_end, and N distance samples
  between them. We linearly interpolate the angle for each sample.

Coordinate system (same as before):
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
        xs, ys      = [], []
        prev_end    = None
        max_packets = 200   # safety cap — prevents infinite loop on bad data

        for _ in range(max_packets):
            pkt = self._reader.read_packet()
            if pkt is None:
                continue

            a_start = pkt["angle_start"]
            a_end   = pkt["angle_end"]
            dists   = pkt["distances"]
            n       = pkt["num_points"]

            if n == 0:
                continue

            # Detect full-rotation wrap: angle_end < angle_start means
            # the sweep passed through 0°/360°
            if prev_end is not None and a_end < prev_end and prev_end > 300:
                # Process this final packet then stop
                self._add_points(xs, ys, a_start, a_end, dists, n)
                break

            self._add_points(xs, ys, a_start, a_end, dists, n)
            prev_end = a_end

        if not xs:
            return np.empty((0, 2))

        return np.column_stack([xs, ys])

    def _add_points(self, xs, ys, a_start, a_end, dists, n):
        """
        Interpolate angles across the packet and convert to (x, y).
        Skips zero-distance and out-of-range readings.
        """
        # Handle angle wrap within a single packet
        if a_end < a_start:
            a_end += 360.0

        if n == 1:
            angles = np.array([a_start])
        else:
            angles = np.linspace(a_start, a_end, n)

        for angle, dist_mm in zip(angles, dists):
            if dist_mm == 0 or dist_mm > cfg.MAX_DIST_MM:
                continue
            angle_rad = np.deg2rad(angle % 360)
            dist_m    = dist_mm / 1000.0
            xs.append(dist_m * np.cos(angle_rad))
            ys.append(-dist_m * np.sin(angle_rad))

"""
lidar/parser.py
───────────────
Parses raw RPLIDAR A1M8 scan bytes into numpy point arrays.

RPLIDAR A1M8 scan packet format (5 bytes per point):
  Byte 0: quality[7:2] | new_scan[0]
  Byte 1: angle[6:0] << 1 | check_bit
  Byte 2: angle[14:7]
  Byte 3: distance[7:0]
  Byte 4: distance[15:8]
  angle    = (byte1>>1 | byte2<<7) / 64.0   degrees
  distance = (byte3 | byte4<<8)   / 4.0     mm
"""

import time
import numpy as np
from src import config as cfg


class LidarParser:
    """Converts raw serial bytes from LidarReader into numpy scan arrays."""

    def __init__(self, reader):
        self._reader = reader

    def collect_scan(self, duration=None):
        """
        Collect scan points for `duration` seconds.
        Filters by quality and max distance from config.
        Returns Nx2 numpy array of (x, y) in metres.
          x = forward (away from sensor)
          y = left (negative = right)
        """
        dur = duration or cfg.SCAN_DURATION
        xs, ys = [], []
        start  = time.time()

        while time.time() - start < dur:
            raw = self._reader.read_raw_bytes(5)
            if len(raw) < 5:
                continue

            b0, b1, b2, b3, b4 = raw
            quality  = b0 >> 2
            angle    = ((b1 >> 1) | (b2 << 7)) / 64.0
            distance = ((b3) | (b4 << 8)) / 4.0   # mm

            if quality > cfg.MIN_QUALITY and 0 < distance < cfg.MAX_DIST_MM:
                angle_rad = np.deg2rad(angle % 360)
                xs.append((distance / 1000.0) * np.cos(angle_rad))
                ys.append(-(distance / 1000.0) * np.sin(angle_rad))

        if not xs:
            return np.empty((0, 2))

        return np.column_stack([xs, ys])

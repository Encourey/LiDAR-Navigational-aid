"""
threads/pipeline.py
───────────────────
Producer-consumer threading pipeline for the navigation loop.

Architecture:
  Thread A (ScanProducer) — continuously reads LiDAR serial data
    and writes the latest Nx2 point array to a shared slot.
  Thread B (main) — reads the latest scan snapshot and runs
    BEV projection + navigation inference independently.

This overlap means YOLO inference time does NOT add to scan
collection time. Result: ~10 FPS vs ~5.6 FPS sequential (v1).

Design choice — shared slot vs queue:
  We use a single shared slot (latest_pts) rather than a Queue
  because for navigation we only care about the most recent scan.
  Stale frames in a queue would increase latency. If the main
  thread is slower than the scan rate, it simply processes the
  most recent available frame and skips intermediate ones.
"""

import threading
import numpy as np
from src.lidar.reader import LidarReader
from src.lidar.parser import LidarParser
from src import config as cfg


class ScanProducer:
    """
    Background thread that continuously collects LiDAR scan frames.
    Exposes the latest scan via get_latest().
    """

    def __init__(self):
        self._reader   = LidarReader()
        self._parser   = LidarParser(self._reader)
        self._latest   = None
        self._lock     = threading.Lock()
        self._active   = False
        self._thread   = None

    def start(self):
        """Connect to LiDAR and start background scan thread."""
        self._reader.connect()
        self._active = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="ScanProducer"
        )
        self._thread.start()
        print("[ScanProducer] Started.")

    def _run(self):
        while self._active:
            pts = self._parser.collect_scan()
            if len(pts) >= 20:
                with self._lock:
                    self._latest = pts

    def get_latest(self):
        """Return the most recent scan array (Nx2) or None."""
        with self._lock:
            return self._latest

    def stop(self):
        """Stop the background thread and disconnect LiDAR."""
        self._active = False
        if self._thread:
            self._thread.join(timeout=2)
        self._reader.disconnect()
        print("[ScanProducer] Stopped.")

"""
navigation/outdoor.py
─────────────────────
Wraps the YOLO-OBB Detector for outdoor navigation alerts.
Handles BEV projection + inference call + alert formatting.
"""

from src.vision.bev import BEVProjector
from src.vision.detector import Detector
from src import config as cfg


class OutdoorNavigator:
    """
    Outdoor navigation using YOLO-OBB on BEV images.
    Detects cars, pedestrians, cyclists from LiDAR point clouds.
    """

    def __init__(self, model_path=None):
        self._bev      = BEVProjector()
        self._detector = Detector(model_path)

    def check(self, pts_array):
        """
        Project points to BEV, run YOLO inference, return alerts.
        pts_array: Nx2 numpy array of (x, y) in metres.
        Returns list of (urgency, class_name, direction, confidence).
        """
        bev    = self._bev.project(pts_array)
        alerts = self._detector.predict(bev)
        return alerts

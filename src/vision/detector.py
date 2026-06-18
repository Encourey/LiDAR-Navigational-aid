"""
vision/detector.py
──────────────────
Wraps Ultralytics YOLO-OBB inference for BEV images.
Uses NCNN format model for ARM CPU inference on Raspberry Pi 5.

Model: YOLOv8n-OBB trained on KITTI BEV dataset
  mAP50      : 0.832
  mAP50-95   : 0.519
  Car        : 0.973
  Pedestrian : 0.734
  Cyclist    : 0.787
  Inference  : ~100-177ms on Pi 5 CPU (NCNN, no GPU)

Detection output format:
  Each alert is a tuple: (urgency, class_name, direction, confidence)
  urgency   : "WARNING" (close) | "caution" (mid-range)
  direction : "left" | "ahead" | "right"
"""

import numpy as np
from ultralytics import YOLO
from src import config as cfg


class Detector:
    """Loads YOLO-OBB model and runs inference on BEV images."""

    def __init__(self, model_path=None):
        path = model_path or cfg.MODEL_PATH
        print(f"[Detector] Loading model from {path}...")
        self._model = YOLO(path, task="obb")
        self._warmup()
        print("[Detector] Model ready.")

    def _warmup(self):
        """
        Run one dummy inference to pre-compile NCNN layers.
        Eliminates the 3-4 second spike on the first real frame.
        """
        dummy = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.uint8)
        self._model.predict(source=dummy, imgsz=cfg.IMG_SIZE, verbose=False)

    def predict(self, bev_image):
        """
        Run YOLO-OBB inference on a BEV image.
        Returns list of (urgency, class_name, direction, confidence) tuples,
        sorted with WARNING alerts first.
        """
        results = self._model.predict(
            source  = bev_image,
            imgsz   = cfg.IMG_SIZE,
            conf    = cfg.CONF_THRESH,
            verbose = False,
        )

        alerts = []
        for r in results:
            if r.obb is None:
                continue
            for i in range(len(r.obb)):
                conf = float(r.obb.conf[i])
                if conf < cfg.CONF_THRESH:
                    continue

                cls  = int(r.obb.cls[i])
                xyxy = r.obb.xyxy[i].cpu().numpy()

                # Normalised centre position in BEV image
                cx = ((xyxy[0] + xyxy[2]) / 2) / cfg.IMG_SIZE
                cy = ((xyxy[1] + xyxy[3]) / 2) / cfg.IMG_SIZE

                # Direction from horizontal position
                if cx < 0.35:
                    direction = "left"
                elif cx > 0.65:
                    direction = "right"
                else:
                    direction = "ahead"

                # Urgency from vertical position (cy=0 far, cy=1 near)
                if cy > 0.70:
                    urgency = "WARNING"
                elif cy > 0.45:
                    urgency = "caution"
                else:
                    urgency = None   # too far, ignore

                if urgency:
                    alerts.append((
                        urgency,
                        cfg.CLASS_NAMES.get(cls, "object"),
                        direction,
                        conf,
                    ))

        # Sort: WARNING first, then by confidence
        alerts.sort(key=lambda a: (a[0] != "WARNING", -a[3]))
        return alerts

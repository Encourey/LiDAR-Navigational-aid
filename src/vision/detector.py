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

OBB attribute fix:
  NCNN-exported OBB models return r.obb.xywhr (cx, cy, w, h, rotation)
  rather than axis-aligned r.obb.xyxy. Using xyxy on an NCNN model can
  return None or incorrect values depending on Ultralytics version.
  We use xywhr throughout and derive cx/cy directly.
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

            # Use xywhr — works correctly on NCNN-exported OBB models.
            # xywhr shape: (N, 5) → [cx, cy, w, h, rotation_rad]
            xywhr = r.obb.xywhr
            confs  = r.obb.conf
            clses  = r.obb.cls

            if xywhr is None or len(xywhr) == 0:
                continue

            xywhr_np = xywhr.cpu().numpy()
            conf_np  = confs.cpu().numpy()
            cls_np   = clses.cpu().numpy()

            for i in range(len(xywhr_np)):
                conf = float(conf_np[i])
                if conf < cfg.CONF_THRESH:
                    continue

                cx = float(xywhr_np[i, 0]) / cfg.IMG_SIZE   # normalised 0–1
                cy = float(xywhr_np[i, 1]) / cfg.IMG_SIZE   # normalised 0–1
                cls = int(cls_np[i])

                # Direction from horizontal position in BEV image
                if cx < 0.35:
                    direction = "left"
                elif cx > 0.65:
                    direction = "right"
                else:
                    direction = "ahead"

                # Urgency from vertical position (cy=0 far, cy=1 near sensor)
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

        # Sort: WARNING first, then by confidence descending
        alerts.sort(key=lambda a: (a[0] != "WARNING", -a[3]))
	return alerts

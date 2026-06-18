"""
training/train.py
─────────────────
YOLOv8n-OBB training on KITTI BEV dataset.
Run on training machine (Windows/Linux with NVIDIA GPU), NOT on Pi.

Results achieved:
  mAP50=0.832  mAP50-95=0.519
  Car=0.973  Pedestrian=0.734  Cyclist=0.787
  100 epochs in ~1.7 hrs on RTX 4060

Usage: python -m src.training.train   (from research/ root)
"""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n-obb.pt")

    results = model.train(
        data     = "data/bev_kitti.yaml",
        epochs   = 100,
        imgsz    = 320,
        batch    = 8,        # reduce to 4 if CUDA OOM
        device   = 0,
        patience = 20,
        workers  = 4,
        project  = "experiments",
        name     = "bev_obb_v1",
    )

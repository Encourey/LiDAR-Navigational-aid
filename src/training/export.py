"""
training/export.py
──────────────────
Exports trained best.pt to NCNN format for Raspberry Pi inference.
NCNN is the fastest inference runtime for ARM CPUs (no GPU needed).

Note: int8=True is not supported for NCNN via Ultralytics export.
INT8 quantisation must be done separately on-device using NCNN tools.

Usage: python -m src.training.export   (from research/ root)
"""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("experiments/bev_obb_v1/weights/best.pt")
    model.export(format="ncnn", imgsz=320)
    print("Exported. Copy best_ncnn_model/ to Pi ~/research/models/")

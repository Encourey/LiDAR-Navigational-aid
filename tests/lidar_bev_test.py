"""
tests/lidar_bev_test.py
───────────────────────
Tests the full YDLIDAR X3 → BEV image pipeline.
Collects one complete 360° scan, saves a fixed 8m×8m BEV
centred on the sensor (orange dot should be in the middle).

Fixes vs previous version:
  - Fixed aspect ratio: always square output (sensor at centre)
  - Prints raw angle/distance stats so you can verify coordinate convention
  - Saves both the auto-range (debug) and fixed-range (navigation) BEV

Usage: python tests/lidar_bev_test.py
"""

from datetime import datetime
import numpy as np
import cv2
import os
import sys
sys.path.insert(0, ".")
from src import config as cfg
from src.lidar.reader import LidarReader
from src.lidar.parser import LidarParser

os.makedirs("outputs", exist_ok=True)

reader = LidarReader()
parser = LidarParser(reader)

print(f"Connecting to {cfg.LIDAR_PORT} @ {cfg.BAUD_RATE}...")
reader.connect()
print("Collecting one full 360° scan...")

pts = parser.collect_scan()
reader.disconnect()

if len(pts) == 0:
    print("No points collected — check wiring and baud rate.")
    sys.exit(1)

xs = pts[:, 0]
ys = pts[:, 1]
dists = np.sqrt(xs**2 + ys**2)

print(f"\nPoints collected : {len(pts)}")
print(f"X range          : {xs.min():.2f} to {xs.max():.2f} m")
print(f"Y range          : {ys.min():.2f} to {ys.max():.2f} m")
print(f"Distance range   : {dists.min():.2f} to {dists.max():.2f} m")
print(f"Mean distance    : {dists.mean():.2f} m")

# ── Fixed-range BEV (what navigation actually uses) ────────────────────────
# Sensor at centre. 8m range each direction. Square output.
RES  = 0.05   # m per pixel
HALF = 8.0    # metres
SIZE = int(HALF * 2 / RES)   # 320 px

bev_fixed = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
for x, y in zip(xs, ys):
    if abs(x) > HALF or abs(y) > HALF:
        continue
    row = int((HALF - x) / RES)
    col = int((y + HALF) / RES)
    row = np.clip(row, 0, SIZE - 1)
    col = np.clip(col, 0, SIZE - 1)
    bev_fixed[row, col] = (0, 255, 128)

# Ego dot at exact centre
cx = SIZE // 2
cy = SIZE // 2
cv2.circle(bev_fixed, (cx, cy), 6, (0, 100, 255), -1)
cv2.arrowedLine(bev_fixed, (cx, cy), (cx, cy - 20), (255, 255, 255), 1)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
filename_fixed = f"outputs/bev_fixed_{ts}.png"
cv2.imwrite(filename_fixed, bev_fixed)
print(f"\nSaved {filename_fixed} ({SIZE}x{SIZE} px, sensor at centre)")
print("White arrow = forward (+X direction). Check this matches physical forward.")

# ── Auto-range BEV (debug — shows all points regardless of range) ──────────
m   = 0.5
RES2 = 0.03
x_min, x_max = xs.min() - m, xs.max() + m
y_min, y_max = ys.min() - m, ys.max() + m
img_h = int((x_max - x_min) / RES2)
img_w = int((y_max - y_min) / RES2)
bev_auto = np.zeros((img_h, img_w, 3), dtype=np.uint8)

for x, y in zip(xs, ys):
    r = np.clip(int((x_max - x) / RES2), 0, img_h - 1)
    c = np.clip(int((y - y_min) / RES2), 0, img_w - 1)
    bev_auto[r, c] = (0, 255, 128)

er = np.clip(int((x_max - 0) / RES2), 0, img_h - 1)
ec = np.clip(int((0 - y_min) / RES2), 0, img_w - 1)
cv2.circle(bev_auto, (ec, er), 6, (0, 100, 255), -1)
bev_auto = cv2.resize(bev_auto, (320, 320))
filename_auto = f"outputs/bev_auto_{ts}.png"
cv2.imwrite(filename_auto, bev_auto)
print(f"Saved {filename_auto} (auto-range debug view)")


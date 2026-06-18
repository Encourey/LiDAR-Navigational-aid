"""
tests/lidar_bev_test.py
───────────────────────
Tests the full LiDAR → BEV image pipeline.
Collects 2 seconds of scan data, auto-ranges BEV window,
saves bev_auto.png in outputs/.

Usage: python tests/lidar_bev_test.py
"""

import serial
import time
import numpy as np
import cv2
import os
import sys
sys.path.insert(0, ".")
from src import config as cfg

os.makedirs("outputs", exist_ok=True)

ser = serial.Serial(cfg.LIDAR_PORT, baudrate=cfg.BAUD_RATE, timeout=2)
ser.write(b"\xA5\x25"); time.sleep(0.5)
ser.reset_input_buffer(); time.sleep(0.3)
ser.write(b"\xA5\x20"); ser.read(7); time.sleep(0.2)

points = []
start  = time.time()
while time.time() - start < 2.0:
    raw = ser.read(5)
    if len(raw) < 5: continue
    b0,b1,b2,b3,b4 = raw
    quality  = b0 >> 2
    angle    = ((b1>>1)|(b2<<7)) / 64.0
    distance = ((b3)|(b4<<8)) / 4.0
    if quality > 0 and 0 < distance < cfg.MAX_DIST_MM:
        ar = np.deg2rad(angle % 360)
        points.append(((distance/1000.0)*np.cos(ar),
                       -(distance/1000.0)*np.sin(ar)))

ser.write(b"\xA5\x25"); ser.close()

print(f"Points collected: {len(points)}")
xs = [p[0] for p in points]
ys = [p[1] for p in points]
print(f"X: {min(xs):.2f} to {max(xs):.2f} m")
print(f"Y: {min(ys):.2f} to {max(ys):.2f} m")

m   = 0.5
RES = 0.03
x_min,x_max = min(xs)-m, max(xs)+m
y_min,y_max = min(ys)-m, max(ys)+m
img_h = int((x_max-x_min)/RES)
img_w = int((y_max-y_min)/RES)
bev   = np.zeros((img_h,img_w,3),dtype=np.uint8)

for x,y in points:
    r = np.clip(int((x_max-x)/RES), 0, img_h-1)
    c = np.clip(int((y-y_min)/RES), 0, img_w-1)
    bev[r,c] = (0,255,128)

er = np.clip(int((x_max-0)/RES), 0, img_h-1)
ec = np.clip(int((0-y_min)/RES), 0, img_w-1)
cv2.circle(bev,(ec,er),6,(0,100,255),-1)
bev = cv2.resize(bev,(320,320))
cv2.imwrite("outputs/bev_auto.png", bev)
print("Saved outputs/bev_auto.png")

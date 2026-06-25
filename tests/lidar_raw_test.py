"""
tests/lidar_raw_test.py
───────────────────────
Direct serial communication test for YDLIDAR X3 YB.
Bypasses all library code — talks raw UART bytes.
Run this first to confirm the LiDAR is detected and responding.

Expected output:
  Connected on /dev/ttyUSB0 @ 128000 baud
  Received scan packet — num_points: 40  angle: 0.00° → 22.50°
  distances (mm): [823, 841, 856, ...]
  Test passed.

Usage: python tests/lidar_raw_test.py
"""

import serial
import time
import sys
sys.path.insert(0, ".")
from src import config as cfg

print(f"Connecting to {cfg.LIDAR_PORT} @ {cfg.BAUD_RATE} baud...")
ser = serial.Serial(
    cfg.LIDAR_PORT,
    baudrate=cfg.BAUD_RATE,
    timeout=2,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)

# Stop any existing scan, flush, start
ser.write(b"\xA5\x65")
time.sleep(0.1)
ser.reset_input_buffer()
time.sleep(0.1)
ser.write(b"\xA5\x60")
time.sleep(0.3)
print(f"Connected on {cfg.LIDAR_PORT} @ {cfg.BAUD_RATE} baud")

# Try to read one packet by syncing to 0xAA 0x55 header
print("Waiting for scan packet...")
found = False
for _ in range(5000):
    b = ser.read(1)
    if not b:
        break
    if b[0] == 0xAA:
        b2 = ser.read(1)
        if b2 and b2[0] == 0x55:
            # Read the rest: ct(1) + num_samples(1) + FSA(2) + LSA(2) = 6
            hdr = ser.read(6)
            if len(hdr) < 6:
                print("Short header read — check wiring/baud rate.")
                break

            ct          = hdr[0]
            num_samples = hdr[1]
            fsa_raw     = hdr[2] | (hdr[3] << 8)
            lsa_raw     = hdr[4] | (hdr[5] << 8)
            angle_start = ((fsa_raw >> 1) & 0x7FFF) / 64.0
            angle_end   = ((lsa_raw  >> 1) & 0x7FFF) / 64.0

            tail = ser.read(2 + num_samples * 2)
            distances = []
            for i in range(num_samples):
                lo = tail[2 + i * 2]
                hi = tail[2 + i * 2 + 1]
                distances.append((lo | (hi << 8)) >> 2)

            print(f"Received scan packet — num_points: {num_samples}  "
                  f"angle: {angle_start:.2f}° → {angle_end:.2f}°")
            print(f"distances (mm): {distances[:10]}{'...' if num_samples > 10 else ''}")
            print("Test passed.")
            found = True
            break

if not found:
    print("No valid packet received — check port, baud rate, and wiring.")

ser.write(b"\xA5\x65")
ser.close()

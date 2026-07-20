"""
tests/lidar_raw_test.py
───────────────────────
Dumps raw YDLIDAR X3 packet data — angles, distances, point counts.
Run this to diagnose wrap detection and distance scaling issues.

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

ser.write(b"\xA5\x65")
time.sleep(0.1)
ser.reset_input_buffer()
time.sleep(0.1)
ser.write(b"\xA5\x60")
time.sleep(0.3)
print("Collecting 30 packets...\n")

packets_collected = 0
all_angles_start  = []
all_angles_end    = []
all_distances     = []

for attempt in range(10000):
    if packets_collected >= 30:
        break

    b = ser.read(1)
    if not b:
        continue
    if b[0] != 0xAA:
        continue
    b2 = ser.read(1)
    if not b2 or b2[0] != 0x55:
        continue

    hdr = ser.read(6)
    if len(hdr) < 6:
        continue

    ct          = hdr[0]
    num_samples = hdr[1]
    fsa_raw     = hdr[2] | (hdr[3] << 8)
    lsa_raw     = hdr[4] | (hdr[5] << 8)
    angle_start = ((fsa_raw >> 1) & 0x7FFF) / 64.0
    angle_end   = ((lsa_raw  >> 1) & 0x7FFF) / 64.0

    tail = ser.read(2 + num_samples * 2)
    if len(tail) < 2 + num_samples * 2:
        continue

    # Decode distances — try both with and without >> 2 shift
    dists_shifted   = []
    dists_raw       = []
    for i in range(num_samples):
        lo = tail[2 + i * 2]
        hi = tail[2 + i * 2 + 1]
        raw_val = lo | (hi << 8)
        dists_raw.append(raw_val)
        dists_shifted.append(raw_val >> 2)

    all_angles_start.append(angle_start)
    all_angles_end.append(angle_end)
    all_distances.extend(dists_shifted)

    print(f"Pkt {packets_collected:02d} | "
          f"angle {angle_start:6.2f}° → {angle_end:6.2f}° | "
          f"n={num_samples:2d} | "
          f"dist(>>2) {min(dists_shifted):4d}–{max(dists_shifted):4d} mm | "
          f"dist(raw) {min(dists_raw):5d}–{max(dists_raw):5d}")

    packets_collected += 1

print(f"\n{'='*60}")
print(f"Angle start range : {min(all_angles_start):.2f}° → {max(all_angles_start):.2f}°")
print(f"Angle end range   : {min(all_angles_end):.2f}° → {max(all_angles_end):.2f}°")
valid = [d for d in all_distances if 0 < d < 8000]
if valid:
    print(f"Valid distances   : {min(valid)}mm – {max(valid)}mm  (mean {sum(valid)//len(valid)}mm)")
    print(f"Expected room     : walls should be ~1000–4000mm away")
    print(f"\nIf dist(raw) values look more realistic than dist(>>2),")
    print(f"remove the >> 2 shift in reader.py read_packet()")
else:
    print("No valid distances found — check wiring")

ser.write(b"\xA5\x65")
ser.close()


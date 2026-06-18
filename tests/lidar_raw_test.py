"""
tests/lidar_raw_test.py
───────────────────────
Direct serial communication test for RPLIDAR A1M8.
Bypasses all library code — talks raw UART bytes.
Run this first to confirm the LiDAR is detected and responding.

Expected output:
  Descriptor hex: aa55...
  Model: 40  Firmware: 1.29  Hardware: 7

Usage: python tests/lidar_raw_test.py
"""

import serial
import time
import sys
sys.path.insert(0, ".")
from src import config as cfg

ser = serial.Serial(cfg.LIDAR_PORT, baudrate=cfg.BAUD_RATE, timeout=1)

ser.write(b"\xA5\x25")
time.sleep(0.5)
ser.reset_input_buffer()
time.sleep(0.5)

ser.write(b"\xA5\x50")
time.sleep(0.2)
descriptor = ser.read(7)
print(f"Descriptor hex: {descriptor.hex()}")

if len(descriptor) >= 2:
    if descriptor[0] == 0xA5 and descriptor[1] == 0x5A:
        data = ser.read(20)
        print(f"Model    : {data[0]}")
        print(f"Firmware : {data[2]}.{data[1]}")
        print(f"Hardware : {data[3]}")
        print(f"Serial   : {data[4:].hex()}")
    elif descriptor[0] == 0xAA and descriptor[1] == 0x55:
        print("Sensor mid-scan — stopping and retrying...")
        ser.write(b"\xA5\x25")
        time.sleep(1)
        ser.reset_input_buffer()
        ser.write(b"\xA5\x50")
        time.sleep(0.2)
        descriptor = ser.read(7)
        data = ser.read(20)
        print(f"Model: {data[0]}  FW: {data[2]}.{data[1]}  HW: {data[3]}")
    else:
        print(f"Unexpected bytes: {descriptor.hex()}")

ser.write(b"\xA5\x25")
ser.close()
print("Done.")

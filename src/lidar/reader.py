"""
lidar/reader.py
───────────────
Manages the serial connection to the RPLIDAR A1M8.
Uses raw UART bytes directly — bypasses rplidar-roboticia library
which is incompatible with Python 3.13 (get_health() return format changed).

RPLIDAR A1M8 serial protocol:
  Commands are 2-byte sequences: 0xA5 <cmd>
    0xA5 0x25 — STOP scan
    0xA5 0x20 — START scan
    0xA5 0x50 — GET_INFO
  Response descriptor: 7 bytes (0xA5 0x5A <dsize x3> <mode> <dtype>)
  Scan packet: 5 bytes per point
"""

import serial
import time
from src import config as cfg


class LidarReader:
    """Opens and manages the RPLIDAR A1M8 serial connection."""

    CMD_STOP  = b"\xA5\x25"
    CMD_SCAN  = b"\xA5\x20"
    CMD_INFO  = b"\xA5\x50"
    DESC_LEN  = 7

    def __init__(self, port=None, baud=None):
        self.port = port or cfg.LIDAR_PORT
        self.baud = baud or cfg.BAUD_RATE
        self._ser = None

    def connect(self):
        """Open serial port and start the LiDAR scan."""
        self._ser = serial.Serial(self.port, baudrate=self.baud, timeout=2)
        self._ser.write(self.CMD_STOP)
        time.sleep(0.5)
        self._ser.reset_input_buffer()
        time.sleep(0.3)
        self._ser.write(self.CMD_SCAN)
        self._ser.read(self.DESC_LEN)   # discard scan descriptor
        time.sleep(0.2)
        print(f"[LidarReader] Connected on {self.port}")

    def get_info(self):
        """
        Query sensor firmware/model info.
        Returns dict with model, firmware, hardware, serial keys.
        Handles case where sensor is mid-scan by stopping first.
        """
        self._ser.write(self.CMD_STOP)
        time.sleep(0.5)
        self._ser.reset_input_buffer()
        self._ser.write(self.CMD_INFO)
        time.sleep(0.2)
        desc = self._ser.read(self.DESC_LEN)
        if len(desc) < 2:
            return None
        # If sensor was mid-scan, stop and retry
        if desc[0] == 0xAA and desc[1] == 0x55:
            self._ser.write(self.CMD_STOP)
            time.sleep(1)
            self._ser.reset_input_buffer()
            self._ser.write(self.CMD_INFO)
            time.sleep(0.2)
            desc = self._ser.read(self.DESC_LEN)
        data = self._ser.read(20)
        return {
            "model":    data[0],
            "firmware": f"{data[2]}.{data[1]}",
            "hardware": data[3],
            "serial":   data[4:].hex(),
        }

    def read_raw_bytes(self, n=5):
        """Read n raw bytes from the serial buffer."""
        return self._ser.read(n)

    def disconnect(self):
        """Stop scan and close serial port."""
        if self._ser and self._ser.is_open:
            self._ser.write(self.CMD_STOP)
            time.sleep(0.1)
            self._ser.close()
            print("[LidarReader] Disconnected.")

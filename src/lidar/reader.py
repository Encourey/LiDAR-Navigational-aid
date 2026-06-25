"""
lidar/reader.py
───────────────
Manages the serial connection to the YDLIDAR X3 YB.
Uses raw UART bytes directly via the YDLIDAR serial protocol.

YDLIDAR X3 serial protocol:
  Commands are 2-byte sequences: 0xA5 <cmd>
    0xA5 0x65 — STOP scan
    0xA5 0x60 — START scan
    0xA5 0x90 — GET_INFO (device info)

  Scan packet structure (variable length):
    Header:     0xAA 0x55
    Packet type: 0x00 (point cloud)
    Num points: 1 byte  (N points in this packet, typically 40)
    FSA:        2 bytes (first sample angle, Q6 fixed point)
    LSA:        2 bytes (last sample angle, Q6 fixed point)
    CS:         2 bytes (checksum)
    Data:       N * 2 bytes (distance, each 2 bytes little-endian, in mm)

  Baud rate: 115200
"""

import serial
import time
from src import config as cfg


class LidarReader:
    """Opens and manages the YDLIDAR X3 YB serial connection."""

    CMD_STOP  = b"\xA5\x65"
    CMD_SCAN  = b"\xA5\x60"
    CMD_INFO  = b"\xA5\x90"

    PKT_HEADER_1 = 0xAA
    PKT_HEADER_2 = 0x55

    def __init__(self, port=None, baud=None):
        self.port = port or cfg.LIDAR_PORT
        self.baud = baud or cfg.BAUD_RATE
        self._ser = None

    def connect(self):
        """Open serial port and start the LiDAR scan."""
        self._ser = serial.Serial(
            self.port,
            baudrate=self.baud,
            timeout=2,
            # YDLIDAR requires these explicitly
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        # Stop any existing scan, flush, then start
        self._ser.write(self.CMD_STOP)
        time.sleep(0.1)
        self._ser.reset_input_buffer()
        time.sleep(0.1)
        self._ser.write(self.CMD_SCAN)
        time.sleep(0.3)   # allow motor to spin up
        print(f"[LidarReader] Connected on {self.port} @ {self.baud} baud")

    def get_info(self):
        """
        Query sensor device info.
        Returns dict with model, firmware, hardware, serial keys.
        Stops scan first, queries, then restarts.
        """
        self._ser.write(self.CMD_STOP)
        time.sleep(0.1)
        self._ser.reset_input_buffer()

        self._ser.write(self.CMD_INFO)
        time.sleep(0.1)

        # Response: 0xA5 0x5A + 5 bytes header + 20 bytes payload
        header = self._ser.read(7)
        if len(header) < 7 or header[0] != 0xA5 or header[1] != 0x5A:
            print(f"[LidarReader] get_info: unexpected header {header.hex()}")
            self._ser.write(self.CMD_SCAN)
            return None

        data = self._ser.read(20)
        info = {
            "model":    data[0] if len(data) > 0 else None,
            "firmware": f"{data[2]}.{data[1]}" if len(data) > 2 else None,
            "hardware": data[3] if len(data) > 3 else None,
            "serial":   data[4:].hex() if len(data) > 4 else None,
        }

        # Restart scan
        self._ser.write(self.CMD_SCAN)
        time.sleep(0.3)
        return info

    def read_raw_bytes(self, n):
        """Read exactly n raw bytes from the serial buffer."""
        return self._ser.read(n)

    def read_packet(self):
        """
        Read one YDLIDAR X3 scan packet.
        Syncs to the 0xAA 0x55 header, then reads the full packet.

        Returns dict with keys:
          - distances: list of distances in mm (0 = invalid)
          - angle_start: float degrees
          - angle_end:   float degrees
          - num_points:  int
        Returns None if sync or read fails.
        """
        # Sync to packet header 0xAA 0x55
        sync_attempts = 0
        while sync_attempts < 2000:
            b = self._ser.read(1)
            if not b:
                return None
            if b[0] == self.PKT_HEADER_1:
                b2 = self._ser.read(1)
                if b2 and b2[0] == self.PKT_HEADER_2:
                    break
            sync_attempts += 1
        else:
            return None

        # Read fixed header fields (6 bytes after the 2-byte magic)
        # ct(1) + num_samples(1) + FSA(2) + LSA(2) = 6 bytes
        hdr = self._ser.read(6)
        if len(hdr) < 6:
            return None

        ct          = hdr[0]          # packet type (0x00 = point cloud)
        num_samples = hdr[1]
        fsa_raw     = hdr[2] | (hdr[3] << 8)   # first sample angle raw
        lsa_raw     = hdr[4] | (hdr[5] << 8)   # last  sample angle raw

        # Read checksum (2 bytes) + distance data (num_samples * 2 bytes)
        tail = self._ser.read(2 + num_samples * 2)
        if len(tail) < 2 + num_samples * 2:
            return None

        # Decode angles: Q6 fixed point, divide by 64, then by 100 for degrees
        angle_start = ((fsa_raw >> 1) & 0x7FFF) / 64.0
        angle_end   = ((lsa_raw  >> 1) & 0x7FFF) / 64.0

        # Decode distances
        distances = []
        for i in range(num_samples):
            lo = tail[2 + i * 2]
            hi = tail[2 + i * 2 + 1]
            dist_mm = (lo | (hi << 8)) >> 2   # distance in mm
            distances.append(dist_mm)

        return {
            "distances":   distances,
            "angle_start": angle_start,
            "angle_end":   angle_end,
            "num_points":  num_samples,
            "ct":          ct,
        }

    def disconnect(self):
        """Stop scan and close serial port."""
        if self._ser and self._ser.is_open:
            self._ser.write(self.CMD_STOP)
            time.sleep(0.1)
            self._ser.close()
            print("[LidarReader] Disconnected.")


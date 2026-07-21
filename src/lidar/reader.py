"""
lidar/reader.py
───────────────
Manages the serial connection to the YDLIDAR X3 YB.

Performance fix:
  Old approach: ser.read(1) byte-by-byte in a sync loop — up to 2000
  individual read() calls per packet, each with timeout overhead.

  New approach: read a large chunk (512 bytes) into a local bytearray
  buffer, parse from the buffer entirely in Python with no serial waits.
  When the buffer runs low, top it up with one read() call.
  This cuts serial overhead by ~95% and removes the timeout bottleneck.
"""

import serial
import time
from src import config as cfg


class LidarReader:

    CMD_STOP = b"\xA5\x65"
    CMD_SCAN = b"\xA5\x60"
    CMD_INFO = b"\xA5\x90"

    PKT_H1 = 0xAA
    PKT_H2 = 0x55

    CHUNK = 512   # bytes to read per serial top-up

    def __init__(self, port=None, baud=None):
        self.port  = port or cfg.LIDAR_PORT
        self.baud  = baud or cfg.BAUD_RATE
        self._ser  = None
        self._buf  = bytearray()   # local read buffer

    def connect(self):
        self._ser = serial.Serial(
            self.port,
            baudrate   = self.baud,
            timeout    = 0.1,       # short timeout — we top up frequently
            bytesize   = serial.EIGHTBITS,
            parity     = serial.PARITY_NONE,
            stopbits   = serial.STOPBITS_ONE,
        )
        self._ser.write(self.CMD_STOP)
        time.sleep(0.1)
        self._ser.reset_input_buffer()
        self._buf.clear()
        time.sleep(0.1)
        self._ser.write(self.CMD_SCAN)
        time.sleep(0.3)
        print(f"[LidarReader] Connected on {self.port} @ {self.baud} baud")

    def _fill(self, need):
        """Top up buffer until we have at least `need` bytes."""
        while len(self._buf) < need:
            chunk = self._ser.read(max(self.CHUNK, need))
            if chunk:
                self._buf.extend(chunk)

    def _consume(self, n):
        """Return and remove the first n bytes from the buffer."""
        out = self._buf[:n]
        del self._buf[:n]
        return out

    def read_packet(self):
        """
        Read one YDLIDAR X3 scan packet from the internal buffer.
        Syncs to 0xAA 0x55 header, parses in-buffer (no per-byte serial calls).
        Returns dict with distances, angle_start, angle_end, num_points.
        Returns None on failure.
        """
        # Sync to 0xAA 0x55 — scan buffer for header
        MAX_SYNC = 4096
        for _ in range(MAX_SYNC):
            self._fill(2)
            if self._buf[0] == self.PKT_H1 and self._buf[1] == self.PKT_H2:
                self._consume(2)   # consume the header
                break
            self._consume(1)       # skip one byte and try again
        else:
            return None

        # Read fixed header: ct(1) + num_samples(1) + FSA(2) + LSA(2) = 6 bytes
        self._fill(6)
        hdr = self._consume(6)

        num_samples = hdr[1]
        fsa_raw     = hdr[2] | (hdr[3] << 8)
        lsa_raw     = hdr[4] | (hdr[5] << 8)
        angle_start = ((fsa_raw >> 1) & 0x7FFF) / 64.0
        angle_end   = ((lsa_raw  >> 1) & 0x7FFF) / 64.0

        # Read checksum (2) + distance data (num_samples * 2)
        tail_len = 2 + num_samples * 2
        self._fill(tail_len)
        tail = self._consume(tail_len)

        distances = []
        for i in range(num_samples):
            lo = tail[2 + i * 2]
            hi = tail[2 + i * 2 + 1]
            distances.append((lo | (hi << 8)) >> 2)

        return {
            "distances":   distances,
            "angle_start": angle_start,
            "angle_end":   angle_end,
            "num_points":  num_samples,
        }

    def disconnect(self):
        if self._ser and self._ser.is_open:
            self._ser.write(self.CMD_STOP)
            time.sleep(0.1)
            self._ser.close()
            print("[LidarReader] Disconnected.")


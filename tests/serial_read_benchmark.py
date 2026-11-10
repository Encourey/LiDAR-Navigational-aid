"""
tests/serial_read_benchmark.py
──────────────────────────────
Measures what the chunked serial reader (src/lidar/reader.py) saves compared
with the original byte-by-byte approach. No LiDAR needed.

How it works
  1. Builds a synthetic YDLIDAR X3 packet stream in a temporary file.
  2. Feeds the SAME stream through
       - ByteByByteReader : the original approach (read(1) to sync, then
                            read(6) and read(2 + 2n) per packet), and
       - LidarReader      : the current chunked reader (512-byte top-ups).
  3. Counts read() calls (each is one system call on a real serial port),
     times the run, and checks both readers decode identical packets.

The call counts are exact and hardware-independent. Timings use real file
reads, so they are indicative only: run this on the Raspberry Pi for the
numbers you quote, and remember a real serial port is also limited by its
baud rate, so this shows CPU/system-call overhead, not scan rate.

Usage: python tests/serial_read_benchmark.py            (from the repo root)
"""

import os
import random
import sys
import tempfile
import time

sys.path.insert(0, ".")
from src.lidar.reader import LidarReader

N_PACKETS = 5000
SEED = 1


# ── Synthetic X3 stream ──────────────────────────────────────────────────────
def make_packet(angle_start, angle_end, dists_mm):
    """AA 55 | CT | LSN | FSA(2) | LSA(2) | CS(2) | data(2*LSN)  (matches reader.py)."""
    fsa = ((int(angle_start * 64) << 1) | 1) & 0xFFFF
    lsa = ((int(angle_end * 64) << 1) | 1) & 0xFFFF
    body = bytearray([0x00, len(dists_mm), fsa & 0xFF, fsa >> 8, lsa & 0xFF, lsa >> 8])
    body += b"\x00\x00"                                  # checksum (not checked by the reader)
    for d in dists_mm:
        v = (d << 2) & 0xFFFF
        body += bytes([v & 0xFF, v >> 8])
    return b"\xAA\x55" + bytes(body)


def build_stream(garbage_every=0):
    """garbage_every=0: clean stream. Otherwise insert 30 junk bytes before every Nth packet."""
    rng = random.Random(SEED)
    out = bytearray()
    angle = 0.0
    for i in range(N_PACKETS):
        n = rng.randint(20, 40)
        step = 360.0 / 500.0 * n                          # ~500 samples per revolution
        dists = [rng.randint(300, 7500) for _ in range(n)]
        out += make_packet(angle % 360, (angle + step) % 360, dists)
        angle += step
        if garbage_every and i % garbage_every == 0:
            out += bytes(rng.choice([b for b in range(256) if b not in (0xAA,)]) for _ in range(30))
    return bytes(out)


# ── File wrapper that counts read() calls (each is a real system call) ───────
class CountingFile:
    def __init__(self, path):
        self._f = open(path, "rb", buffering=0)
        self.calls = 0

    def read(self, n=1):
        self.calls += 1
        return self._f.read(n)

    def close(self):
        self._f.close()


# ── The original approach, kept here so the comparison is reproducible ───────
class ByteByByteReader:
    PKT_H1, PKT_H2 = 0xAA, 0x55

    def __init__(self, ser):
        self._ser = ser

    def read_packet(self):
        for _ in range(4096):                              # sync: one read(1) per byte
            b = self._ser.read(1)
            if not b:
                return None
            if b[0] == self.PKT_H1:
                b2 = self._ser.read(1)
                if b2 and b2[0] == self.PKT_H2:
                    break
        else:
            return None
        hdr = self._ser.read(6)
        if len(hdr) < 6:
            return None
        n = hdr[1]
        tail = self._ser.read(2 + n * 2)
        if len(tail) < 2 + n * 2:
            return None
        dists = [((tail[2 + i * 2] | (tail[3 + i * 2] << 8)) >> 2) for i in range(n)]
        return {
            "distances": dists,
            "angle_start": (((hdr[2] | (hdr[3] << 8)) >> 1) & 0x7FFF) / 64.0,
            "angle_end":   (((hdr[4] | (hdr[5] << 8)) >> 1) & 0x7FFF) / 64.0,
            "num_points":  n,
        }


def chunked_reader(ser):
    r = LidarReader.__new__(LidarReader)                   # skip connect(); inject our fake port
    r._ser, r._buf = ser, bytearray()
    return r


def run(reader, ser, label, expected=None):
    packets = []
    t_wall, t_cpu = time.perf_counter(), time.process_time()
    while len(packets) < N_PACKETS:
        p = reader.read_packet()
        if p is None:
            break
        packets.append(p)
    wall = (time.perf_counter() - t_wall) * 1000
    cpu = (time.process_time() - t_cpu) * 1000
    ok = ""
    if expected is not None:
        ok = "identical" if packets == expected else "DIFFERENT"
    return packets, {"label": label, "packets": len(packets), "calls": ser.calls,
                     "per_pkt": ser.calls / max(len(packets), 1), "wall": wall, "cpu": cpu, "match": ok}


def benchmark(name, data):
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(data)
        path = tf.name
    try:
        s_old = CountingFile(path)
        ref, old = run(ByteByByteReader(s_old), s_old, "byte-by-byte (original)")
        s_new = CountingFile(path)
        _, new = run(chunked_reader(s_new), s_new, "chunked 512 B (current)", expected=ref)
        s_old.close(); s_new.close()
    finally:
        os.unlink(path)

    print(f"\n{name}  ({len(data) / 1024:.0f} KB stream, {N_PACKETS} packets)")
    print(f"  {'reader':26s} {'read() calls':>13s} {'per packet':>11s} {'wall ms':>9s} {'cpu ms':>8s}")
    for r in (old, new):
        print(f"  {r['label']:26s} {r['calls']:13d} {r['per_pkt']:11.2f} {r['wall']:9.1f} {r['cpu']:8.1f}"
              f"   {r['match']}")
    cut = 100 * (1 - new["calls"] / old["calls"])
    print(f"  -> {cut:.1f}% fewer read() calls;  CPU time x{old['cpu'] / max(new['cpu'], 1e-9):.2f}")


if __name__ == "__main__":
    benchmark("Clean stream", build_stream())
    benchmark("Noisy stream (30 junk bytes before every 20th packet)", build_stream(garbage_every=20))

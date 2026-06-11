#FOR PI NAVIGATION: LiDAR-based obstacle detection with YOLO-OBB and audio/haptic feedback
import numpy as np
import cv2
import time
import serial
import struct
from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────
MODEL_PATH   = "best_ncnn_model"
LIDAR_PORT   = "/dev/ttyUSB0"   # change if needed: check with ls /dev/tty*
LIDAR_BAUD   = 115200
IMG_SIZE     = 320
CONF_THRESH  = 0.4

# BEV settings — must match convert.py
SIDE_RANGE   = (-20, 20)
FWD_RANGE    = (0, 40)
HEIGHT_RANGE = (-2, 0.5)
RESOLUTION   = 0.1

# Danger zones (normalised 0-1 in BEV image)
# BEV: top = far, bottom = near (ego vehicle)
ZONE_NEAR    = 0.75   # bottom 25% of image = within ~10m
ZONE_MID     = 0.50   # middle band = 10-20m

CLASS_NAMES  = {0: "car", 1: "pedestrian", 2: "cyclist"}

# ── FEEDBACK — choose your mode ───────────────────────────────
FEEDBACK_MODE = "audio"   # "audio" | "haptic" | "both"
# ─────────────────────────────────────────────────────────────


# ── AUDIO SETUP (pyttsx3 TTS) ─────────────────────────────────
if FEEDBACK_MODE in ("audio", "both"):
    import pyttsx3
    tts = pyttsx3.init()
    tts.setProperty("rate", 160)

def speak(text):
    if FEEDBACK_MODE in ("audio", "both"):
        tts.say(text)
        tts.runAndWait()


# ── HAPTIC SETUP (DRV2605L via I2C) ───────────────────────────
if FEEDBACK_MODE in ("haptic", "both"):
    import smbus2
    I2C_BUS      = 1
    DRV2605_ADDR = 0x5A
    bus = smbus2.SMBus(I2C_BUS)

    def haptic_init():
        bus.write_byte_data(DRV2605_ADDR, 0x01, 0x00)  # internal trigger
        bus.write_byte_data(DRV2605_ADDR, 0x03, 0x02)  # ERM open loop
        bus.write_byte_data(DRV2605_ADDR, 0x1A, 0x36)  # library B

    def haptic_pulse(effect=1):
        """Effect 1=click, 10=strong buzz, 14=sharp click."""
        bus.write_byte_data(DRV2605_ADDR, 0x04, effect)
        bus.write_byte_data(DRV2605_ADDR, 0x00, 0x01)  # GO

    haptic_init()


# ── LIDAR READER (RPLIDAR A1 simple protocol) ─────────────────
def read_lidar_scan(ser):
    """Read one full 360° scan from RPLIDAR A1."""
    points = []
    start_time = time.time()

    while time.time() - start_time < 1.0:   # 1 second timeout
        header = ser.read(1)
        if header != b'\xA5':
            continue
        descriptor = ser.read(6)
        if len(descriptor) < 6:
            continue

        # Read scan packets until we get a full rotation
        packet = ser.read(5)
        if len(packet) < 5:
            continue

        quality  = packet[0] >> 2
        angle    = ((packet[1] >> 1) | (packet[2] << 7)) / 64.0
        distance = ((packet[3]) | (packet[4] << 8)) / 1000.0  # mm to m

        if distance > 0 and quality > 0:
            # Convert polar to Cartesian (velodyne frame: X=forward, Y=left)
            angle_rad = np.deg2rad(angle)
            x =  distance * np.cos(angle_rad)
            y = -distance * np.sin(angle_rad)
            z = 0.0          # 2D LiDAR — no height info
            intensity = quality / 47.0
            points.append([x, y, z, intensity])

        if len(points) >= 360:   # one full rotation
            break

    return np.array(points, dtype=np.float32) if points else None


# ── BEV PROJECTION ────────────────────────────────────────────
def point_cloud_to_bev(pts):
    x, y, z, intensity = pts[:,0], pts[:,1], pts[:,2], pts[:,3]

    mask = ((x >= FWD_RANGE[0])    & (x <= FWD_RANGE[1]) &
            (y >= SIDE_RANGE[0])   & (y <= SIDE_RANGE[1]) &
            (z >= HEIGHT_RANGE[0]) & (z <= HEIGHT_RANGE[1]))
    x, y, z, intensity = x[mask], y[mask], z[mask], intensity[mask]

    img_h = int((FWD_RANGE[1]  - FWD_RANGE[0])  / RESOLUTION)
    img_w = int((SIDE_RANGE[1] - SIDE_RANGE[0]) / RESOLUTION)

    row = ((FWD_RANGE[1] - x) / RESOLUTION).astype(int)
    col = ((y - SIDE_RANGE[0]) / RESOLUTION).astype(int)
    row = np.clip(row, 0, img_h - 1)
    col = np.clip(col, 0, img_w - 1)

    bev = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    bev[row, col, 0] = ((z - HEIGHT_RANGE[0]) /
                        (HEIGHT_RANGE[1] - HEIGHT_RANGE[0]) * 255).astype(np.uint8)
    bev[row, col, 1] = np.clip(intensity * 255, 0, 255).astype(np.uint8)
    bev[row, col, 2] = 128

    return cv2.resize(bev, (IMG_SIZE, IMG_SIZE))


# ── DETECTION → FEEDBACK ──────────────────────────────────────
def process_detections(results):
    """Map YOLO detections to directional feedback cues."""
    alerts = []

    for result in results:
        if result.obb is None:
            continue

        boxes = result.obb
        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            if conf < CONF_THRESH:
                continue

            cls_id = int(boxes.cls[i])
            label  = CLASS_NAMES.get(cls_id, "object")

            # Get bounding box centre (normalised)
            xyxy = boxes.xyxy[i].cpu().numpy()
            cx = ((xyxy[0] + xyxy[2]) / 2) / IMG_SIZE   # left-right (0=left, 1=right)
            cy = ((xyxy[1] + xyxy[3]) / 2) / IMG_SIZE   # depth (0=far, 1=near)

            # Determine direction
            if cx < 0.35:
                direction = "left"
            elif cx > 0.65:
                direction = "right"
            else:
                direction = "ahead"

            # Determine urgency
            if cy > ZONE_NEAR:
                urgency = "warning"    # close
            elif cy > ZONE_MID:
                urgency = "caution"    # mid range
            else:
                urgency = None         # far — ignore

            if urgency:
                alerts.append({
                    "label": label,
                    "direction": direction,
                    "urgency": urgency,
                    "conf": conf,
                    "cy": cy
                })

    # Sort by proximity — nearest first
    alerts.sort(key=lambda a: -a["cy"])
    return alerts


def deliver_feedback(alerts):
    if not alerts:
        return

    # Only announce the closest threat
    a = alerts[0]

    if a["urgency"] == "warning":
        msg = f"{a['label']} {a['direction']}"
        if FEEDBACK_MODE in ("haptic", "both"):
            haptic_pulse(effect=14)   # sharp click = danger
        speak(msg)
        print(f"[ALERT] {msg} (conf={a['conf']:.2f})")

    elif a["urgency"] == "caution":
        msg = f"caution {a['direction']}"
        if FEEDBACK_MODE in ("haptic", "both"):
            haptic_pulse(effect=1)    # soft click = caution
        speak(msg)
        print(f"[CAUTION] {msg} (conf={a['conf']:.2f})")


# ── MAIN LOOP ─────────────────────────────────────────────────
def main():
    print("Loading model...")
    model = YOLO(MODEL_PATH, task="obb")

    print(f"Opening LiDAR on {LIDAR_PORT}...")
    ser = serial.Serial(LIDAR_PORT, LIDAR_BAUD, timeout=1)
    time.sleep(2)   # let LiDAR spin up

    print("Navigation started. Press Ctrl+C to stop.\n")

    try:
        while True:
            t0 = time.time()

            # 1. Read LiDAR scan
            pts = read_lidar_scan(ser)
            if pts is None or len(pts) < 10:
                print("No scan data — retrying...")
                continue

            # 2. Project to BEV
            bev = point_cloud_to_bev(pts)

            # 3. Run YOLO-OBB detection
            results = model.predict(
                source    = bev,
                imgsz     = IMG_SIZE,
                conf      = CONF_THRESH,
                verbose   = False
            )

            # 4. Process detections → alerts
            alerts = process_detections(results)

            # 5. Deliver feedback
            deliver_feedback(alerts)

            elapsed = time.time() - t0
            print(f"Loop: {elapsed*1000:.0f}ms | Detections: {len(alerts)}")

    except KeyboardInterrupt:
        print("\nStopped.")
        ser.close()


if __name__ == "__main__":
    main()
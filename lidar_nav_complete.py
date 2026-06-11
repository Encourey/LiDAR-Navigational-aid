"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║         LiDAR-ASSISTED NAVIGATION SYSTEM — COMPLETE CODEBASE                   ║
║         YOLO-OBB · BEV · Raspberry Pi 5 · Haptic / Audio Feedback              ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  FILE INDEX                                                                      ║
║  ─────────────────────────────────────────────────────────────────────────────  ║
║  [1]  verify_gpu.py          — Verify PyTorch CUDA on training machine          ║
║  [2]  setup_folders.py       — Create project directory structure               ║
║  [3]  convert.py             — KITTI → BEV PNG + YOLO-OBB labels               ║
║  [4]  check_labels.py        — Visualise OBB labels on BEV images              ║
║  [5]  bev_kitti.yaml         — Dataset config for Ultralytics training          ║
║  [6]  train.py               — YOLOv8n-OBB training script                     ║
║  [7]  export.py              — Export best.pt → NCNN format                    ║
║  [8]  lidar_raw_test.py      — Direct serial test (no library)                  ║
║  [9]  lidar_bev_test.py      — LiDAR → BEV image generation test               ║
║  [10] navigate.py            — Original navigation loop (v1)                    ║
║  [11] navigate_v2.py         — Optimised: threaded, vectorized, indoor mode     ║
╚══════════════════════════════════════════════════════════════════════════════════╝

HARDWARE
  • Compute   : Raspberry Pi 5 (4 GB) — BCM2712 quad Cortex-A76 2.4 GHz
  • AI accel  : Raspberry Pi AI HAT+ (Hailo-8L 26 TOPS) — future install
  • LiDAR     : SLAMTEC RPLIDAR A1M8 — 360° 2D, 12m, 8kSps, USB-UART
  • Haptic    : DRV2605L breakout (I2C 0x5A) + ERM coin motor 10mm
  • Audio     : MAX98357A I2S DAC + 4Ω 3W speaker (optional, not in current build)
  • Power     : 10,000 mAh USB-C power bank → 27W official Pi PSU

WIRING SUMMARY
  DRV2605L  VCC  → Pin 1  (3.3V)
  DRV2605L  GND  → Pin 6  (GND)
  DRV2605L  SDA  → Pin 3  (GPIO 2)
  DRV2605L  SCL  → Pin 5  (GPIO 3)
  DRV2605L  OUT+ → ERM motor IN+
  DRV2605L  OUT- → ERM motor IN-
  RPLIDAR         → USB-A port (via included USB-UART adapter)

TRAINING RESULTS (YOLOv8n-OBB on KITTI BEV)
  mAP50        : 0.832
  mAP50-95     : 0.519
  Car mAP50    : 0.973
  Pedestrian   : 0.734
  Cyclist      : 0.787
  Inference    : 1.5ms/frame GPU · ~177ms/frame Pi 5 CPU (NCNN)

DEPENDENCIES (training machine — Windows/Linux with NVIDIA GPU)
  pip install ultralytics torch torchvision torchaudio --index-url
              https://download.pytorch.org/whl/cu121
  pip install scikit-learn opencv-python numpy

DEPENDENCIES (Raspberry Pi 5)
  pip install ultralytics rplidar-roboticia pyserial smbus2
              pyttsx3 opencv-python-headless numpy
"""

# ══════════════════════════════════════════════════════════════════════════════
# [1]  verify_gpu.py
#      Run on training machine to confirm CUDA is available before training.
#      Usage: python verify_gpu.py
# ══════════════════════════════════════════════════════════════════════════════

VERIFY_GPU = '''
import torch

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("No CUDA GPU found.")
    print("Fix: pip uninstall torch torchvision torchaudio -y")
    print("     pip install torch torchvision torchaudio "
          "--index-url https://download.pytorch.org/whl/cu121")
'''


# ══════════════════════════════════════════════════════════════════════════════
# [2]  setup_folders.py
#      Creates the full project directory structure.
#      Usage: python setup_folders.py
# ══════════════════════════════════════════════════════════════════════════════

SETUP_FOLDERS = '''
import os

folders = [
    "kitti_raw/training/velodyne",   # KITTI raw .bin point clouds
    "kitti_raw/training/label_2",    # KITTI 3D bounding box labels
    "kitti_raw/training/calib",      # KITTI calibration files
    "kitti_raw/testing/velodyne",    # KITTI test split
    "dataset/images/train",          # Converted BEV images — training split
    "dataset/images/val",            # Converted BEV images — validation split
    "dataset/labels/train",          # YOLO-OBB label files — training split
    "dataset/labels/val",            # YOLO-OBB label files — validation split
]

for f in folders:
    os.makedirs(f, exist_ok=True)
    print(f"Created: {f}")

print("\\nFolder structure ready.")
print("Next: copy KITTI downloads into kitti_raw/training/")
'''


# ══════════════════════════════════════════════════════════════════════════════
# [3]  convert.py
#      Converts KITTI Velodyne .bin point clouds + label_2 annotations into
#      3-channel BEV PNG images and YOLO-OBB format label files.
#
#      BEV channels:
#        Channel 0 — max height (z) normalised to 0-255
#        Channel 1 — LiDAR intensity normalised to 0-255
#        Channel 2 — point density (occupied cells = 128)
#
#      YOLO-OBB label format (per line):
#        class_id  x1 y1  x2 y2  x3 y3  x4 y4
#        (4 corners normalised 0-1, col=x, row=y)
#
#      Coordinate system notes:
#        KITTI camera frame: X=right, Y=down, Z=forward
#        KITTI velodyne frame: X=forward, Y=left, Z=up
#        Labels are in camera frame → converted to velodyne frame before projection
#
#      Usage: python convert.py
# ══════════════════════════════════════════════════════════════════════════════

CONVERT = '''
import numpy as np
import cv2
import os
import glob
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
KITTI_VELODYNE = "kitti_raw/training/velodyne"
KITTI_LABELS   = "kitti_raw/training/label_2"
OUT_IMAGES     = "dataset/images"
OUT_LABELS     = "dataset/labels"

SIDE_RANGE   = (-20, 20)    # metres left/right
FWD_RANGE    = (0,   40)    # metres forward
HEIGHT_RANGE = (-2,  0.5)   # metres vertical
RESOLUTION   = 0.1          # metres per BEV pixel
IMG_SIZE     = 320          # final image size for YOLO

CLASS_MAP = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
VAL_SPLIT = 0.2             # 80/20 train/val split
# ─────────────────────────────────────────────────────────────────────────────


def load_velodyne(path):
    """Load raw KITTI Velodyne point cloud from .bin file."""
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pts  # columns: x, y, z, intensity


def point_cloud_to_bev(pts):
    """
    Project 3D point cloud onto a 2D bird's-eye-view image.
    Returns: (bev image, original grid height, original grid width)
    """
    x, y, z, intensity = pts[:,0], pts[:,1], pts[:,2], pts[:,3]

    # Filter to configured range
    mask = ((x >= FWD_RANGE[0])    & (x <= FWD_RANGE[1]) &
            (y >= SIDE_RANGE[0])   & (y <= SIDE_RANGE[1]) &
            (z >= HEIGHT_RANGE[0]) & (z <= HEIGHT_RANGE[1]))
    x, y, z, intensity = x[mask], y[mask], z[mask], intensity[mask]

    img_h = int((FWD_RANGE[1]  - FWD_RANGE[0])  / RESOLUTION)
    img_w = int((SIDE_RANGE[1] - SIDE_RANGE[0]) / RESOLUTION)

    # KITTI velodyne: X=forward → row (flip so far=top)
    #                 Y=left    → col
    row = ((FWD_RANGE[1] - x) / RESOLUTION).astype(int)
    col = ((y - SIDE_RANGE[0]) / RESOLUTION).astype(int)
    row = np.clip(row, 0, img_h - 1)
    col = np.clip(col, 0, img_w - 1)

    bev = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    # Ch0: height
    bev[row, col, 0] = ((z - HEIGHT_RANGE[0]) /
                        (HEIGHT_RANGE[1] - HEIGHT_RANGE[0]) * 255).astype(np.uint8)
    # Ch1: intensity
    bev[row, col, 1] = np.clip(intensity * 255, 0, 255).astype(np.uint8)
    # Ch2: density (occupied)
    bev[row, col, 2] = 128

    bev = cv2.resize(bev, (IMG_SIZE, IMG_SIZE))
    return bev, img_h, img_w


def load_kitti_labels(label_path, img_h, img_w):
    """
    Parse KITTI label_2 file and convert 3D bounding boxes to YOLO-OBB format.
    KITTI label columns: type trunc occ alpha x1 y1 x2 y2 h w l x y z ry
    Returns list of YOLO-OBB label strings.
    """
    obb_lines = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            cls_name = parts[0]
            if cls_name not in CLASS_MAP:
                continue

            cls_id = CLASS_MAP[cls_name]
            h_box  = float(parts[8])   # 3D box height
            w_box  = float(parts[9])   # 3D box width
            l_box  = float(parts[10])  # 3D box length
            x3d    = float(parts[11])  # camera X (right)
            y3d    = float(parts[12])  # camera Y (down)
            z3d    = float(parts[13])  # camera Z (forward)
            ry     = float(parts[14])  # rotation around camera Y axis

            # Skip objects outside BEV range
            if not (FWD_RANGE[0]  <= z3d <= FWD_RANGE[1]):  continue
            if not (SIDE_RANGE[0] <= x3d <= SIDE_RANGE[1]): continue

            # Convert camera frame → velodyne frame
            # camera Z (forward) → velodyne X (forward)
            # camera X (right)   → velodyne Y (left, negated)
            velo_x =  z3d
            velo_y = -x3d

            # Project to BEV pixel centre
            row_c = (FWD_RANGE[1] - velo_x) / RESOLUTION
            col_c = (velo_y - SIDE_RANGE[0]) / RESOLUTION

            # Box dimensions in pixels
            box_l = l_box / RESOLUTION
            box_w = w_box / RESOLUTION

            # Scale to resized image
            sx = IMG_SIZE / img_h
            sy = IMG_SIZE / img_w
            row_c *= sx;  col_c *= sy
            box_l *= sx;  box_w *= sy

            # Compute 4 oriented corners
            angle = -ry
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            corners = np.array([[-box_l/2, -box_w/2],
                                 [ box_l/2, -box_w/2],
                                 [ box_l/2,  box_w/2],
                                 [-box_l/2,  box_w/2]])
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            corners = (rot @ corners.T).T + [row_c, col_c]

            # Normalise: YOLO OBB wants x=col/W, y=row/H
            norm = np.zeros_like(corners)
            norm[:, 0] = corners[:, 1] / IMG_SIZE   # x = col
            norm[:, 1] = corners[:, 0] / IMG_SIZE   # y = row
            norm = np.clip(norm, 0, 1)

            coords = " ".join([f"{v:.6f}" for v in norm.flatten()])
            obb_lines.append(f"{cls_id} {coords}")

    return obb_lines


def convert():
    # Clear old dataset
    for split in ["train", "val"]:
        for d in [f"{OUT_IMAGES}/{split}", f"{OUT_LABELS}/{split}"]:
            for f in glob.glob(f"{d}/*"):
                os.remove(f)
    print("Cleared old dataset.")

    bin_files = sorted(glob.glob(f"{KITTI_VELODYNE}/*.bin"))
    print(f"Found {len(bin_files)} point cloud files")

    train_files, val_files = train_test_split(
        bin_files, test_size=VAL_SPLIT, random_state=42)
    splits = {"train": train_files, "val": val_files}

    for split, files in splits.items():
        print(f"\\nConverting {split} — {len(files)} files...")
        skipped = 0
        for bin_path in files:
            stem = os.path.splitext(os.path.basename(bin_path))[0]
            label_path = f"{KITTI_LABELS}/{stem}.txt"
            if not os.path.exists(label_path):
                skipped += 1; continue
            pts = load_velodyne(bin_path)
            bev, img_h, img_w = point_cloud_to_bev(pts)
            obb_lines = load_kitti_labels(label_path, img_h, img_w)
            if not obb_lines:
                skipped += 1; continue
            cv2.imwrite(f"{OUT_IMAGES}/{split}/{stem}.png", bev)
            with open(f"{OUT_LABELS}/{split}/{stem}.txt", "w") as f:
                f.write("\\n".join(obb_lines))
        print(f"  Done. Skipped {skipped} files.")

    # Remove cache so training uses fresh labels
    for f in ["dataset/labels/train.cache", "dataset/labels/val.cache"]:
        if os.path.exists(f): os.remove(f)

    print("\\nConversion complete.")
    print(f"  Train: {len(os.listdir(OUT_IMAGES+\'/train\'))} images")
    print(f"  Val:   {len(os.listdir(OUT_IMAGES+\'/val\'))} images")


if __name__ == "__main__":
    convert()
'''


# ══════════════════════════════════════════════════════════════════════════════
# [4]  check_labels.py
#      Draws OBB label boxes on BEV images so you can visually verify
#      that the coordinate conversion in convert.py is correct.
#      Green = car, Red = pedestrian, Blue = cyclist
#      Saves check_XXXXXX.png files in current directory.
#      Usage: python check_labels.py
# ══════════════════════════════════════════════════════════════════════════════

CHECK_LABELS = '''
import cv2
import numpy as np
import os

IMG_DIR = "dataset/images/train"
LBL_DIR = "dataset/labels/train"
N_SHOW  = 5     # how many frames to check

files = sorted(os.listdir(IMG_DIR))[:N_SHOW]

for fname in files:
    stem = os.path.splitext(fname)[0]
    img  = cv2.imread(f"{IMG_DIR}/{fname}")
    h, w = img.shape[:2]

    with open(f"{LBL_DIR}/{stem}.txt") as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            cls_id = int(parts[0])
            coords = parts[1:]
            pts = np.array(coords).reshape(4, 2)
            pts[:, 0] *= w  # x = col
            pts[:, 1] *= h  # y = row
            pts = pts.astype(np.int32)
            color = [(0,255,0), (0,0,255), (255,0,0)][cls_id]
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

    out = f"check_{stem}.png"
    cv2.imwrite(out, img)
    print(f"Saved {out}")

print("Done. Boxes should sit ON the bright point clusters.")
print("Green=car  Red=pedestrian  Blue=cyclist")
'''


# ══════════════════════════════════════════════════════════════════════════════
# [5]  bev_kitti.yaml
#      Dataset configuration file for Ultralytics YOLOv8-OBB training.
#      Place in same directory as train.py.
# ══════════════════════════════════════════════════════════════════════════════

BEV_KITTI_YAML = '''
# bev_kitti.yaml — KITTI BEV dataset config for YOLOv8-OBB
path:  ./dataset        # root dataset directory
train: images/train     # training images (relative to path)
val:   images/val       # validation images (relative to path)

nc: 3                   # number of classes
names:
  0: car
  1: pedestrian
  2: cyclist
'''


# ══════════════════════════════════════════════════════════════════════════════
# [6]  train.py
#      YOLOv8n-OBB training on KITTI BEV dataset.
#      Requires GPU. Trains for 100 epochs with early stopping.
#      Results saved to lidar_nav/bev_obb_v1/
#      Best weights: lidar_nav/bev_obb_v1/weights/best.pt
#
#      Final results achieved:
#        mAP50=0.832  mAP50-95=0.519
#        Car=0.973  Pedestrian=0.734  Cyclist=0.787
#        Inference: 1.5ms/image on GPU
#
#      Usage: python train.py
# ══════════════════════════════════════════════════════════════════════════════

TRAIN = '''
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n-obb.pt")   # downloads pretrained weights on first run

    results = model.train(
        data     = "bev_kitti.yaml",
        epochs   = 100,
        imgsz    = 320,      # matches Pi target resolution
        batch    = 8,        # reduce to 4 if CUDA out of memory
        device   = 0,        # GPU 0
        patience = 20,       # early stop if no improvement for 20 epochs
        workers  = 4,        # safe on Windows (avoids multiprocessing issues)
        project  = "lidar_nav",
        name     = "bev_obb_v1"
    )
    # Monitor during training:
    #   box_loss + cls_loss should fall each epoch
    #   mAP50 should reach 0.75+ by epoch 50-60
    #   Full 100 epochs takes ~1.7 hrs on RTX 4060
'''


# ══════════════════════════════════════════════════════════════════════════════
# [7]  export.py
#      Exports trained best.pt to NCNN format for Raspberry Pi inference.
#      NCNN is the fastest inference format for ARM CPUs (no GPU needed).
#      Note: int8=True is not supported via Ultralytics for NCNN —
#            INT8 quantisation must be done separately using NCNN tools on Pi.
#      Output: best_ncnn_model/ folder next to best.pt
#
#      Usage: python export.py
# ══════════════════════════════════════════════════════════════════════════════

EXPORT = '''
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("lidar_nav/bev_obb_v1/weights/best.pt")

    model.export(
        format = "ncnn",
        imgsz  = 320,
        # int8=True is NOT supported for NCNN via Ultralytics
        # Do INT8 quantisation on-device using NCNN calibration tools
    )
    print("Export complete.")
    print("Copy best_ncnn_model/ folder to Raspberry Pi ~/lidar_nav/")
'''


# ══════════════════════════════════════════════════════════════════════════════
# [8]  lidar_raw_test.py
#      Direct serial communication test for RPLIDAR A1M8.
#      Bypasses the rplidar-roboticia library entirely (incompatible with
#      Python 3.13 — get_health() return value format changed).
#      Reads GET_INFO response directly from UART at 115200 baud.
#
#      Expected output:
#        Model: 40  Firmware: 1.29  Hardware: 7  Serial: XXXXXXXX
#
#      Usage (on Raspberry Pi): python lidar_raw_test.py
# ══════════════════════════════════════════════════════════════════════════════

LIDAR_RAW_TEST = '''
import serial
import time

ser = serial.Serial("/dev/ttyUSB0", baudrate=115200, timeout=1)

# Stop any running scan and flush buffer
ser.write(b"\\xA5\\x25")
time.sleep(0.5)
ser.reset_input_buffer()
time.sleep(0.5)

# Send GET_INFO command (0xA5 0x50)
ser.write(b"\\xA5\\x50")
time.sleep(0.2)

descriptor = ser.read(7)
print(f"Descriptor hex: {descriptor.hex()}")

if len(descriptor) >= 2:
    if descriptor[0] == 0xA5 and descriptor[1] == 0x5A:
        # Valid descriptor — read 20-byte info payload
        data = ser.read(20)
        print(f"Model    : {data[0]}")
        print(f"Firmware : {data[2]}.{data[1]}")
        print(f"Hardware : {data[3]}")
        print(f"Serial   : {data[4:].hex()}")
    elif descriptor[0] == 0xAA and descriptor[1] == 0x55:
        # Sensor was mid-scan — stop and retry
        print("Sensor in scan mode — stopping and retrying...")
        ser.write(b"\\xA5\\x25")
        time.sleep(1)
        ser.reset_input_buffer()
        ser.write(b"\\xA5\\x50")
        time.sleep(0.2)
        descriptor = ser.read(7)
        data = ser.read(20)
        print(f"Model: {data[0]}, FW: {data[2]}.{data[1]}, HW: {data[3]}")
    else:
        print(f"Unexpected descriptor bytes: {descriptor.hex()}")

ser.write(b"\\xA5\\x25")
ser.close()
print("Done.")
'''


# ══════════════════════════════════════════════════════════════════════════════
# [9]  lidar_bev_test.py
#      Tests the full LiDAR → BEV image pipeline on the Raspberry Pi.
#      Collects 2 seconds of raw serial scan data, auto-ranges the BEV
#      window to fit all detected points, and saves bev_auto.png.
#
#      BEV image: green dots = detected surfaces, orange dot = ego position
#
#      Usage (on Raspberry Pi): python lidar_bev_test.py
# ══════════════════════════════════════════════════════════════════════════════

LIDAR_BEV_TEST = '''
import serial
import time
import numpy as np
import cv2

PORT         = "/dev/ttyUSB0"
BAUD         = 115200
MAX_DIST_MM  = 12000   # filter no-return values (A1M8 returns ~16000 for no object)

ser = serial.Serial(PORT, baudrate=BAUD, timeout=2)
ser.write(b"\\xA5\\x25")
time.sleep(0.5)
ser.reset_input_buffer()
time.sleep(0.3)
ser.write(b"\\xA5\\x20")   # start scan
ser.read(7)                  # discard 7-byte descriptor
time.sleep(0.2)

# Collect 2 seconds of scan data
points = []
start = time.time()
while time.time() - start < 2.0:
    raw = ser.read(5)
    if len(raw) < 5: continue
    b0,b1,b2,b3,b4 = raw
    quality  = b0 >> 2
    angle    = ((b1 >> 1) | (b2 << 7)) / 64.0
    distance = ((b3) | (b4 << 8)) / 4.0   # mm
    if quality > 0 and 0 < distance < MAX_DIST_MM:
        angle_rad = np.deg2rad(angle % 360)
        x =  (distance / 1000.0) * np.cos(angle_rad)
        y = -(distance / 1000.0) * np.sin(angle_rad)
        points.append((x, y))

ser.write(b"\\xA5\\x25")
ser.close()

print(f"Total points collected: {len(points)}")
xs = [p[0] for p in points]
ys = [p[1] for p in points]
print(f"X range: {min(xs):.2f} to {max(xs):.2f} m")
print(f"Y range: {min(ys):.2f} to {max(ys):.2f} m")

# Build BEV image with auto-range
margin = 0.5
x_min, x_max = min(xs)-margin, max(xs)+margin
y_min, y_max = min(ys)-margin, max(ys)+margin
RES = 0.03   # 3cm per pixel

img_h = int((x_max - x_min) / RES)
img_w = int((y_max - y_min) / RES)
bev   = np.zeros((img_h, img_w, 3), dtype=np.uint8)

for x,y in points:
    row = int((x_max - x) / RES)
    col = int((y - y_min) / RES)
    row = np.clip(row, 0, img_h-1)
    col = np.clip(col, 0, img_w-1)
    bev[row, col] = (0, 255, 128)   # bright green

# Ego dot at sensor origin
ego_r = int((x_max - 0) / RES)
ego_c = int((0   - y_min) / RES)
cv2.circle(bev, (ego_c, ego_r), 6, (0, 100, 255), -1)

bev = cv2.resize(bev, (320, 320))
cv2.imwrite("bev_auto.png", bev)
print("Saved bev_auto.png — copy to laptop to view.")
'''


# ══════════════════════════════════════════════════════════════════════════════
# [10] navigate.py  (v1 — original)
#      First working end-to-end navigation loop.
#      Sequential: scan → BEV → YOLO → print alerts.
#      Achieves ~177ms per frame (~5.6 FPS) on Pi 5 CPU with NCNN.
#      Outdoor mode only (KITTI-trained model).
#
#      Usage (on Raspberry Pi): python navigate.py
# ══════════════════════════════════════════════════════════════════════════════

NAVIGATE_V1 = '''
import serial
import time
import numpy as np
import cv2
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────
MODEL_PATH  = "/home/admin/lidar_nav/best_ncnn_model"
PORT        = "/dev/ttyUSB0"
BAUD        = 115200
IMG_SIZE    = 320
CONF        = 0.35
MAX_DIST_MM = 12000
SIDE_RANGE  = (-8, 8)
FWD_RANGE   = (-8, 8)
RESOLUTION  = 0.05
CLASS_NAMES = {0: "car", 1: "pedestrian", 2: "cyclist"}
FEEDBACK    = "print"    # "print" | "haptic" | "audio"
# ─────────────────────────────────────────────────────────────


def points_to_bev(points):
    img_h = int((FWD_RANGE[1] - FWD_RANGE[0]) / RESOLUTION)
    img_w = int((SIDE_RANGE[1] - SIDE_RANGE[0]) / RESOLUTION)
    bev   = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    for x, y in points:
        if not (FWD_RANGE[0] <= x <= FWD_RANGE[1]): continue
        if not (SIDE_RANGE[0] <= y <= SIDE_RANGE[1]): continue
        row = int((FWD_RANGE[1] - x) / RESOLUTION)
        col = int((y - SIDE_RANGE[0]) / RESOLUTION)
        bev[np.clip(row,0,img_h-1), np.clip(col,0,img_w-1)] = (0, 255, 128)
    return cv2.resize(bev, (IMG_SIZE, IMG_SIZE))


def collect_scan(ser, duration=0.15):
    points = []
    start  = time.time()
    while time.time() - start < duration:
        raw = ser.read(5)
        if len(raw) < 5: continue
        b0,b1,b2,b3,b4 = raw
        quality  = b0 >> 2
        angle    = ((b1>>1)|(b2<<7)) / 64.0
        distance = ((b3)|(b4<<8)) / 4.0
        if quality > 5 and 0 < distance < MAX_DIST_MM:
            angle_rad = np.deg2rad(angle % 360)
            x =  (distance/1000.0) * np.cos(angle_rad)
            y = -(distance/1000.0) * np.sin(angle_rad)
            points.append((x, y))
    return points


def process(results):
    alerts = []
    for r in results:
        if r.obb is None: continue
        for i in range(len(r.obb)):
            conf = float(r.obb.conf[i])
            if conf < CONF: continue
            cls  = int(r.obb.cls[i])
            xyxy = r.obb.xyxy[i].cpu().numpy()
            cx = ((xyxy[0]+xyxy[2])/2) / IMG_SIZE
            cy = ((xyxy[1]+xyxy[3])/2) / IMG_SIZE
            direction = "left" if cx<0.35 else "right" if cx>0.65 else "ahead"
            urgency   = "WARNING" if cy>0.70 else "caution" if cy>0.45 else None
            if urgency:
                alerts.append((urgency, CLASS_NAMES.get(cls,"object"),
                               direction, conf))
    alerts.sort(key=lambda a: a[0]=="WARNING", reverse=True)
    return alerts


def main():
    print("Loading model...")
    model = YOLO(MODEL_PATH, task="obb")
    print("Model loaded.")

    ser = serial.Serial(PORT, baudrate=BAUD, timeout=2)
    ser.write(b"\\xA5\\x25"); time.sleep(0.5)
    ser.reset_input_buffer();  time.sleep(0.3)
    ser.write(b"\\xA5\\x20"); ser.read(7); time.sleep(0.2)
    print("LiDAR started. Running navigation loop...\\n")

    try:
        while True:
            t0  = time.time()
            pts = collect_scan(ser, duration=0.15)
            if len(pts) < 20:
                print("Low point count, skipping frame"); continue
            bev    = points_to_bev(pts)
            result = model.predict(source=bev, imgsz=IMG_SIZE,
                                   conf=CONF, verbose=False)
            alerts  = process(result)
            elapsed = (time.time()-t0) * 1000
            if alerts:
                u,cls,d,c = alerts[0]
                print(f"[{u}] {cls} {d}  conf={c:.2f}  ({elapsed:.0f}ms)")
            else:
                print(f"[ clear ]  {len(pts)} pts  ({elapsed:.0f}ms)")
    except KeyboardInterrupt:
        print("\\nStopped.")
    finally:
        ser.write(b"\\xA5\\x25"); ser.close(); print("Done.")


if __name__ == "__main__":
    main()
'''


# ══════════════════════════════════════════════════════════════════════════════
# [11] navigate_v2.py  — OPTIMISED + INDOOR MODE
#
#      Improvements over v1:
#        • Vectorized BEV projection (NumPy, no Python loop) → ~20ms saved
#        • Threaded: scan collection overlaps with YOLO inference → ~60ms saved
#        • Indoor mode: zone-based proximity alerts (no ML needed indoors)
#        • Mode switching: set MODE = "outdoor" | "indoor" | "auto"
#        • DRV2605L haptic output wired in (smbus2)
#        • imgsz=256 option for extra speed (uncomment if needed)
#
#      Expected performance on Pi 5 (CPU only, NCNN):
#        v1 sequential : ~177ms (~5.6 FPS)
#        v2 threaded   : ~100ms (~10 FPS)    ← target
#
#      Indoor mode logic:
#        Divides BEV into 5 directional zones (front, front-left, front-right,
#        left, right). Counts points in each zone within INDOOR_WARN_DIST.
#        If density exceeds INDOOR_DENSITY_THRESH, triggers haptic/audio alert.
#        No YOLO inference in indoor mode — pure geometry, always works.
#
#      Auto mode:
#        Starts in indoor mode. Switches to outdoor mode when average point
#        distance exceeds AUTO_SWITCH_DIST (suggests open outdoor environment).
#
#      Usage (on Raspberry Pi): python navigate_v2.py
# ══════════════════════════════════════════════════════════════════════════════

NAVIGATE_V2 = '''
import serial
import time
import threading
import numpy as np
import cv2
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
MODEL_PATH  = "/home/admin/lidar_nav/best_ncnn_model"
PORT        = "/dev/ttyUSB0"
BAUD        = 115200
IMG_SIZE    = 320       # try 256 for extra speed with slight accuracy drop
CONF        = 0.35
MAX_DIST_MM = 12000
SIDE_RANGE  = (-8, 8)
FWD_RANGE   = (-8, 8)
RESOLUTION  = 0.05
CLASS_NAMES = {0: "car", 1: "pedestrian", 2: "cyclist"}

# Mode: "outdoor" | "indoor" | "auto"
MODE = "auto"

# Indoor mode settings
INDOOR_WARN_DIST     = 1.5    # metres — alert if object within this distance
INDOOR_DENSITY_THRESH= 8      # minimum points in zone to trigger alert
AUTO_SWITCH_DIST     = 4.0    # metres avg — switch to outdoor mode above this

# Haptic (DRV2605L via I2C) — set HAPTIC_ENABLED=True when wired up
HAPTIC_ENABLED = False
DRV_ADDR       = 0x5A
I2C_BUS        = 1
# ══════════════════════════════════════════════════════════════


# ── Haptic init ───────────────────────────────────────────────
if HAPTIC_ENABLED:
    import smbus2
    bus = smbus2.SMBus(I2C_BUS)
    bus.write_byte_data(DRV_ADDR, 0x01, 0x00)   # internal trigger mode
    bus.write_byte_data(DRV_ADDR, 0x03, 0x02)   # ERM open loop
    bus.write_byte_data(DRV_ADDR, 0x1A, 0x36)   # library B

def haptic_pulse(effect=1):
    """
    Trigger a haptic effect on DRV2605L.
    effect 1  = soft click (caution)
    effect 10 = strong buzz (warning)
    effect 14 = sharp double-click (danger)
    """
    if not HAPTIC_ENABLED: return
    bus.write_byte_data(DRV_ADDR, 0x04, effect)
    bus.write_byte_data(DRV_ADDR, 0x00, 0x01)   # GO bit


# ── Vectorized BEV projection ─────────────────────────────────
def points_to_bev(pts_array):
    """
    Vectorized BEV projection — 10-20x faster than per-point Python loop.
    pts_array: Nx2 numpy array of (x, y) in metres.
    """
    img_h = int((FWD_RANGE[1] - FWD_RANGE[0]) / RESOLUTION)
    img_w = int((SIDE_RANGE[1] - SIDE_RANGE[0]) / RESOLUTION)

    x = pts_array[:, 0]
    y = pts_array[:, 1]

    # Mask to BEV range
    mask = ((x >= FWD_RANGE[0]) & (x <= FWD_RANGE[1]) &
            (y >= SIDE_RANGE[0]) & (y <= SIDE_RANGE[1]))
    x, y = x[mask], y[mask]

    row = ((FWD_RANGE[1] - x) / RESOLUTION).astype(np.int32)
    col = ((y - SIDE_RANGE[0]) / RESOLUTION).astype(np.int32)
    row = np.clip(row, 0, img_h-1)
    col = np.clip(col, 0, img_w-1)

    bev = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    bev[row, col] = (0, 255, 128)   # green points

    # Resize with INTER_NEAREST — fastest, acceptable for point data
    return cv2.resize(bev, (IMG_SIZE, IMG_SIZE),
                      interpolation=cv2.INTER_NEAREST)


# ── Raw serial scan collection ────────────────────────────────
def collect_scan(ser, duration=0.12):
    """
    Collect LiDAR scan points for `duration` seconds.
    Returns Nx2 numpy array of (x, y) in metres.
    """
    xs, ys = [], []
    start  = time.time()
    while time.time() - start < duration:
        raw = ser.read(5)
        if len(raw) < 5: continue
        b0,b1,b2,b3,b4 = raw
        quality  = b0 >> 2
        angle    = ((b1>>1)|(b2<<7)) / 64.0
        distance = ((b3)|(b4<<8)) / 4.0   # mm
        if quality > 5 and 0 < distance < MAX_DIST_MM:
            angle_rad = np.deg2rad(angle % 360)
            xs.append((distance/1000.0) * np.cos(angle_rad))
            ys.append(-(distance/1000.0) * np.sin(angle_rad))
    if not xs:
        return np.empty((0,2))
    return np.column_stack([xs, ys])


# ── Indoor zone-based proximity detection ─────────────────────
def indoor_check(pts_array):
    """
    Divide BEV into directional zones and count points within warning distance.
    Returns list of (urgency, direction) alerts.
    No ML inference — pure geometry, works for any environment.

    Zones (based on angle from sensor):
      front        :  -30  to  30 deg
      front-left   :  30   to  90 deg
      front-right  : -90   to -30 deg
      left         :  90   to  150 deg
      right        : -150  to -90 deg
    """
    alerts = []
    if len(pts_array) == 0:
        return alerts

    x = pts_array[:, 0]
    y = pts_array[:, 1]
    dist  = np.sqrt(x**2 + y**2)
    angle = np.degrees(np.arctan2(y, x))   # -180 to 180

    zones = {
        "ahead":       (dist < INDOOR_WARN_DIST) & (angle > -30)  & (angle < 30),
        "left":        (dist < INDOOR_WARN_DIST) & (angle >= 30)  & (angle < 90),
        "right":       (dist < INDOOR_WARN_DIST) & (angle > -90)  & (angle <= -30),
        "hard-left":   (dist < INDOOR_WARN_DIST) & (angle >= 90)  & (angle < 150),
        "hard-right":  (dist < INDOOR_WARN_DIST) & (angle > -150) & (angle <= -90),
    }

    for direction, mask in zones.items():
        count = np.sum(mask)
        if count >= INDOOR_DENSITY_THRESH:
            # Determine urgency by closest point in zone
            zone_dists = dist[mask]
            closest    = np.min(zone_dists)
            urgency    = "WARNING" if closest < INDOOR_WARN_DIST * 0.5 else "caution"
            alerts.append((urgency, "obstacle", direction, count))

    alerts.sort(key=lambda a: a[0]=="WARNING", reverse=True)
    return alerts


# ── Outdoor YOLO detection ────────────────────────────────────
def outdoor_check(model, bev):
    """Run YOLO-OBB inference on BEV image and return alerts."""
    results = model.predict(source=bev, imgsz=IMG_SIZE,
                            conf=CONF, verbose=False)
    alerts  = []
    for r in results:
        if r.obb is None: continue
        for i in range(len(r.obb)):
            conf = float(r.obb.conf[i])
            if conf < CONF: continue
            cls  = int(r.obb.cls[i])
            xyxy = r.obb.xyxy[i].cpu().numpy()
            cx   = ((xyxy[0]+xyxy[2])/2) / IMG_SIZE
            cy   = ((xyxy[1]+xyxy[3])/2) / IMG_SIZE
            direction = "left" if cx<0.35 else "right" if cx>0.65 else "ahead"
            urgency   = "WARNING" if cy>0.70 else "caution" if cy>0.45 else None
            if urgency:
                alerts.append((urgency, CLASS_NAMES.get(cls,"object"),
                               direction, conf))
    alerts.sort(key=lambda a: a[0]=="WARNING", reverse=True)
    return alerts


# ── Feedback delivery ─────────────────────────────────────────
def deliver_feedback(alerts):
    if not alerts: return
    urgency, cls, direction, info = alerts[0]
    if urgency == "WARNING":
        haptic_pulse(effect=14)   # sharp double-click
        print(f"[WARNING] {cls} {direction}  ({info})")
    else:
        haptic_pulse(effect=1)    # soft click
        print(f"[caution] {cls} {direction}  ({info})")


# ══════════════════════════════════════════════════════════════
# THREADED MAIN LOOP
# Thread A: continuously collects LiDAR scan frames into a buffer
# Thread B (main): pulls latest frame, runs BEV + inference
# Overlap means inference time doesn\'t add to scan time → higher FPS
# ══════════════════════════════════════════════════════════════

latest_pts  = None
pts_lock    = threading.Lock()
scan_active = True

def scan_thread(ser):
    global latest_pts, scan_active
    while scan_active:
        pts = collect_scan(ser, duration=0.12)
        if len(pts) >= 20:
            with pts_lock:
                latest_pts = pts

def main():
    global scan_active
    current_mode = "indoor" if MODE in ("indoor", "auto") else "outdoor"

    print("Loading model...")
    model = YOLO(MODEL_PATH, task="obb")
    # Warm up model — first inference is always slow
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    model.predict(source=dummy, imgsz=IMG_SIZE, verbose=False)
    print(f"Model loaded. Mode: {current_mode}")

    ser = serial.Serial(PORT, baudrate=BAUD, timeout=2)
    ser.write(b"\\xA5\\x25"); time.sleep(0.5)
    ser.reset_input_buffer();  time.sleep(0.3)
    ser.write(b"\\xA5\\x20"); ser.read(7); time.sleep(0.2)
    print("LiDAR started. Running navigation loop...\\n")

    # Start background scan thread
    t = threading.Thread(target=scan_thread, args=(ser,), daemon=True)
    t.start()

    try:
        while True:
            t0 = time.time()

            # Get latest scan snapshot
            with pts_lock:
                pts = latest_pts
            if pts is None or len(pts) < 20:
                time.sleep(0.01)
                continue

            # Auto mode: switch based on average distance
            if MODE == "auto":
                dists = np.sqrt(pts[:,0]**2 + pts[:,1]**2)
                avg_dist = np.mean(dists)
                if avg_dist > AUTO_SWITCH_DIST and current_mode == "indoor":
                    current_mode = "outdoor"
                    print(f"[AUTO] Switched to OUTDOOR mode (avg dist {avg_dist:.1f}m)")
                elif avg_dist < AUTO_SWITCH_DIST * 0.8 and current_mode == "outdoor":
                    current_mode = "indoor"
                    print(f"[AUTO] Switched to INDOOR mode (avg dist {avg_dist:.1f}m)")

            # Run detection based on mode
            if current_mode == "indoor":
                alerts = indoor_check(pts)
                mode_tag = "IN"
            else:
                bev    = points_to_bev(pts)
                alerts = outdoor_check(model, bev)
                mode_tag = "OUT"

            elapsed = (time.time() - t0) * 1000

            if alerts:
                deliver_feedback(alerts)
            else:
                print(f"[{mode_tag}] clear  {len(pts)} pts  {elapsed:.0f}ms")

    except KeyboardInterrupt:
        print("\\nStopped.")
    finally:
        scan_active = False
        ser.write(b"\\xA5\\x25")
        ser.close()


if __name__ == "__main__":
    main()
'''


# ══════════════════════════════════════════════════════════════════════════════
# PRINT ALL FILES
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os

    files = {
        "verify_gpu.py":       VERIFY_GPU,
        "setup_folders.py":    SETUP_FOLDERS,
        "convert.py":          CONVERT,
        "check_labels.py":     CHECK_LABELS,
        "bev_kitti.yaml":      BEV_KITTI_YAML,
        "train.py":            TRAIN,
        "export.py":           EXPORT,
        "lidar_raw_test.py":   LIDAR_RAW_TEST,
        "lidar_bev_test.py":   LIDAR_BEV_TEST,
        "navigate.py":         NAVIGATE_V1,
        "navigate_v2.py":      NAVIGATE_V2,
    }

    os.makedirs("extracted_scripts", exist_ok=True)

    for fname, code in files.items():
        path = os.path.join("extracted_scripts", fname)
        with open(path, "w") as f:
            # Strip leading newline from triple-quoted strings
            f.write(code.lstrip("\n"))
        print(f"Written: {path}")

    print(f"\nAll {len(files)} files extracted to ./extracted_scripts/")
    print("You can also read each script directly from this file as string constants.")

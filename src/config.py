"""
config.py
─────────
Central configuration for the LiDAR navigation system.
All tunable parameters live here — no magic numbers elsewhere.
"""

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = "/home/admin/research/models/best_ncnn_model"
LIDAR_PORT   = "/dev/ttyUSB0"

# ── LiDAR serial ──────────────────────────────────────────────────────────────
BAUD_RATE    = 115200
MAX_DIST_MM  = 12000     # filter no-return readings (A1M8 returns ~16000)
MIN_QUALITY  = 5         # discard low-quality scan points

# ── BEV projection ────────────────────────────────────────────────────────────
SIDE_RANGE   = (-8, 8)   # metres left/right
FWD_RANGE    = (-8, 8)   # metres forward/back (full 360)
RESOLUTION   = 0.05      # metres per BEV pixel
IMG_SIZE     = 320       # final image size fed to YOLO

# ── YOLO inference ────────────────────────────────────────────────────────────
CONF_THRESH  = 0.35
CLASS_NAMES  = {0: "car", 1: "pedestrian", 2: "cyclist"}

# ── Navigation mode ───────────────────────────────────────────────────────────
MODE         = "auto"    # "outdoor" | "indoor" | "auto"

# ── Indoor zone detection ─────────────────────────────────────────────────────
INDOOR_WARN_DIST      = 2.5
INDOOR_DENSITY_THRESH = 4

# ── Auto mode switching ───────────────────────────────────────────────────────
AUTO_SWITCH_DIST      = 4.0

# ── Scan collection ───────────────────────────────────────────────────────────
SCAN_DURATION         = 0.12

# ── Haptic (DRV2605L via I2C) ─────────────────────────────────────────────────
HAPTIC_ENABLED = False
DRV_I2C_ADDR   = 0x5A
DRV_I2C_BUS    = 1

# ── Audio (pyttsx3 TTS) ───────────────────────────────────────────────────────
AUDIO_ENABLED  = False
AUDIO_RATE     = 160

# ── Training (KITTI conversion) ───────────────────────────────────────────────
KITTI_VELODYNE   = "data/kitti_raw/training/velodyne"
KITTI_LABELS     = "data/kitti_raw/training/label_2"
KITTI_CALIB      = "data/kitti_raw/training/calib"
OUT_IMAGES       = "data/dataset/images"
OUT_LABELS       = "data/dataset/labels"
TRAIN_SIDE_RANGE = (-20, 20)
TRAIN_FWD_RANGE  = (0, 40)
TRAIN_HEIGHT     = (-2, 0.5)
TRAIN_RESOLUTION = 0.1
TRAIN_IMG_SIZE   = 320
TRAIN_VAL_SPLIT  = 0.2
TRAIN_CLASS_MAP  = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}

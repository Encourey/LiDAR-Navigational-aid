"""
training/convert.py
───────────────────
Converts KITTI Velodyne .bin point clouds + label_2 annotations
into 3-channel BEV PNG images and YOLO-OBB format label files.

BEV channels:
  Ch 0 — max height (z) normalised 0-255
  Ch 1 — LiDAR intensity normalised 0-255
  Ch 2 — point density (occupied cells = 128)

YOLO-OBB label format (per line):
  class_id  x1 y1  x2 y2  x3 y3  x4 y4
  (4 corners normalised 0-1, col=x, row=y)

Coordinate system:
  KITTI camera : X=right, Y=down, Z=forward
  KITTI velodyne: X=forward, Y=left, Z=up
  Labels are in camera frame → converted to velodyne before projection.

Usage: python -m src.training.convert   (from research/ root)
"""

import numpy as np
import cv2
import os
import glob
from sklearn.model_selection import train_test_split
from src import config as cfg


def load_velodyne(path):
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pts  # x, y, z, intensity


def point_cloud_to_bev(pts):
    x, y, z, intensity = pts[:,0], pts[:,1], pts[:,2], pts[:,3]
    mask = ((x >= cfg.TRAIN_FWD_RANGE[0])  & (x <= cfg.TRAIN_FWD_RANGE[1]) &
            (y >= cfg.TRAIN_SIDE_RANGE[0]) & (y <= cfg.TRAIN_SIDE_RANGE[1]) &
            (z >= cfg.TRAIN_HEIGHT[0])      & (z <= cfg.TRAIN_HEIGHT[1]))
    x, y, z, intensity = x[mask], y[mask], z[mask], intensity[mask]

    img_h = int((cfg.TRAIN_FWD_RANGE[1]  - cfg.TRAIN_FWD_RANGE[0])  / cfg.TRAIN_RESOLUTION)
    img_w = int((cfg.TRAIN_SIDE_RANGE[1] - cfg.TRAIN_SIDE_RANGE[0]) / cfg.TRAIN_RESOLUTION)

    row = ((cfg.TRAIN_FWD_RANGE[1] - x) / cfg.TRAIN_RESOLUTION).astype(int)
    col = ((y - cfg.TRAIN_SIDE_RANGE[0]) / cfg.TRAIN_RESOLUTION).astype(int)
    row = np.clip(row, 0, img_h - 1)
    col = np.clip(col, 0, img_w - 1)

    bev = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    bev[row, col, 0] = ((z - cfg.TRAIN_HEIGHT[0]) /
                        (cfg.TRAIN_HEIGHT[1] - cfg.TRAIN_HEIGHT[0]) * 255).astype(np.uint8)
    bev[row, col, 1] = np.clip(intensity * 255, 0, 255).astype(np.uint8)
    bev[row, col, 2] = 128
    bev = cv2.resize(bev, (cfg.TRAIN_IMG_SIZE, cfg.TRAIN_IMG_SIZE))
    return bev, img_h, img_w


def load_kitti_labels(label_path, img_h, img_w):
    obb_lines = []
    with open(label_path) as f:
        for line in f:
            parts    = line.strip().split()
            cls_name = parts[0]
            if cls_name not in cfg.TRAIN_CLASS_MAP:
                continue
            cls_id = cfg.TRAIN_CLASS_MAP[cls_name]
            w_box  = float(parts[9])
            l_box  = float(parts[10])
            x3d    = float(parts[11])
            z3d    = float(parts[13])
            ry     = float(parts[14])

            if not (cfg.TRAIN_FWD_RANGE[0]  <= z3d <= cfg.TRAIN_FWD_RANGE[1]):  continue
            if not (cfg.TRAIN_SIDE_RANGE[0] <= x3d <= cfg.TRAIN_SIDE_RANGE[1]): continue

            velo_x =  z3d
            velo_y = -x3d
            row_c  = (cfg.TRAIN_FWD_RANGE[1] - velo_x) / cfg.TRAIN_RESOLUTION
            col_c  = (velo_y - cfg.TRAIN_SIDE_RANGE[0]) / cfg.TRAIN_RESOLUTION

            box_l  = l_box / cfg.TRAIN_RESOLUTION
            box_w  = w_box / cfg.TRAIN_RESOLUTION
            sx     = cfg.TRAIN_IMG_SIZE / img_h
            sy     = cfg.TRAIN_IMG_SIZE / img_w
            row_c *= sx; col_c *= sy; box_l *= sx; box_w *= sy

            angle       = -ry
            cos_a, sin_a= np.cos(angle), np.sin(angle)
            corners     = np.array([[-box_l/2,-box_w/2],[box_l/2,-box_w/2],
                                     [box_l/2, box_w/2],[-box_l/2, box_w/2]])
            rot         = np.array([[cos_a,-sin_a],[sin_a,cos_a]])
            corners     = (rot @ corners.T).T + [row_c, col_c]
            norm        = np.zeros_like(corners)
            norm[:,0]   = corners[:,1] / cfg.TRAIN_IMG_SIZE
            norm[:,1]   = corners[:,0] / cfg.TRAIN_IMG_SIZE
            norm        = np.clip(norm, 0, 1)
            coords      = " ".join([f"{v:.6f}" for v in norm.flatten()])
            obb_lines.append(f"{cls_id} {coords}")
    return obb_lines


def convert():
    for split in ["train", "val"]:
        for d in [f"{cfg.OUT_IMAGES}/{split}", f"{cfg.OUT_LABELS}/{split}"]:
            os.makedirs(d, exist_ok=True)
            for f in glob.glob(f"{d}/*"): os.remove(f)
    print("Cleared old dataset.")

    bin_files              = sorted(glob.glob(f"{cfg.KITTI_VELODYNE}/*.bin"))
    train_files, val_files = train_test_split(
        bin_files, test_size=cfg.TRAIN_VAL_SPLIT, random_state=42)

    for split, files in [("train", train_files), ("val", val_files)]:
        print(f"\nConverting {split} — {len(files)} files...")
        skipped = 0
        for bin_path in files:
            stem       = os.path.splitext(os.path.basename(bin_path))[0]
            label_path = f"{cfg.KITTI_LABELS}/{stem}.txt"
            if not os.path.exists(label_path): skipped += 1; continue
            pts        = load_velodyne(bin_path)
            bev, h, w  = point_cloud_to_bev(pts)
            obb_lines  = load_kitti_labels(label_path, h, w)
            if not obb_lines: skipped += 1; continue
            cv2.imwrite(f"{cfg.OUT_IMAGES}/{split}/{stem}.png", bev)
            with open(f"{cfg.OUT_LABELS}/{split}/{stem}.txt", "w") as f:
                f.write("\n".join(obb_lines))
        print(f"  Done. Skipped {skipped}.")

    for c in [f"{cfg.OUT_LABELS}/train.cache", f"{cfg.OUT_LABELS}/val.cache"]:
        if os.path.exists(c): os.remove(c)

    print("\nConversion complete.")
    print(f"  Train: {len(os.listdir(cfg.OUT_IMAGES+'/train'))} images")
    print(f"  Val:   {len(os.listdir(cfg.OUT_IMAGES+'/val'))} images")


if __name__ == "__main__":
    convert()

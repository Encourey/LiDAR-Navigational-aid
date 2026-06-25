"""
vision/bev.py
─────────────
Converts a 2D LiDAR point cloud (Nx2 numpy array) into a
3-channel Bird's-Eye View (BEV) image for YOLO-OBB inference.

BEV coordinate system:
  Top of image    = maximum forward distance (FWD_RANGE[1])
  Bottom of image = minimum forward distance (FWD_RANGE[0])
  Left of image   = SIDE_RANGE[0] (left side)
  Right of image  = SIDE_RANGE[1] (right side)
  Orange dot      = ego position (sensor origin)

Channel encoding (3-channel PNG):
  Ch 0 (B) — point presence: 128 where a point exists
  Ch 1 (G) — bright green for visibility
  Ch 2 (R) — reserved for future height encoding

Note: The KITTI-trained model uses a different 3-channel encoding
(height, intensity, density) during training. For inference with
a 2D LiDAR (no height data), we use a simplified single-channel
presence map. Detection still works because the spatial layout
of points in BEV is the primary discriminating feature.
"""

import numpy as np
import cv2
from src import config as cfg


class BEVProjector:
    """Projects LiDAR point arrays into BEV images."""

    def __init__(self):
        self.img_h = int((cfg.FWD_RANGE[1]  - cfg.FWD_RANGE[0])  / cfg.RESOLUTION)
        self.img_w = int((cfg.SIDE_RANGE[1] - cfg.SIDE_RANGE[0]) / cfg.RESOLUTION)

    def project(self, pts_array):
        """
        Vectorized BEV projection.
        pts_array: Nx2 numpy array of (x, y) in metres.
        Returns: IMG_SIZE x IMG_SIZE x 3 uint8 numpy array.
        """
        bev = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)

        if len(pts_array) == 0:
            return cv2.resize(bev, (cfg.IMG_SIZE, cfg.IMG_SIZE),
                              interpolation=cv2.INTER_NEAREST)

        x = pts_array[:, 0]
        y = pts_array[:, 1]

        # Mask points inside BEV window
        mask = ((x >= cfg.FWD_RANGE[0])  & (x <= cfg.FWD_RANGE[1]) &
                (y >= cfg.SIDE_RANGE[0]) & (y <= cfg.SIDE_RANGE[1]))
        x, y = x[mask], y[mask]

        if len(x) == 0:
            return cv2.resize(bev, (cfg.IMG_SIZE, cfg.IMG_SIZE),
                              interpolation=cv2.INTER_NEAREST)

        # Project to pixel coordinates
        row = ((cfg.FWD_RANGE[1] - x) / cfg.RESOLUTION).astype(np.int32)
        col = ((y - cfg.SIDE_RANGE[0]) / cfg.RESOLUTION).astype(np.int32)
        row = np.clip(row, 0, self.img_h - 1)
        col = np.clip(col, 0, self.img_w - 1)

        # Mark occupied cells
        bev[row, col] = (128, 255, 0)   # BGR: bright green

        # Resize with INTER_NEAREST — fastest, appropriate for point data
        return cv2.resize(bev, (cfg.IMG_SIZE, cfg.IMG_SIZE),
                          interpolation=cv2.INTER_NEAREST)

    def project_with_ego(self, pts_array):
        """Same as project() but draws the ego position dot."""
        bev = self.project(pts_array)
        cx  = cfg.IMG_SIZE // 2
        cy  = cfg.IMG_SIZE // 2
        cv2.circle(bev, (cx, cy), 5, (0, 100, 255), -1)   # orange dot
        cv2.arrowedLine(bev, (cx, cy), (cx, cy - 20),
                        (255, 255, 255), 1)                # north arrow
        return bev

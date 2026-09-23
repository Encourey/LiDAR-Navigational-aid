# LiDAR Navigational Aid

A real-time obstacle-detection and navigation-assistance system built on a 2D LiDAR and a Raspberry Pi 5. It fuses geometric proximity detection with a lightweight object-detection model to warn a user of nearby obstacles through haptic and audio feedback, and streams live scan data to a web dashboard.

## Overview

The system spins a YDLIDAR X3 to build a live 360° point cloud, then reasons about it in one of two ways depending on the environment:

- **Indoor mode** — pure geometry. Points are bucketed into five directional zones (ahead / left / right / hard-left / hard-right) and an alert fires when point density inside a zone crosses a distance threshold. No model required.
- **Outdoor mode** — the point cloud is projected into a Bird's-Eye-View (BEV) image and passed through a YOLOv8n-OBB model (exported to NCNN for fast ARM CPU inference) to detect and localize cars, pedestrians, and cyclists.
- **Auto mode** — switches between the two based on the average point distance per scan, with a hysteresis/voting scheme so a single noisy frame can't flip the mode.

Alerts are delivered as haptic pulses (DRV2605L) and spoken phrases (pyttsx3 TTS), and the whole system can optionally stream live scan + alert data over WebSocket to a browser dashboard for visualization and remote start/stop.

## Features

- Custom serial driver and packet parser for the YDLIDAR X3, reading full 512-byte chunks instead of byte-by-byte for ~95% less serial overhead
- Producer/consumer threading pipeline so LiDAR acquisition and inference run concurrently (~10 FPS vs ~5.6 FPS sequential)
- Geometry-only indoor obstacle detection (no ML dependency)
- YOLOv8n-OBB object detection on BEV projections for outdoor mode, running on-device via NCNN
- Automatic indoor/outdoor mode switching with frame-voting hysteresis
- Haptic (DRV2605L) and audio (TTS) feedback, decoupled from the main loop so alerts never block scan processing
- Live WebSocket dashboard for visualizing the scan, current mode, and alerts, with remote start/stop
- End-to-end training pipeline: KITTI point clouds → BEV images + YOLO-OBB labels → trained model → NCNN export

## System Architecture

```
YDLIDAR X3 (serial)
      │
      ▼
LidarReader → LidarParser        (src/lidar/)
      │  Nx2 point cloud (x, y)
      ▼
ScanProducer (background thread) (src/threads/)
      │  latest scan + frame_id
      ▼
  ┌───────────────┬──────────────────┐
  │  Indoor mode  │   Outdoor mode   │
  │  zone/density │  BEV projection  │  (src/vision/, src/navigation/)
  │  geometry     │  + YOLO-OBB      │
  └───────┬───────┴─────────┬────────┘
          │                 │
          ▼                 ▼
        AutoNavigator (mode switch)      (src/navigation/auto.py)
                    │
                    ▼
            alerts (urgency, class, direction, distance)
                    │
        ┌───────────┼───────────────┐
        ▼           ▼               ▼
   HapticFeedback  AudioFeedback  WebSocket → Dashboard
   (src/feedback/) (src/feedback/) (src/server/)
```

## Project Structure

```
src/
├── config.py            # all tunable parameters (paths, thresholds, ranges)
├── main.py               # entry point — CLI + main navigation loop
├── lidar/
│   ├── reader.py          # serial driver for the YDLIDAR X3
│   └── parser.py          # packet → Nx2 numpy point cloud
├── vision/
│   ├── bev.py              # point cloud → Bird's-Eye-View image
│   └── detector.py         # YOLO-OBB (NCNN) inference wrapper
├── navigation/
│   ├── indoor.py            # zone/density-based proximity detection
│   ├── outdoor.py            # BEV + YOLO detection pipeline
│   └── auto.py                # indoor/outdoor mode switching logic
├── feedback/
│   ├── haptic.py                # DRV2605L I2C haptic driver
│   └── audio.py                  # pyttsx3 TTS, non-blocking
├── threads/
│   └── pipeline.py                # producer/consumer scan acquisition thread
├── server/
│   ├── ws_server.py                # WebSocket bridge to the dashboard
│   └── dashboard.html               # live scan/alert visualizer
└── training/
    ├── convert.py                    # KITTI → BEV images + YOLO-OBB labels
    ├── train.py                        # YOLOv8n-OBB training
    └── export.py                        # export best.pt → NCNN

models/best_ncnn_model/   # trained, exported detection model
tests/                     # LiDAR read/BEV sanity tests
```

## Hardware

| Component | Purpose |
|---|---|
| YDLIDAR X3 | 360° 2D LiDAR, serial @ 115200 baud |
| Raspberry Pi 5 | On-device inference (NCNN, CPU only) |
| DRV2605L + ERM motor | Haptic alert feedback (I2C) |
| MAX98357A + speaker | Audio/TTS feedback (I2S) — stubbed, not yet wired |

## Model

YOLOv8n-OBB trained on the KITTI dataset, projected into BEV, and exported to NCNN for ARM CPU inference.

| Metric | Value |
|---|---|
| mAP50 | 0.832 |
| mAP50-95 | 0.519 |
| Car | 0.973 |
| Pedestrian | 0.734 |
| Cyclist | 0.787 |
| Inference speed (Pi 5, CPU) | ~100–177 ms/frame |

## Usage

```bash
# Run the navigation loop
python src/main.py                    # uses MODE from config.py
python src/main.py --mode indoor      # force indoor mode
python src/main.py --mode outdoor     # force outdoor mode
python src/main.py --mode auto        # auto-switch (default)

# Run with the live dashboard
pip install websockets --break-system-packages
python src/server/ws_server.py --mode auto
```

Press `Ctrl+C` to stop — the LiDAR is safely disconnected and the serial port closed.

### Retraining the detection model

```bash
python -m src.training.convert   # KITTI → BEV images + YOLO-OBB labels
python -m src.training.train     # train YOLOv8n-OBB (needs a CUDA GPU)
python -m src.training.export    # export best.pt → NCNN for the Pi
```

## PowerShell Shortcuts (Windows dev machine)

If you're controlling the Pi from a Windows machine, add these functions to your PowerShell profile for quick access to the dashboard and services:

```powershell
notepad $PROFILE   # paste the block below in, save, restart PowerShell
```

```powershell
#── LiDAR Nav Shortcuts ────────────────────────────────────────────────
$PI_IP   = "lidarnav.local"
$PI_USER = "admin"

# Open dashboard in browser
function lidar-dash {
    Start-Process "http://${PI_IP}:8080/src/server/dashboard.html"
}

# SSH into Pi and follow nav logs
function lidar-logs {
    ssh "${PI_USER}@${PI_IP}" "journalctl -u lidar-nav -f"
}

# Restart both services on Pi
function lidar-restart {
    ssh "${PI_USER}@${PI_IP}" "sudo systemctl restart lidar-nav lidar-dashboard"
    Write-Host "Restarted lidar-nav and lidar-dashboard." -ForegroundColor Green
}

# Check status of both services
function lidar-status {
    ssh "${PI_USER}@${PI_IP}" "sudo systemctl status lidar-nav lidar-dashboard --no-pager"
}

# SSH into Pi directly
function lidar-ssh {
    ssh "${PI_USER}@${PI_IP}"
}

# Stop LiDAR from collecting points and stop dashboard broadcast
function lidar-stop {
    ssh "${PI_USER}@${PI_IP}" "sudo systemctl stop lidar-nav lidar-dashboard"
    Write-Host "LiDAR nav and dashboard stopped." -ForegroundColor Red
}

Write-Host "LiDAR shortcuts loaded: lidar-dash | lidar-logs | lidar-restart | lidar-status | lidar-ssh | lidar-stop" -ForegroundColor Cyan
```

| Command | Effect |
|---|---|
| `lidar-dash` | Opens the live dashboard in your default browser |
| `lidar-logs` | Tails `lidar-nav` service logs over SSH |
| `lidar-restart` | Restarts both `lidar-nav` and `lidar-dashboard` services |
| `lidar-status` | Shows systemd status for both services |
| `lidar-ssh` | Opens an SSH session to the Pi |
| `lidar-stop` | Stops both services |

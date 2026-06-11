# LiDAR Navigational Aid

A LiDAR-based navigation system for autonomous robotics.

## Overview
This project implements real-time LiDAR processing for obstacle detection and navigation path planning.

## Features
- Real-time LiDAR processing
- Obstacle detection
- Navigation decision logic
- NCNN-based model inference

## Project Structure
- src/ # Core scripts
- models/ # Trained models
- best_ncnn_model/ # Optimized inference model

## System Architecture Diagram

                +----------------------+
                |  Power Bank 5V      |
                +----------+----------+
                           |
          +----------------+----------------+
          |                                 |
   +------v------+                  +-------v--------+
   | Raspberry Pi |                  | 5V Audio Rail  |
   |     5        |                  | MAX98357A      |
   +------+-------+                  +-------+--------+
          |                                  |
          | USB                              | I2S
          v                                  v
   +--------------+                 +------------------+
   | RPLIDAR A1M8 |                 | Speaker 4Ω 3W    |
   +--------------+                 +------------------+

          |
          | I2C
          v
   +----------------+
   | DRV2605L       |
   +--------+-------+
            |
            v
        ERM Motor

          |
          | PCIe
          v
   +----------------+
   | AI HAT+        |
   +----------------+

ALL SYSTEMS SHARE COMMON GND
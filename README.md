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
src/ # Core scripts
models/ # Trained models
best_ncnn_model/ # Optimized inference model

## System Architecture Diagram
flowchart TD

    PWR[Power Bank<br/>5V USB-C 10,000mAh] --> PI[Raspberry Pi 5<br/>Quad Cortex-A76]

    %% Compute / AI
    PI -->|PCIe| AIHAT[AI HAT+<br/>Hailo-8L 26 TOPS]

    %% LiDAR
    PI -->|USB| LIDAR[RPLIDAR A1M8<br/>360° 12m]

    %% I2C Devices
    PI -->|I2C (SDA/SCL)| HAPTIC[DRV2605L Haptic Driver]
    HAPTIC --> MOTOR[ERM Coin Motor 10mm]

    %% Audio path
    PI -->|I2S| DAC[MAX98357A DAC + Class D Amp]
    DAC --> SPEAKER[4Ω 3W Speaker 28–40mm]

    %% Power distribution
    PI -->|5V| PWR
    PI -->|3.3V| HAPTIC
    PI -->|5V| DAC


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
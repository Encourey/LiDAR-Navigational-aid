#!/bin/bash
# Run this once on the Pi to install the systemd service
# Usage: bash setup_service.sh

set -e
echo "=== LiDAR Nav Service Setup ==="

# Copy service file
sudo cp lidar-nav.service /etc/systemd/system/lidar-nav.service
echo "Service file copied."

# Reload systemd
sudo systemctl daemon-reload
echo "Systemd reloaded."

# Enable on boot
sudo systemctl enable lidar-nav.service
echo "Service enabled on boot."

# Start now
sudo systemctl start lidar-nav.service
echo "Service started."

echo ""
echo "Done. Useful commands:"
echo "  sudo systemctl status lidar-nav    — check if running"
echo "  sudo systemctl stop lidar-nav      — stop it"
echo "  sudo systemctl start lidar-nav     — start it"
echo "  sudo systemctl restart lidar-nav   — restart it"
echo "  sudo systemctl disable lidar-nav   — remove from boot"
echo "  journalctl -u lidar-nav -f         — live logs"

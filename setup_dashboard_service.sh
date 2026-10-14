#!/bin/bash
# Installs the dashboard HTTP server as a systemd service
set -e
echo "=== LiDAR Dashboard Service Setup ==="

sudo cp lidar-dashboard.service /etc/systemd/system/lidar-dashboard.service
echo "Service file copied."

sudo systemctl daemon-reload
sudo systemctl enable lidar-dashboard.service
sudo systemctl start lidar-dashboard.service
echo "Dashboard service enabled and started."

echo ""
echo "Dashboard available at: http://$(hostname -I | awk '{print $1}'):8080/src/server/dashboard.html"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status lidar-dashboard   — check status"
echo "  sudo systemctl restart lidar-dashboard  — restart"
echo "  sudo systemctl disable lidar-dashboard  — remove from boot"

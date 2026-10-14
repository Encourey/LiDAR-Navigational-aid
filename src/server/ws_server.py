"""
ws_server.py
────────────
WebSocket server that streams navigation data to the dashboard app.
Run alongside main.py: python ws_server.py --mode indoor

Broadcasts JSON frames every ~100ms:
{
  "frame": 42,
  "pts": 7968,
  "mode": "indoor",
  "alerts": [{"urgency": "WARNING", "cls": "obstacle", "direction": "ahead", "dist": 0.84}],
  "clear": false
}

Usage:
  pip install websockets --break-system-packages
  python ws_server.py --mode indoor
"""

import asyncio
import json
import time
import argparse
import threading
import sys
import os
import numpy as np
sys.path.insert(0, '/home/admin/research')

import websockets
from src import config as cfg
from src.threads.pipeline import ScanProducer
from src.feedback.haptic import HapticFeedback
from src.feedback.audio import AudioFeedback


latest_data = {"frame": 0, "pts": 0, "mode": "indoor", "alerts": [], "clear": True, "running": True}
data_lock   = threading.Lock()
run_event   = threading.Event()
run_event.set()   # start running by default


def nav_thread(mode, model_path):
    """Runs the navigation loop in a background thread, updates latest_data."""
    from src.navigation.indoor import IndoorNavigator
    from src.navigation.outdoor import OutdoorNavigator
    from src.navigation.auto import AutoNavigator

    if mode == "indoor":
        navigator = IndoorNavigator()
    elif mode == "outdoor":
        navigator = OutdoorNavigator(model_path)
    else:
        navigator = AutoNavigator(model_path)

    haptic   = HapticFeedback()
    audio    = AudioFeedback()
    producer = ScanProducer()
    producer.start()

    frame_count   = 0
    last_frame_id = -1

    print(f"[Nav] Started in {mode} mode.")

    while True:
        if not run_event.is_set():
            with data_lock:
                latest_data["running"] = False
            time.sleep(0.05)
            continue

        pts, frame_id = producer.get_latest()
        if pts is None or frame_id == last_frame_id:
            time.sleep(0.01)
            continue

        last_frame_id = frame_id
        frame_count  += 1

        if mode == "auto":
            current_mode, alerts = navigator.check(pts)
        else:
            alerts       = navigator.check(pts)
            current_mode = mode

        alert_list = []
        for u, cls, d, info in alerts:
            alert_list.append({
                "urgency":   u,
                "cls":       cls,
                "direction": d,
                "dist":      round(info, 2) if isinstance(info, float) else None,
            })

        # Fine-grained scan for the radar visualizer: (angle_deg, dist_m) per point.
        # Kept separate from the 5-sector `alerts` above, which still drives haptic/audio.
        x = pts[:, 0]
        y = pts[:, 1]
        scan_dist = np.sqrt(x**2 + y**2)
        scan_ang  = np.degrees(np.arctan2(y, x))   # 0°=forward, 90°=left, -90°=right
        scan = [[round(float(a), 1), round(float(d), 3)]
                for a, d in zip(scan_ang, scan_dist)]

        with data_lock:
            latest_data.update({
                "frame": frame_count,
                "pts":   len(pts),
                "mode":  current_mode,
                "alerts": alert_list,
                "clear": len(alert_list) == 0,
                "scan":  scan,
                "running": True,
            })

        if alerts:
            u, cls, d, info = alerts[0]
            haptic.alert(u)
            audio.alert(u, cls, d)


async def sender(websocket):
    while True:
        with data_lock:
            running = latest_data["running"]
            payload = json.dumps(latest_data)
        await websocket.send(payload)
        await asyncio.sleep(0.1 if not running else 0.05)


async def receiver(websocket):
    async for message in websocket:
        try:
            cmd = json.loads(message).get("cmd")
        except (json.JSONDecodeError, AttributeError):
            continue
        if cmd == "stop":
            run_event.clear()
            print("[WS] Nav paused by client.")
        elif cmd == "start":
            run_event.set()
            with data_lock:
                latest_data["running"] = True
            print("[WS] Nav resumed by client.")


async def handler(websocket):
    print(f"[WS] Client connected: {websocket.remote_address}")
    send_task = asyncio.create_task(sender(websocket))
    recv_task = asyncio.create_task(receiver(websocket))
    try:
        await asyncio.wait([send_task, recv_task], return_when=asyncio.FIRST_COMPLETED)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        send_task.cancel()
        recv_task.cancel()
        print(f"[WS] Client disconnected.")


async def main_async(host, port, mode, model):
    t = threading.Thread(target=nav_thread, args=(mode, model), daemon=True)
    t.start()
    print(f"[WS] Server starting on ws://{host}:{port}")
    async with websockets.serve(handler, host, port):
        await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  default=cfg.MODE, choices=["indoor","outdoor","auto"])
    parser.add_argument("--model", default=cfg.MODEL_PATH)
    parser.add_argument("--host",  default="0.0.0.0")
    parser.add_argument("--port",  type=int, default=8765)
    args = parser.parse_args()
    asyncio.run(main_async(args.host, args.port, args.mode, args.model))



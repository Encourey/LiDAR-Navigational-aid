"""
main.py
───────
Entry point for the LiDAR navigation system.
Wires together all modules and runs the main navigation loop.

Usage:
  python src/main.py                    # uses MODE from config.py
  python src/main.py --mode indoor      # force indoor mode
  python src/main.py --mode outdoor     # force outdoor mode
  python src/main.py --mode auto        # auto-switch (default)

Ctrl+C to stop — safely disconnects LiDAR and closes serial port.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import argparse

from src import config as cfg
from src.threads.pipeline import ScanProducer
from src.feedback.haptic import HapticFeedback
from src.feedback.audio import AudioFeedback


def get_navigator(mode, model_path=None):
    """Instantiate the appropriate navigator based on mode."""
    if mode == "indoor":
        from src.navigation.indoor import IndoorNavigator
        return "indoor", IndoorNavigator()
    elif mode == "outdoor":
        from src.navigation.outdoor import OutdoorNavigator
        return "outdoor", OutdoorNavigator(model_path)
    else:
        from src.navigation.auto import AutoNavigator
        return "auto", AutoNavigator(model_path)


def deliver_feedback(haptic, audio, alerts):
    """Send haptic and/or audio feedback for the highest priority alert."""
    if not alerts:
        return
    urgency, cls_name, direction, info = alerts[0]
    haptic.alert(urgency)
    audio.alert(urgency, cls_name, direction)


def main():
    parser = argparse.ArgumentParser(description="LiDAR Navigation System")
    parser.add_argument("--mode", default=cfg.MODE,
                        choices=["indoor", "outdoor", "auto"],
                        help="Navigation mode (default: from config.py)")
    parser.add_argument("--model", default=cfg.MODEL_PATH,
                        help="Path to NCNN model folder")
    args = parser.parse_args()

    print("=" * 60)
    print("  LiDAR Navigation System")
    print(f"  Mode  : {args.mode}")
    print(f"  Model : {args.model}")
    print("=" * 60)

    # Initialise subsystems
    mode_label, navigator = get_navigator(args.mode, args.model)
    haptic   = HapticFeedback()
    audio    = AudioFeedback()
    producer = ScanProducer()

    producer.start()
    print("\nNavigation loop running. Press Ctrl+C to stop.\n")

    frame_count = 0
    try:
        while True:
            t0  = time.time()
            pts = producer.get_latest()

            if pts is None or len(pts) < 20:
                time.sleep(0.01)
                continue

            # Run navigation check
            if args.mode == "auto":
                current_mode, alerts = navigator.check(pts)
            else:
                alerts       = navigator.check(pts)
                current_mode = args.mode

            elapsed = (time.time() - t0) * 1000
            frame_count += 1

            # Deliver feedback
            deliver_feedback(haptic, audio, alerts)

            # Console output
            tag = current_mode[:3].upper()
            if alerts:
                u, cls, d, info = alerts[0]
                print(f"[{tag}][{u:7s}] {cls:12s} {d:11s} "
                      f"| {len(pts):4d} pts | {elapsed:5.0f}ms "
                      f"| frame {frame_count}")
            else:
                print(f"[{tag}][ clear ] {'':24s}"
                      f"| {len(pts):4d} pts | {elapsed:5.0f}ms "
                      f"| frame {frame_count}")

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        producer.stop()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()

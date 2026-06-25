"""
feedback/haptic.py
──────────────────
DRV2605L haptic driver interface via I2C.

Hardware:
  DRV2605L breakout board (Adafruit #2305 or equivalent)
  ERM coin motor 10mm, 3V, ~80mA
  Connected via I2C on GPIO 2 (SDA) and GPIO 3 (SCL)

I2C address: 0x5A (default, fixed)

Haptic effects used (Texas Instruments TouchSense 2200 library):
  Effect  1 — Soft click          (caution, far obstacle)
  Effect 10 — Strong buzz         (warning, close obstacle)
  Effect 14 — Sharp double-click  (danger, immediate)

Enable by setting HAPTIC_ENABLED = True in config.py
and confirming DRV2605L is wired to Pi 5 GPIO pins.
"""

from src import config as cfg


class HapticFeedback:
    """Controls DRV2605L haptic driver over I2C."""

    # DRV2605L register map
    REG_STATUS   = 0x00
    REG_MODE     = 0x01
    REG_RTPIN    = 0x02
    REG_LIBRARY  = 0x03
    REG_WAVESEQ1 = 0x04
    REG_GO       = 0x0C
    REG_CONTROL3 = 0x1A

    def __init__(self):
        self._bus     = None
        self._enabled = cfg.HAPTIC_ENABLED

        if self._enabled:
            try:
                import smbus2
                self._bus = smbus2.SMBus(cfg.DRV_I2C_BUS)
                self._init_device()
                print("[Haptic] DRV2605L initialised.")
            except Exception as e:
                print(f"[Haptic] Init failed: {e} — haptic disabled.")
                self._enabled = False
        else:
            print("[Haptic] Disabled in config (HAPTIC_ENABLED=False).")

    def _write(self, reg, val):
        self._bus.write_byte_data(cfg.DRV_I2C_ADDR, reg, val)

    def _init_device(self):
        """Configure DRV2605L for ERM internal trigger mode."""
        self._write(self.REG_MODE,    0x00)   # internal trigger
        self._write(self.REG_CONTROL3, 0x02)  # ERM open loop
        self._write(self.REG_LIBRARY, 0x36)   # library B (ERM)

    def pulse(self, effect=1):
        """
        Trigger a haptic effect.
        effect: int — TouchSense 2200 waveform library effect number
          1  = soft click         (caution)
          10 = strong buzz        (warning)
          14 = sharp double-click (danger/WARNING)
        """
        if not self._enabled:
            return
        try:
            self._write(self.REG_WAVESEQ1, effect)
            self._write(self.REG_GO, 0x01)   # GO bit
        except Exception as e:
            print(f"[Haptic] pulse error: {e}")

    def alert(self, urgency):
        """Map urgency string to appropriate haptic effect."""
        effects = {
            "WARNING": 14,
            "caution": 1,
        }
        self.pulse(effects.get(urgency, 1))

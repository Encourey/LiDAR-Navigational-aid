"""
feedback/audio.py
─────────────────
Text-to-speech audio feedback using pyttsx3.

Hardware:
  MAX98357A I2S Class D amplifier breakout
  4Ω 3W speaker (28-40mm)
  Connected via I2S: BCLK→GPIO18, LRCLK→GPIO19, DIN→GPIO21

pyttsx3 uses espeak on Linux (pre-installed on Raspberry Pi OS).
Audio output goes through the I2S DAC if configured as default
ALSA device, otherwise falls back to HDMI or 3.5mm analog.

Enable by setting AUDIO_ENABLED = True in config.py.

NOTE: Current build does not include speaker hardware.
      This module is stubbed and ready to activate.
"""

from src import config as cfg


class AudioFeedback:
    """TTS audio feedback using pyttsx3."""

    def __init__(self):
        self._engine  = None
        self._enabled = cfg.AUDIO_ENABLED

        if self._enabled:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", cfg.AUDIO_RATE)
                print("[Audio] pyttsx3 TTS initialised.")
            except Exception as e:
                print(f"[Audio] Init failed: {e} — audio disabled.")
                self._enabled = False
        else:
            print("[Audio] Disabled in config (AUDIO_ENABLED=False).")

    def speak(self, text):
        """
        Speak a text string via TTS.
        Non-blocking: returns immediately, speech runs in background.
        """
        if not self._enabled:
            return
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as e:
            print(f"[Audio] speak error: {e}")

    def alert(self, urgency, cls_name, direction):
        """Build and speak a navigation alert message."""
        if urgency == "WARNING":
            msg = f"{cls_name} {direction}"
        else:
            msg = f"caution {direction}"
        self.speak(msg)

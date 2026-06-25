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

Threading fix:
  pyttsx3.runAndWait() is synchronous — it blocks for the full duration
  of speech (~600ms per phrase). Calling it from the main nav loop would
  stall scan processing every time an alert fires.

  Fix: speech runs in a dedicated daemon thread. A threading.Event
  (_speaking) acts as a lock — if the previous phrase hasn't finished,
  the new alert is dropped rather than queued (stale navigation alerts
  are useless anyway; only the current state matters).
"""

import threading
from src import config as cfg


class AudioFeedback:
    """TTS audio feedback using pyttsx3, non-blocking via background thread."""

    def __init__(self):
        self._engine   = None
        self._enabled  = cfg.AUDIO_ENABLED
        self._speaking = threading.Event()   # set = currently speaking

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
        Speak a text string via TTS — non-blocking.
        If audio is already playing, the new phrase is dropped (not queued).
        Navigation alerts are time-sensitive; stale phrases are useless.
        """
        if not self._enabled:
            return
        # Drop the alert if we're still speaking the previous one
        if self._speaking.is_set():
            return
        self._speaking.set()
        t = threading.Thread(target=self._speak_worker, args=(text,), daemon=True)
        t.start()

    def _speak_worker(self, text):
        """Background worker — runs pyttsx3 synchronously off the main thread."""
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as e:
            print(f"[Audio] speak error: {e}")
        finally:
            self._speaking.clear()

    def alert(self, urgency, cls_name, direction):
        """Build and speak a navigation alert message."""
        if urgency == "WARNING":
            msg = f"{cls_name} {direction}"
        else:
            msg = f"caution {direction}"
        self.speak(msg)


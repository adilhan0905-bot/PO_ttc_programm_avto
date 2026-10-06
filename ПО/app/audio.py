from __future__ import annotations

import sys
import threading


def play_alert(level: str) -> None:
    def _play() -> None:
        try:
            if sys.platform.startswith("win"):
                import winsound
                frequency, duration = (1400, 120) if level == "CRITICAL" else (900, 100)
                winsound.Beep(frequency, duration)
            else:
                # Tkinter handles the visual/UI side; terminal bell is a safe fallback on Unix.
                print("\a", end="", flush=True)
        except Exception:
            pass

    threading.Thread(target=_play, daemon=True).start()

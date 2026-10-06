from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
SETTINGS_FILE = CONFIG_DIR / "settings.json"
LOG_DIR = BASE_DIR / "logs"
RECORDINGS_DIR = BASE_DIR / "recordings"
MODELS_DIR = BASE_DIR / "models"

@dataclass
class Settings:
    sensitivity_seconds: float = 2.0
    vehicle_speed_kmh: float = 30.0
    confidence: float = 0.35
    camera_index: int = 0
    hfov_deg: float = 90.0
    show_fps: bool = True
    show_distance: bool = True
    sound_enabled: bool = True
    can_enabled: bool = True
    model_path: str = ""

    @classmethod
    def load(cls) -> "Settings":
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if not SETTINGS_FILE.exists():
            settings = cls()
            settings.save()
            return settings
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            allowed = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**allowed)
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )


def ensure_dirs() -> None:
    for folder in (CONFIG_DIR, LOG_DIR, RECORDINGS_DIR, MODELS_DIR):
        folder.mkdir(parents=True, exist_ok=True)

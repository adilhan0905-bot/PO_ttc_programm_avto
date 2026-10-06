from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

from app.config import LOG_DIR, ensure_dirs

CSV_FILE = LOG_DIR / "warnings.csv"


def setup_logging() -> None:
    ensure_dirs()
    logging.basicConfig(
        filename=LOG_DIR / "application.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        encoding="utf-8",
    )


def log_warning(object_type: str, confidence: float, distance_m: float, ttc_s: float | None, level: str) -> None:
    ensure_dirs()
    is_new = not CSV_FILE.exists()
    with CSV_FILE.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        if is_new:
            writer.writerow(["timestamp", "object", "confidence", "distance_m", "ttc_s", "level"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            object_type,
            f"{confidence:.3f}",
            f"{distance_m:.2f}",
            "" if ttc_s is None else f"{ttc_s:.2f}",
            level,
        ])
    logging.info(
        "WARNING object=%s confidence=%.3f distance=%.2fm ttc=%s level=%s",
        object_type,
        confidence,
        distance_m,
        "n/a" if ttc_s is None else f"{ttc_s:.2f}",
        level,
    )

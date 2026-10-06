from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from ultralytics import YOLO

from app.ttc import TTCResult, calculate_ttc, danger_level, estimate_distance_m


@dataclass
class Detection:
    track_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    distance_m: float | None
    ttc_s: float | None
    level: str


class ObjectDetector:
    TARGET_CLASSES = {"person", "car", "truck", "bus", "motorcycle", "bicycle"}

    def __init__(self, model_path: str, confidence: float = 0.35, hfov_deg: float = 90.0) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.hfov_deg = hfov_deg
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logging.info("Loading model %s on %s", model_path, self.device)
        self.model = YOLO(model_path)
        self.history: dict[int, float] = {}
        self.frames = 0
        self.fps = 30.0

    def set_model(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.history.clear()

    def set_confidence(self, confidence: float) -> None:
        self.confidence = confidence

    def infer(self, frame, ego_speed_kmh: float, threshold_s: float, fps: float) -> list[Detection]:
        self.frames += 1
        self.fps = fps if fps > 0 else self.fps
        results = self.model.track(
            source=frame,
            persist=True,
            conf=self.confidence,
            imgsz=640,
            device=self.device,
            verbose=False,
            tracker="bytetrack.yaml",
        )
        if not results:
            return []

        result = results[0]
        names: dict[int, str] = result.names if hasattr(result, "names") else {}
        boxes = result.boxes
        detections: list[Detection] = []
        current_ids: set[int] = set()

        if boxes is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        classes = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        for idx, box in enumerate(xyxy):
            cls_id = int(classes[idx])
            class_name = str(names.get(cls_id, cls_id)).lower()
            if class_name not in self.TARGET_CLASSES:
                continue
            confidence = float(confs[idx])
            track_id = int(ids[idx]) if ids is not None else idx
            current_ids.add(track_id)
            x1, y1, x2, y2 = [int(v) for v in box]
            bbox_h = max(1, y2 - y1)
            distance = estimate_distance_m(frame.shape[1], bbox_h, class_name, self.hfov_deg)
            previous = self.history.get(track_id)
            ttc = calculate_ttc(distance, ego_speed_kmh, previous, self.fps)
            level = danger_level(ttc, threshold_s)
            if distance is not None:
                self.history[track_id] = distance
            detections.append(
                Detection(track_id, class_name, confidence, (x1, y1, x2, y2), distance, ttc, level)
            )

        # Limit stale history so long videos do not grow memory endlessly.
        self.history = {k: v for k, v in self.history.items() if k in current_ids}
        return detections

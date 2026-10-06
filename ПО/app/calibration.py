from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


class CameraCalibrator:
    def __init__(self, board_size: tuple[int, int] = (9, 6), square_size_m: float = 0.025) -> None:
        self.board_size = board_size
        self.square_size_m = square_size_m

    def calibrate_from_images(self, image_paths: Iterable[str], output_path: str) -> tuple[bool, str]:
        cols, rows = self.board_size
        object_points = []
        image_points = []
        pattern = np.zeros((rows * cols, 3), np.float32)
        pattern[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * self.square_size_m
        image_size = None
        used = 0

        for path in image_paths:
            image = cv2.imread(str(path))
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
            if not found:
                continue
            refined = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            object_points.append(pattern.copy())
            image_points.append(refined)
            image_size = gray.shape[::-1]
            used += 1

        if used < 3 or image_size is None:
            return False, "Нужно минимум 3 изображения с шахматной доской 9×6."

        rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
            object_points, image_points, image_size, None, None
        )
        payload = {
            "rms_reprojection_error": float(rms),
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.tolist(),
            "image_size": list(image_size),
            "board_size": list(self.board_size),
            "square_size_m": self.square_size_m,
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return True, f"Калибровка завершена. RMS = {rms:.3f} px, кадров: {used}."

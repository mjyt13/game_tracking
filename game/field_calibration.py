"""
Калибровка поля: 4 угла → гомография → сетка клеток.
Порядок углов при клике: ЛВ → ПВ → ПН → ЛН (по часовой).
"""
import json
import os
import cv2
import numpy as np


class FieldCalibration:
    _FIELD_DST = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])

    def __init__(self):
        self.corners: list[tuple[int, int]] = []
        self.grid_cols: int = 0
        self.grid_rows: int = 0
        self._H: np.ndarray | None = None      # pixel → field [0,1]²
        self._H_inv: np.ndarray | None = None  # field → pixel

    def is_calibrated(self) -> bool:
        return self._H is not None

    def set_corners(self, corners: list, cols: int, rows: int) -> None:
        assert len(corners) == 4
        self.corners = [tuple(int(v) for v in c) for c in corners]
        self.grid_cols = cols
        self.grid_rows = rows
        src = np.float32(corners)
        self._H = cv2.getPerspectiveTransform(src, self._FIELD_DST)
        self._H_inv = cv2.getPerspectiveTransform(self._FIELD_DST, src)

    def pixel_to_cell(self, px: int, py: int) -> tuple[int, int] | None:
        """Пиксель кадра → (row, col). None если вне поля."""
        if not self.is_calibrated():
            return None
        pt = cv2.perspectiveTransform(
            np.array([[[float(px), float(py)]]], np.float32), self._H
        )
        fx, fy = float(pt[0, 0, 0]), float(pt[0, 0, 1])
        if not (0.0 <= fx < 1.0 and 0.0 <= fy < 1.0):
            return None
        return int(fy * self.grid_rows), int(fx * self.grid_cols)

    def pixel_to_cell_number(self, px: int, py: int) -> int | None:
        """Пиксель кадра → номер клетки (1-indexed, слева направо сверху вниз). None если вне поля."""
        cell = self.pixel_to_cell(px, py)
        if cell is None:
            return None
        row, col = cell
        return row * self.grid_cols + col + 1

    def field_to_pixel(self, fx: float, fy: float) -> tuple[int, int]:
        """Нормированные координаты поля → пиксель кадра."""
        pt = cv2.perspectiveTransform(
            np.array([[[fx, fy]]], np.float32), self._H_inv
        )
        return int(round(float(pt[0, 0, 0]))), int(round(float(pt[0, 0, 1])))

    def draw_grid(self, frame: np.ndarray) -> None:
        """Нарисовать сетку и угловые маркеры на полном кадре."""
        if not self.is_calibrated():
            return
        color = (0, 200, 0)
        for r in range(self.grid_rows + 1):
            fy = r / self.grid_rows
            cv2.line(frame, self.field_to_pixel(0.0, fy),
                     self.field_to_pixel(1.0, fy), color, 1, cv2.LINE_AA)
        for c in range(self.grid_cols + 1):
            fx = c / self.grid_cols
            cv2.line(frame, self.field_to_pixel(fx, 0.0),
                     self.field_to_pixel(fx, 1.0), color, 1, cv2.LINE_AA)
        for corner in self.corners:
            cv2.circle(frame, corner, 6, (0, 255, 255), 2)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "corners": [list(c) for c in self.corners],
            "grid_cols": self.grid_cols,
            "grid_rows": self.grid_rows,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> bool:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.set_corners(data["corners"], data["grid_cols"], data["grid_rows"])
            return True
        except (FileNotFoundError, KeyError, ValueError):
            return False

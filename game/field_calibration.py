"""
Калибровка поля: 4 угла → гомография → сетка клеток.
Порядок углов при клике: ЛВ → ПВ → ПН → ЛН (по часовой).
Поддерживает разные порядки обхода: linear, snake, spiral.
"""
import json
import os
import cv2
import numpy as np
from game.path_manager import PathManager


class FieldCalibration:
    _FIELD_DST = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])

    def __init__(self, path_type: str = "linear"):
        self.corners: list[tuple[int, int]] = []
        self.grid_cols: int = 0
        self.grid_rows: int = 0
        self._H: np.ndarray | None = None      # pixel → field [0,1]²
        self._H_inv: np.ndarray | None = None  # field → pixel
        self.path_manager: PathManager | None = None
        self.path_type = path_type
        # Предвычисленная сетка пиксельных точек узлов [row][col], размер (rows+1)×(cols+1).
        # Считается один раз при калибровке — отрисовка кадра больше не вызывает perspectiveTransform.
        self._grid_pts: list[list[tuple[int, int]]] = []

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
        # Инициализировать менеджер пути
        self.path_manager = PathManager(self.path_type, cols, rows)
        self._precompute_grid()

    def set_path_type(self, path_type: str) -> None:
        """Сменить порядок обхода клеток без перекалибровки (узлы сетки не меняются)."""
        self.path_type = path_type
        if self.is_calibrated():
            self.path_manager = PathManager(path_type, self.grid_cols, self.grid_rows)

    def _precompute_grid(self) -> None:
        """Один раз спроецировать все узлы сетки в пиксели (rows+1 × cols+1)."""
        self._grid_pts = [
            [self.field_to_pixel(c / self.grid_cols, r / self.grid_rows)
             for c in range(self.grid_cols + 1)]
            for r in range(self.grid_rows + 1)
        ]

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
        """Пиксель кадра → номер клетки (1-indexed, порядок зависит от path_type). None если вне поля."""
        cell = self.pixel_to_cell(px, py)
        if cell is None:
            return None
        row, col = cell
        if self.path_manager:
            return self.path_manager.position_to_cell_number(row, col)
        # Fallback на linear
        return row * self.grid_cols + col + 1

    def field_to_pixel(self, fx: float, fy: float) -> tuple[int, int]:
        """Нормированные координаты поля → пиксель кадра."""
        pt = cv2.perspectiveTransform(
            np.array([[[fx, fy]]], np.float32), self._H_inv
        )
        return int(round(float(pt[0, 0, 0]))), int(round(float(pt[0, 0, 1])))

    def draw_grid(self, frame: np.ndarray) -> None:
        """Нарисовать сетку и угловые маркеры на полном кадре (по предвычисленным узлам)."""
        if not self.is_calibrated() or not self._grid_pts:
            return
        color = (0, 200, 0)
        gp = self._grid_pts
        # Гомография переводит прямые в прямые → линия сетки = отрезок между крайними узлами
        for r in range(self.grid_rows + 1):
            cv2.line(frame, gp[r][0], gp[r][self.grid_cols], color, 1, cv2.LINE_AA)
        for c in range(self.grid_cols + 1):
            cv2.line(frame, gp[0][c], gp[self.grid_rows][c], color, 1, cv2.LINE_AA)
        for corner in self.corners:
            cv2.circle(frame, corner, 6, (0, 255, 255), 2)

    def _cell_polygon(self, row: int, col: int) -> np.ndarray:
        """Четыре угла клетки (row, col) в пиксельных координатах кадра (из предвычисленных узлов)."""
        gp = self._grid_pts
        pts = [gp[row][col], gp[row][col + 1], gp[row + 1][col + 1], gp[row + 1][col]]
        return np.array(pts, dtype=np.int32)

    def draw_cell_fills(self, frame: np.ndarray, cell_colors: dict, alpha: float) -> None:
        """Полупрозрачная заливка клеток с эффектами.
        cell_colors: {номер_клетки: (B, G, R)}. Рисуется только на отображаемом кадре."""
        if not self.is_calibrated() or not self.path_manager or not cell_colors:
            return
        overlay = frame.copy()
        for num, color in cell_colors.items():
            pos = self.path_manager.cell_number_to_position(num)
            if pos is None:
                continue
            cv2.fillPoly(overlay, [self._cell_polygon(pos[0], pos[1])], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def draw_cell_numbers(self, frame: np.ndarray, color=(255, 255, 255),
                          position: str = 'center') -> None:
        """Номер клетки (в порядке маршрута) в заданном углу/центре клетки, с тёмной обводкой.
        position: 'center' | 'tl' | 'tr' | 'br' | 'bl'. Использует предвычисленные узлы сетки."""
        if not self.is_calibrated() or not self.path_manager or not self._grid_pts:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        # доля смещения от угла к центру клетки (чтобы номер не лип к самой границе)
        # индекс угла полигона: 0=tl, 1=tr, 2=br, 3=bl
        corner_idx = {'tl': 0, 'tr': 1, 'br': 2, 'bl': 3}.get(position)
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                num = self.path_manager.position_to_cell_number(row, col)
                if num is None:
                    continue
                poly = self._cell_polygon(row, col)
                cx = float(np.mean(poly[:, 0]))
                cy = float(np.mean(poly[:, 1]))
                if corner_idx is not None:
                    ax = poly[corner_idx][0] * 0.75 + cx * 0.25
                    ay = poly[corner_idx][1] * 0.75 + cy * 0.25
                else:  # center
                    ax, ay = cx, cy
                text = str(num)
                (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)
                org = (int(ax - tw / 2), int(ay + th / 2))
                cv2.putText(frame, text, org, font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, text, org, font, 0.5, color, 1, cv2.LINE_AA)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "corners": [list(c) for c in self.corners],
            "grid_cols": self.grid_cols,
            "grid_rows": self.grid_rows,
            "path_type": self.path_type,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> bool:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("path_type"):
                self.path_type = data["path_type"]
            self.set_corners(data["corners"], data["grid_cols"], data["grid_rows"])
            return True
        except (FileNotFoundError, KeyError, ValueError):
            return False

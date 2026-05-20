"""
Модуль метрик системы сканирования
Собирает FPS, время обработки, статистику детекций и отображает в отдельном окне
"""

import cv2
import numpy as np
import time
from collections import deque


class MetricsTracker:
    WINDOW_NAME = "Metrics"

    def __init__(self, fps_window: int = 30):
        self._frame_times: deque = deque(maxlen=fps_window)
        self._proc_times: deque = deque(maxlen=fps_window)
        self._last_tick: float = None
        self.camera_mode: str = "—"
        self.registered_count: int = 0
        self.detections: list = []  # [{name, conf, matches}]
        self.active_objects: list = []  # [{name, cell}]

    def tick(self) -> float:
        """Отметить начало кадра. Возвращает время старта для record_proc."""
        now = time.perf_counter()
        if self._last_tick is not None:
            self._frame_times.append(now - self._last_tick)
        self._last_tick = now
        return now

    def record_proc(self, start: float):
        """Записать время обработки одного кадра (SIFT + детекция)."""
        self._proc_times.append(time.perf_counter() - start)

    def update_detections(self, detections: list):
        self.detections = [
            {"name": d["object_name"], "conf": d["confidence"], "matches": d["matches_count"]}
            for d in detections
        ]

    def update_active_objects(self, active_objects: list):
        self.active_objects = active_objects

    @property
    def fps(self) -> float:
        if not self._frame_times:
            return 0.0
        return 1.0 / (sum(self._frame_times) / len(self._frame_times))

    @property
    def proc_ms(self) -> float:
        if not self._proc_times:
            return 0.0
        return sum(self._proc_times) / len(self._proc_times) * 1000

    def render(self):
        """Отрисовать и показать окно метрик."""
        row_h = 22
        rows = 8 + len(self.active_objects) + max(len(self.detections), 0)
        panel = np.zeros((rows * row_h + 10, 360, 3), dtype=np.uint8)

        y = row_h
        def line(text, color=(200, 200, 200)):
            nonlocal y
            cv2.putText(panel, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, color, 1, cv2.LINE_AA)
            y += row_h

        fps_color = (0, 220, 0) if self.fps >= 15 else (0, 140, 255)

        line("METRICS", (120, 255, 120))
        line(f"FPS:        {self.fps:.1f}", fps_color)
        line(f"Proc time:  {self.proc_ms:.1f} ms")
        line(f"Camera:     {self.camera_mode}")
        line(f"Registered: {self.registered_count}")
        line(f"Active:     {len(self.detections)}")
        line(f"Appeared:   {len(self.active_objects)}")

        if self.active_objects:
            line("--- appeared ---", (120, 255, 120))
            for obj in self.active_objects:
                cell = f"cell {obj['cell']}" if obj["cell"] else "outside"
                line(f"  {obj['name']}: {cell}", (180, 220, 180))

        if self.detections:
            line("--- detections ---", (160, 160, 255))
            for d in self.detections:
                line(f"  {d['name']}: {d['conf']:.2f} ({d['matches']} pts)")

        cv2.imshow(self.WINDOW_NAME, panel)

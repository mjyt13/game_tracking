"""
Адаптивное вычитание фона для регистрации объектов.
MOG2 (Mixture of Gaussians) для работы с разноцветными фонами.
"""

import cv2
import numpy as np
from config import settings


class MOG2BackgroundLearner:
    """
    MOG2 — обучить модель фона на N кадрах чистого фона,
    затем обнаружить объекты как отклонения от модели.
    """

    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=settings.MOG2_DETECT_SHADOWS,
            varThreshold=settings.MOG2_VAR_THRESHOLD
        )
        self.is_trained = False

    def train_on_frames(self, frames: list[np.ndarray]) -> None:
        """
        Обучить MOG2 на N кадрах чистого фона.
        После обучения система будет выделять объекты как отклонения от этой модели.
        """
        print(f"[MOG2] Обучение на {len(frames)} кадрах чистого фона...")
        for i, frame in enumerate(frames):
            # MOG2 обновляется при каждом вызове apply()
            _ = self.bg_subtractor.apply(frame)
            if (i + 1) % 5 == 0:
                print(f"  {i + 1}/{len(frames)} кадров обработано")

        self.is_trained = True
        print("[MOG2] Обучение завершено")

    def detect_foreground(self, frame: np.ndarray) -> np.ndarray:
        """
        Выделить объекты (фишки) как foreground (белые области маски).
        Фон → 0 (чёрный), фишка → 255 (белый).
        """
        if not self.is_trained:
            print("⚠️ MOG2 не обучен. Вызовите train_on_frames() сначала")
            return None

        mask = self.bg_subtractor.apply(frame)
        return mask

    def extract_roi_from_mask(self, frame: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, tuple] | None:
        """
        Из маски MOG2 найти bounding box объекта и вернуть ROI.
        Возвращает (roi_image, (x1, y1, x2, y2)) или None если нет объектов.
        """
        # Морфологическая обработка: убрать шум
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Найти контуры объектов
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Найти самый большой контур (предполагаем, это фишка)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Фильтр по минимальной площади (убрать шум)
        if area < settings.MOG2_MIN_AREA:
            return None

        # Bounding box вокруг контура
        x, y, w, h = cv2.boundingRect(largest_contour)
        x2, y2 = x + w, y + h

        # Извлечь ROI из исходного кадра
        roi = frame[y:y2, x:x2]

        return roi, (x, y, x2, y2)

    def get_mask_visualization(self, mask: np.ndarray) -> np.ndarray:
        """Визуализировать маску для отладки (можно показать в окне)."""
        return cv2.applyColorMap(mask, cv2.COLORMAP_JET)

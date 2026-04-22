"""
Модуль извлечения признаков из изображений
Использует SIFT для генерации дескрипторов
"""

import cv2
import numpy as np
from config import settings


class FeatureExtractor:
    """Извлечение SIFT признаков из изображений"""
    
    def __init__(self, nfeatures=None):
        """
        Инициализация детектора (SIFT или AKAZE, по settings.DETECTOR_TYPE)

        Args:
            nfeatures: Максимальное количество признаков SIFT (игнорируется для AKAZE)
        """
        if settings.DETECTOR_TYPE == 'akaze':
            self.detector = cv2.AKAZE_create(threshold=settings.AKAZE_THRESHOLD)
        else:
            nfeatures = nfeatures or settings.SIFT_FEATURES
            self.detector = cv2.SIFT_create(nfeatures=nfeatures)

        # CLAHE усиливает локальный контраст → больше признаков на объектах с неравномерным освещением
        self._clahe = cv2.createCLAHE(
            clipLimit=settings.CLAHE_CLIP_LIMIT,
            tileGridSize=settings.CLAHE_TILE_GRID
        )
    
    def extract_features(self, image):
        """
        Извлечение ключевых точек и дескрипторов из изображения
        
        Args:
            image: Входное изображение (numpy array)
            
        Returns:
            tuple: (keypoints, descriptors) или (None, None) при ошибке
        """
        if image is None or image.size == 0:
            return None, None
        
        # Конвертация в grayscale если нужно
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Извлечение признаков
        gray = self._clahe.apply(gray)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def extract_from_roi(self, image, roi):
        """
        Извлечение признаков из области интереса (ROI)
        
        Args:
            image: Полное изображение
            roi: Область интереса (x1, y1, x2, y2) или (x1, y1, width, height)
            
        Returns:
            tuple: (keypoints, descriptors, roi_image) или (None, None, None) при ошибке
        """
        if len(roi) == 4:
            x1, y1, x2, y2 = roi
            # Нормализация координат
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            roi_image = image[y1:y2, x1:x2]
        else:
            return None, None, None
        
        if roi_image.size == 0:
            return None, None, None
        
        keypoints, descriptors = self.extract_features(roi_image)
        return keypoints, descriptors, roi_image


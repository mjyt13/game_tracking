"""
Модуль детекции и распознавания объектов
Сопоставляет признаки текущего кадра с зарегистрированными объектами
"""

import cv2
import numpy as np
from config import settings


class ObjectDetector:
    """Детекция объектов на кадре по сопоставлению признаков"""
    
    def __init__(self, ratio_threshold=None, min_matches=None, ransac_threshold=None):
        """
        Инициализация детектора
        
        Args:
            ratio_threshold: Порог для ratio test (если None, из settings)
            min_matches: Минимальное количество совпадений (если None, из settings)
            ransac_threshold: Порог для RANSAC (если None, из settings)
        """
        self.ratio_threshold = ratio_threshold or settings.MATCH_RATIO_THRESHOLD
        self.min_matches = min_matches or settings.MIN_MATCHES_THRESHOLD
        self.ransac_threshold = ransac_threshold or settings.RANSAC_THRESHOLD
        
        # AKAZE использует бинарные дескрипторы → LSH; SIFT — вещественные → KD-tree
        if settings.DETECTOR_TYPE == 'akaze':
            index_params = dict(
                algorithm=settings.FLANN_INDEX_LSH,
                table_number=settings.FLANN_LSH_TABLE_NUMBER,
                key_size=settings.FLANN_LSH_KEY_SIZE,
                multi_probe_level=settings.FLANN_LSH_MULTI_PROBE_LEVEL,
            )
            search_params = dict()
        else:
            index_params = dict(
                algorithm=settings.FLANN_INDEX_KDTREE,
                trees=settings.FLANN_TREES
            )
            search_params = dict(checks=settings.FLANN_CHECKS)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    def detect_objects(self, frame_keypoints, frame_descriptors, registered_objects):
        """
        Детекция объектов на кадре
        
        Args:
            frame_keypoints: Ключевые точки текущего кадра
            frame_descriptors: Дескрипторы текущего кадра
            registered_objects: Словарь зарегистрированных объектов
                               {object_id: {des, kp, img_shape, ...}}
        
        Returns:
            list: Список обнаружений [Detection, ...]
        """
        detections = []
        
        if frame_descriptors is None or len(frame_descriptors) < 2:
            return detections
        
        # Проходим по всем зарегистрированным объектам
        for object_id, object_data in registered_objects.items():
            if "des" not in object_data or object_data["des"] is None:
                continue
            
            # Сопоставление дескрипторов
            matches = self.flann.knnMatch(object_data["des"], frame_descriptors, k=2)
            
            # Применение ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.ratio_threshold * n.distance:
                        good_matches.append(m)
            
            # Проверка наличия ключевых точек в данных объекта
            if "kp" not in object_data or object_data["kp"] is None:
                continue

            # Адаптивный порог: для малых объектов снижаем требование
            kp_count = len(object_data["kp"])
            obj_min_matches = max(4, min(self.min_matches, kp_count // 3))
            if len(good_matches) < obj_min_matches:
                continue
            
            # Извлечение координат точек
            src_pts = np.float32(
                [object_data["kp"][m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [frame_keypoints[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            
            # Нахождение гомографии
            M, mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold
            )
            
            if M is not None:
                # Вычисление границ объекта
                h, w = object_data.get("img_shape", (100, 100))
                pts = np.float32([
                    [0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]
                ]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                
                # Вычисление центра
                center_x = int(np.mean(dst[:, 0, 0]))
                center_y = int(np.mean(dst[:, 0, 1]))
                
                # Вычисление уверенности (на основе количества совпадений)
                confidence = min(len(good_matches) / 100.0, 1.0)
                
                detection = {
                    "object_id": object_id,
                    "object_name": object_data.get("name", "Unknown"),
                    "center": (center_x, center_y),
                    "corners": [(int(dst[i][0][0]), int(dst[i][0][1])) for i in range(4)],
                    "confidence": confidence,
                    "matches_count": len(good_matches),
                    "homography": M
                }
                
                detections.append(detection)
        
        # Сортировка по уверенности (лучшие первыми)
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        
        return detections


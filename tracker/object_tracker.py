"""
Модуль отслеживания позиций объектов между кадрами
Обеспечивает сглаживание и историю позиций
"""

import numpy as np
from typing import List, Dict, Optional
from collections import deque


class TrackedObject:
    """Отслеживаемый объект с историей позиций"""
    
    def __init__(self, object_id: str, initial_position: tuple, max_history: int = 10):
        """
        Инициализация отслеживаемого объекта
        
        Args:
            object_id: Уникальный ID объекта
            initial_position: Начальная позиция (x, y)
            max_history: Максимальная длина истории позиций
        """
        self.object_id = object_id
        self.position_history = deque(maxlen=max_history)
        self.position_history.append(initial_position)
        self.last_seen_frame = 0
        self.is_active = True
    
    def update_position(self, position: tuple, frame_number: int):
        """
        Обновление позиции объекта
        
        Args:
            position: Новая позиция (x, y)
            frame_number: Номер текущего кадра
        """
        self.position_history.append(position)
        self.last_seen_frame = frame_number
        self.is_active = True
    
    def get_smoothed_position(self, window_size: int = 5) -> tuple:
        """
        Получение сглаженной позиции (медиана последних позиций)
        
        Args:
            window_size: Размер окна для сглаживания
        
        Returns:
            tuple: Сглаженная позиция (x, y)
        """
        if len(self.position_history) == 0:
            return None
        
        recent_positions = list(self.position_history)[-window_size:]
        if len(recent_positions) == 1:
            return recent_positions[0]
        
        x_coords = [p[0] for p in recent_positions]
        y_coords = [p[1] for p in recent_positions]
        
        smoothed_x = int(np.median(x_coords))
        smoothed_y = int(np.median(y_coords))
        
        return (smoothed_x, smoothed_y)
    
    def get_current_position(self) -> tuple:
        """Получение текущей позиции"""
        if len(self.position_history) == 0:
            return None
        return self.position_history[-1]


class ObjectTracker:
    """Отслеживание множественных объектов на поле"""
    
    def __init__(self, max_history: int = 10, inactive_threshold: int = 30):
        """
        Инициализация трекера
        
        Args:
            max_history: Максимальная длина истории позиций для каждого объекта
            inactive_threshold: Количество кадров без обнаружения для пометки объекта как неактивного
        """
        self.tracked_objects: Dict[str, TrackedObject] = {}
        self.max_history = max_history
        self.inactive_threshold = inactive_threshold
        self.frame_number = 0
    
    def update(self, detections: List[dict]):
        """
        Обновление позиций объектов на основе детекций
        
        Args:
            detections: Список детекций от ObjectDetector
        """
        self.frame_number += 1
        
        # Обновляем существующие объекты
        detected_ids = set()
        for detection in detections:
            object_id = detection["object_id"]
            center = detection["center"]
            detected_ids.add(object_id)
            
            if object_id in self.tracked_objects:
                # Обновляем существующий объект
                self.tracked_objects[object_id].update_position(center, self.frame_number)
            else:
                # Создаем новый отслеживаемый объект
                self.tracked_objects[object_id] = TrackedObject(
                    object_id, center, self.max_history
                )
        
        # Помечаем необнаруженные объекты как неактивные
        for obj_id, tracked_obj in self.tracked_objects.items():
            if obj_id not in detected_ids:
                frames_since_seen = self.frame_number - tracked_obj.last_seen_frame
                if frames_since_seen > self.inactive_threshold:
                    tracked_obj.is_active = False
    
    def get_active_objects(self) -> List[TrackedObject]:
        """
        Получение списка активных объектов
        
        Returns:
            list: Список активных TrackedObject
        """
        return [obj for obj in self.tracked_objects.values() if obj.is_active]
    
    def get_object_position(self, object_id: str, smoothed: bool = True) -> Optional[tuple]:
        """
        Получение позиции объекта
        
        Args:
            object_id: Уникальный ID объекта
            smoothed: Использовать сглаженную позицию
        
        Returns:
            tuple: Позиция (x, y) или None если объект не найден
        """
        if object_id not in self.tracked_objects:
            return None
        
        tracked_obj = self.tracked_objects[object_id]
        if smoothed:
            return tracked_obj.get_smoothed_position()
        else:
            return tracked_obj.get_current_position()
    
    def get_all_positions(self, smoothed: bool = True) -> Dict[str, tuple]:
        """
        Получение всех позиций объектов
        
        Args:
            smoothed: Использовать сглаженные позиции
        
        Returns:
            dict: {object_id: (x, y), ...}
        """
        positions = {}
        for obj_id, tracked_obj in self.tracked_objects.items():
            if tracked_obj.is_active:
                if smoothed:
                    pos = tracked_obj.get_smoothed_position()
                else:
                    pos = tracked_obj.get_current_position()
                if pos:
                    positions[obj_id] = pos
        return positions
    
    def clear_inactive(self):
        """Удаление неактивных объектов из трекера"""
        inactive_ids = [
            obj_id for obj_id, obj in self.tracked_objects.items() 
            if not obj.is_active
        ]
        for obj_id in inactive_ids:
            del self.tracked_objects[obj_id]
    
    def reset(self):
        """Сброс трекера"""
        self.tracked_objects.clear()
        self.frame_number = 0


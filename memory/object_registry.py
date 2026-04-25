"""
Модуль реестра объектов с уникальными ID
Управление метаданными и жизненным циклом объектов
"""

import uuid
from datetime import datetime
from typing import Dict, Optional, List
from memory.feature_storage import FeatureStorage


class ObjectRegistry:
    """Централизованное управление зарегистрированными объектами"""
    
    def __init__(self, storage: Optional[FeatureStorage] = None):
        """
        Инициализация реестра
        
        Args:
            storage: Экземпляр FeatureStorage (если None, создается новый)
        """
        self.storage = storage or FeatureStorage()
        self.objects: Dict[str, dict] = {}
        self._load_from_storage()
    
    def _load_from_storage(self):
        """Загрузка объектов из хранилища"""
        loaded_data = self.storage.load_features()
        
        # Валидация и загрузка объектов в новом формате
        valid_objects = {}
        skipped_count = 0
        
        for obj_id, obj_data in loaded_data.items():
            if not isinstance(obj_data, dict):
                skipped_count += 1
                continue
            
            # Проверка наличия обязательных полей
            if "id" not in obj_data:
                print(f"⚠️ Пропущен объект без ID: {obj_id}")
                skipped_count += 1
                continue
            
            # Проверка соответствия ID ключа и ID в данных
            if obj_data["id"] != obj_id:
                print(f"⚠️ Несоответствие ID ключа и данных: ключ={obj_id}, данные={obj_data['id']}")
                skipped_count += 1
                continue
            
            # Убеждаемся, что есть все необходимые поля
            if "name" not in obj_data:
                obj_data["name"] = "Unknown"
            if "registration_time" not in obj_data:
                obj_data["registration_time"] = datetime.now().isoformat()
            if "metadata" not in obj_data:
                obj_data["metadata"] = {}
            
            valid_objects[obj_id] = obj_data
        
        self.objects = valid_objects
        
        if skipped_count > 0:
            print(f"⚠️ Пропущено объектов со старой структурой: {skipped_count}")
            # Сохраняем только валидные объекты
            if valid_objects:
                self.save()
    
    def register_object(self, name: str, features: dict, image_shape: tuple, metadata: dict = None) -> str:
        """
        Регистрация нового объекта
        
        Args:
            name: Имя объекта (задается пользователем)
            features: Словарь с признаками {des, kp, ...}
            image_shape: Размер изображения (height, width)
            metadata: Дополнительные метаданные
        
        Returns:
            str: Уникальный ID объекта
        """
        # Генерация уникального ID
        object_id = str(uuid.uuid4())
        
        # Создание записи объекта
        object_data = {
            "id": object_id,
            "name": name,
            "des": features.get("des"),
            "kp": features.get("kp"),
            "kp_num": len(features.get("kp", [])) if features.get("kp") else 0,
            "img": features.get("img"),
            "img_shape": image_shape,
            "hsv_profile": features.get("hsv_profile"),
            "registration_time": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.objects[object_id] = object_data
        
        # Автосохранение
        self.save()
        
        print(f"✅ Объект '{name}' зарегистрирован с ID: {object_id}")
        return object_id
    
    def get_object(self, object_id: str) -> Optional[dict]:
        """
        Получение данных объекта по ID
        
        Args:
            object_id: Уникальный ID объекта
        
        Returns:
            dict: Данные объекта или None если не найден
        """
        return self.objects.get(object_id)
    
    def get_object_by_name(self, name: str) -> Optional[dict]:
        """
        Получение объекта по имени (возвращает первый найденный)
        
        Args:
            name: Имя объекта
        
        Returns:
            dict: Данные объекта или None если не найден
        """
        for obj in self.objects.values():
            if obj.get("name") == name:
                return obj
        return None
    
    def list_objects(self) -> List[dict]:
        """
        Получение списка всех объектов
        
        Returns:
            list: Список словарей с базовой информацией об объектах
        """
        return [
            {
                "id": obj["id"],
                "name": obj.get("name", "Unknown"),
                "kp_num": obj.get("kp_num", 0),
                "img_shape": obj.get("img_shape"),
                "registration_time": obj.get("registration_time")
            }
            for obj in self.objects.values()
        ]
    
    def update_object(self, object_id: str, **kwargs) -> bool:
        """
        Обновление данных объекта
        
        Args:
            object_id: Уникальный ID объекта
            **kwargs: Поля для обновления
        
        Returns:
            bool: True если объект найден и обновлен
        """
        if object_id not in self.objects:
            return False
        
        self.objects[object_id].update(kwargs)
        self.save()
        return True
    
    def delete_object(self, object_id: str) -> bool:
        """
        Удаление объекта из реестра
        
        Args:
            object_id: Уникальный ID объекта
        
        Returns:
            bool: True если объект найден и удален
        """
        if object_id not in self.objects:
            return False
        
        name = self.objects[object_id].get("name", "Unknown")
        del self.objects[object_id]
        self.save()
        print(f"✅ Объект '{name}' (ID: {object_id}) удален из реестра")
        return True
    
    def save(self):
        """Сохранение реестра в хранилище"""
        self.storage.save_features(self.objects)
    
    def get_all_for_detection(self) -> Dict[str, dict]:
        """
        Получение всех объектов в формате для детектора
        (включая ключевые точки, если они есть в памяти)
        
        Returns:
            dict: {object_id: {des, kp, img_shape, name, ...}}
        """
        return self.objects.copy()
    


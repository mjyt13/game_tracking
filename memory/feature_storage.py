"""
Модуль сохранения и загрузки признаков объектов
Сериализация/десериализация данных в pickle файл
"""

import pickle
import numpy as np
from config import settings


class FeatureStorage:
    """Управление персистентным хранилищем признаков"""
    
    def __init__(self, memory_file=None):
        """
        Инициализация хранилища
        
        Args:
            memory_file: Путь к файлу памяти (если None, из settings)
        """
        self.memory_file = memory_file or settings.FEATURE_MEMORY_FILE
    
    def save_features(self, features_dict):
        """
        Сохранение словаря признаков в файл
        
        Args:
            features_dict: Словарь {object_id: {id, name, des, kp_num, img_shape, ...}}
        
        Returns:
            bool: True при успешном сохранении
        """
        try:
            import os
            import cv2 as _cv2
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            data_to_save = {}
            for obj_id, obj_data in features_dict.items():
                if not isinstance(obj_data, dict) or "id" not in obj_data:
                    print(f"⚠️ Пропущен объект с невалидной структурой: {obj_id}")
                    continue

                # kp сериализуем как список кортежей (cv2.KeyPoint не всегда pickl-ится надёжно)
                entry = {k: v for k, v in obj_data.items() if k != 'kp'}
                if obj_data.get('kp'):
                    entry['kp_tuples'] = [
                        (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                        for kp in obj_data['kp']
                    ]
                data_to_save[obj_id] = entry
            
            with open(self.memory_file, 'wb') as f:
                pickle.dump(data_to_save, f)
            
            print(f"✅ Память фишек сохранена в {self.memory_file}")
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения памяти: {e}")
            return False
    
    def load_features(self):
        """
        Загрузка словаря признаков из файла
        
        Returns:
            dict: Словарь {object_id: {id, name, des, kp_num, img_shape, ...}} или пустой dict
        """
        try:
            import cv2 as _cv2
            with open(self.memory_file, 'rb') as f:
                features_dict = pickle.load(f)

            # Восстанавливаем kp из kp_tuples
            for obj_data in features_dict.values():
                if 'kp_tuples' in obj_data:
                    obj_data['kp'] = [
                        _cv2.KeyPoint(x=pt[0], y=pt[1], size=sz, angle=ang,
                                      response=resp, octave=int(oct), class_id=int(cid))
                        for (pt, sz, ang, resp, oct, cid) in obj_data['kp_tuples']
                    ]

            print(f"✅ Память фишек успешно загружена из {self.memory_file}")
            return features_dict
        except FileNotFoundError:
            print(f"⚠️ Файл памяти {self.memory_file} не найден. Создаем новую пустую память.")
            return {}
        except Exception as e:
            print(f"❌ Ошибка загрузки памяти: {e}")
            return {}
    
    def backup_features(self, backup_suffix=None):
        """
        Создание резервной копии файла памяти
        
        Args:
            backup_suffix: Суффикс для backup файла (если None, используется timestamp)
        
        Returns:
            bool: True при успешном backup
        """
        import os
        from datetime import datetime
        
        if not os.path.exists(self.memory_file):
            return False
        
        if backup_suffix is None:
            backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backup_file = f"{self.memory_file}.backup_{backup_suffix}"
        
        try:
            import shutil
            shutil.copy2(self.memory_file, backup_file)
            print(f"✅ Резервная копия создана: {backup_file}")
            return True
        except Exception as e:
            print(f"❌ Ошибка создания резервной копии: {e}")
            return False


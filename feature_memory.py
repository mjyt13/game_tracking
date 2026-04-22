import pickle
import numpy as np
from config import settings

def save_features(features_dict):
    """Сохраняет словарь признаков (дескрипторы и имена) в файл."""
    try:
        with open(settings.FEATURE_MEMORY_FILE, 'wb') as f:
            pickle.dump(features_dict, f)
        print(f"✅ Память фишек сохранена в {settings.FEATURE_MEMORY_FILE}")
    except Exception as e:
        print(f"❌ Ошибка сохранения памяти: {e}")

def load_features():
    """Загружает словарь признаков из файла."""
    try:
        with open(settings.FEATURE_MEMORY_FILE, 'rb') as f:
            features_dict = pickle.load(f)
        print(f"✅ Память фишек успешно загружена из {settings.FEATURE_MEMORY_FILE}")
        return features_dict
    except FileNotFoundError:
        print(f"⚠️ Файл памяти {settings.FEATURE_MEMORY_FILE} не найден. Создаем новую пустую память.")
        return {}
    except Exception as e:
        print(f"❌ Ошибка загрузки памяти: {e}")
        return {}

# Структура словаря:
# {
#    "Наперсток": {"des": np.array([...]), "kp_num": 500, "img_shape": (h, w)},
#    "Камешек":   {"des": np.array([...]), "kp_num": 350, "img_shape": (h, w)}
# }
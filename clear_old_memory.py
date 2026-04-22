"""
Скрипт для очистки старого файла памяти
Удаляет объекты старого формата (без ID) из game_features.pkl
"""

import os
from memory.feature_storage import FeatureStorage
from memory.object_registry import ObjectRegistry


def clear_old_memory():
    """Очистка старого файла памяти от объектов старого формата"""
    storage = FeatureStorage()
    registry = ObjectRegistry(storage)
    
    # Проверка наличия файла
    if not os.path.exists(storage.memory_file):
        print(f"ℹ️ Файл памяти {storage.memory_file} не найден. Нечего очищать.")
        return
    
    # Создание резервной копии
    print("📦 Создание резервной копии...")
    storage.backup_features()
    
    # Загрузка и валидация
    print("\n🔍 Проверка объектов...")
    objects = registry.list_objects()
    
    if not objects:
        print("ℹ️ В памяти нет объектов.")
        return
    
    print(f"✅ Найдено объектов в новом формате: {len(objects)}")
    
    # Сохранение только валидных объектов
    registry.save()
    print("\n✅ Очистка завершена. Старые объекты удалены, валидные сохранены.")


if __name__ == "__main__":
    print("="*50)
    print("🧹 ОЧИСТКА СТАРОГО ФАЙЛА ПАМЯТИ")
    print("="*50)
    print("\nЭтот скрипт удалит все объекты старого формата (без ID)")
    print("и сохранит только объекты в новом формате.\n")
    
    response = input("Продолжить? (yes/no): ").strip().lower()
    if response in ['yes', 'y', 'да', 'д']:
        clear_old_memory()
    else:
        print("❌ Отменено.")


from feature_memory import load_features


def print_existing_features():
    """Загружает и выводит информацию обо всех зарегистрированных фишках."""
    features = load_features()

    if not features:
        print("Память пуста: нет зарегистрированных фишек (game_features.pkl не найден или пуст).")
        return

    print("\n--- СОДЕРЖИМОЕ ПАМЯТИ ФИШЕК ---")
    for name, data in features.items():
        # Мы знаем, что в файле хранятся des, kp_num и img_shape
        num_descriptors = data.get("des").shape[0] if data.get("des") is not None else 0
        img_dims = data.get("img_shape", "Неизвестно")

        print(f"Фишка: '{name}'")
        print(f"  Количество дескрипторов (SIFT): {num_descriptors}")
        print(f"  Размер шаблона (В x Ш): {img_dims}")
        print("-" * 30)


if __name__ == '__main__':
    # Если Вы добавите этот блок в feature_memory.py, он будет работать как тестовый скрипт
    print_existing_features()
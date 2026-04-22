import cv2
import numpy as np

# --- Глобальные переменные для хранения шаблона ---
# Здесь будем хранить ключевые точки (kp) и дескрипторы (des) для запомненной фишки
TEMPLATE_KP = None
TEMPLATE_DES = None
TEMPLATE_IMAGE = None
registration_in_progress = False


# --- Функция для выбора области интереса (ROI) ---
# Позволяет пользователю мышью выбрать объект на кадре
def register_object(event, x, y, flags, param):
    global registration_in_progress, TEMPLATE_IMAGE

    # 1. Захватываем текущий кадр из параметра (param[0] - текущий кадр)
    current_frame = param[0]

    # 2. Обработка нажатия левой кнопки мыши
    if event == cv2.EVENT_LBUTTONDOWN and not registration_in_progress:
        # Устанавливаем флаг начала регистрации
        registration_in_progress = True
        print("Начало регистрации. Кликните мышью в верхнем левом углу объекта.")
        param[1] = (x, y)  # Сохраняем начальные координаты (x1, y1)

    # 3. Обработка отпускания левой кнопки мыши
    elif event == cv2.EVENT_LBUTTONUP and registration_in_progress:
        # Получаем конечные координаты (x2, y2)
        x1, y1 = param[1]
        x2, y2 = x, y

        # Обрезка кадра для получения изображения фишки (ROI - область интереса)
        TEMPLATE_IMAGE = current_frame[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

        if TEMPLATE_IMAGE.size > 0:
            print(f"✅ Фишка захвачена (Размер: {TEMPLATE_IMAGE.shape[1]}x{TEMPLATE_IMAGE.shape[0]})")

            # Сохраняем дескрипторы объекта (см. ниже)
            extract_features(TEMPLATE_IMAGE)

            # Отображаем запомненное изображение для проверки
            cv2.imshow("Registered Template", TEMPLATE_IMAGE)
        else:
            print("❌ Ошибка захвата: выбрана нулевая область.")

        registration_in_progress = False
        param[1] = None  # Сброс начальных координат


# --- Функция извлечения ключевых признаков ORB ---
def extract_features(image):
    global TEMPLATE_KP, TEMPLATE_DES

    # Инициализация детектора ORB
    orb = cv2.ORB_create(nfeatures=1000)  # Попробуем найти до 1000 признаков

    # Находим ключевые точки и дескрипторы
    kp, des = orb.detectAndCompute(image, None)

    if des is not None and len(kp) > 0:
        TEMPLATE_KP = kp
        TEMPLATE_DES = des
        print(f"   Дескрипторы ORB сохранены. Всего точек: {len(kp)}")
    else:
        print("   ⚠️ Не удалось найти достаточно ключевых точек.")
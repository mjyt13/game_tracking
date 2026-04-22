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
        print("Начало регистрации. отпустите кнопку мыши в верхнем левом углу объекта.")
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
# --- Основной цикл трекинга ---

def tracking_loop():
    global TEMPLATE_DES, TEMPLATE_KP

    # Инициализация захвата (предположим, что это IP-камера)
    ip_camera_url = 'http://192.168.0.102:4747/video'
    cap = cv2.VideoCapture(ip_camera_url)

    if not cap.isOpened():
        print("❌ ОШИБКА: Невозможно открыть поток. Пробуем локальную камеру (0)...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ ОШИБКА: Невозможно открыть локальную камеру. Выход.")
            return

    # Параметры для функции-обработчика мыши: [текущий кадр, (x1, y1)]
    mouse_handler_params = [None, None]

    # Создание окна и привязка функции-обработчика мыши
    cv2.namedWindow('Game Field')
    cv2.setMouseCallback('Game Field', register_object, mouse_handler_params)

    # Инициализация детектора ORB и сопоставителя признаков (Brute-Force Matcher)
    orb = cv2.ORB_create(nfeatures=1000)
    # Используем BFMatcher (Brute-Force) с метрикой NORM_HAMMING для ORB
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    print("\n--- СИСТЕМА ЗАПУЩЕНА ---")
    print("1. Поставьте фишку на поле.")
    print("2. В окне 'Game Field' выделите фишку мышью (клик-перетащить-отпустить).")
    print("3. Нажмите 'q' для выхода.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mouse_handler_params[0] = frame.copy()  # Передаем копию текущего кадра

        # --- ФАЗА ОТСЛЕЖИВАНИЯ: Если фишка уже запомнена ---
        if TEMPLATE_DES is not None:

            # Находим признаки на текущем кадре
            kp_frame, des_frame = orb.detectAndCompute(frame, None)

            if des_frame is not None:
                # 1. Сопоставление дескрипторов (Match)
                matches = bf.match(TEMPLATE_DES, des_frame)

                # 2. Сортировка по дистанции (лучшие сопоставления)
                matches = sorted(matches, key=lambda x: x.distance)

                # 3. Фильтрация лучших сопоставлений (например, берем 10% лучших)
                good_matches = matches[:int(len(matches) * 0.1)]

                # 4. Если найдено достаточно хороших совпадений
                if len(good_matches) > 10:  # Порог: минимум 10 совпадений для "узнавания"

                    # Извлечение координат точек
                    src_pts = np.float32([TEMPLATE_KP[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    # 5. Нахождение гомографии (преобразования) для определения положения объекта
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if M is not None:
                        # 6. Определение границ и центра объекта в текущем кадре
                        h, w = TEMPLATE_IMAGE.shape[:2]
                        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)

                        # Вывод ограничивающего контура
                        frame = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

                        # Вычисление центра объекта для передачи в игровую логику
                        center_x = int(np.mean(dst[:, 0, 0]))
                        center_y = int(np.mean(dst[:, 0, 1]))
                        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                        # --- ВАША ИГРОВАЯ ЛОГИКА ---
                        # print(f"Объект 'Наперсток' найден! Координаты центра: ({center_x}, {center_y})")
                        # Здесь Вы сравниваете (center_x, center_y) с клетками поля.
                        # ---------------------------

        # Вывод кадра
        cv2.imshow('Game Field', frame)

        # Условие выхода
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Очистка и завершение
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    tracking_loop()
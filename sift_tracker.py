import cv2
import numpy as np
from feature_memory import load_features, save_features

# --- Глобальные переменные ---
# Загружаем или инициализируем память фишек
GAME_FEATURES = load_features()
TRACKING_OBJECT_NAME = None

# Инициализация SIFT и FLANN Matcher
# SIFT используется для генерации дескрипторов
sift = cv2.SIFT_create()

# FLANN Matcher (Fast Library for Approximate Nearest Neighbors)
# Гораздо быстрее, чем Brute-Force, для SIFT дескрипторов
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # Количество проверок
flann = cv2.FlannBasedMatcher(index_params, search_params)


# --- Функция для выбора области интереса (ROI) ---
def register_object_roi(event, x, y, flags, param):
    """Обработчик событий мыши для выбора области интереса."""
    # param[0] - текущий кадр, param[1] - координаты (x1, y1), param[2] - имя фишки

    current_frame = param[0]

    if event == cv2.EVENT_LBUTTONDOWN and param[1] is None:
        print(f"\nНачало регистрации '{param[2]}'. Кликните в верхнем левом углу объекта.")
        param[1] = (x, y)  # Сохраняем начальные координаты (x1, y1)

    elif event == cv2.EVENT_LBUTTONUP and param[1] is not None:
        x1, y1 = param[1]
        x2, y2 = x, y

        # Обрезка кадра для получения изображения фишки
        template_img = current_frame[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

        if template_img.size > 0:
            name = param[2]

            # Извлечение SIFT признаков
            kp, des = sift.detectAndCompute(template_img, None)

            if des is not None and len(kp) >= 10:
                # !!! ИСПРАВЛЕНИЕ: Мы добавляем kp в память, чтобы использовать его в цикле трекинга
                GAME_FEATURES[name] = {
                    "des": des,
                    "kp": kp,  # <-- !!! ДОБАВЛЕНО: Теперь kp хранится в памяти во время сессии
                    "kp_num": len(kp),
                    "img_shape": template_img.shape[:2]
                }

                # При сохранении в файл, мы можем исключить kp, так как их сложнее сериализовать
                # Но для текущей сессии они нужны!

                # Создаем временную копию данных для сохранения, исключая 'kp'
                data_to_save = {n: {k: v for k, v in d.items() if k != 'kp'} for n, d in GAME_FEATURES.items()}
                save_features(data_to_save)
                print(f"✅ Фишка '{name}' успешно зарегистрирована! Признаков: {len(kp)}")
                cv2.imshow(f"Registered: {name}", template_img)
            else:
                print(f"❌ Ошибка: Не удалось найти достаточно признаков SIFT на объекте '{name}'.")
        else:
            print("❌ Ошибка захвата: выбрана нулевая область.")

        param[1] = None  # Сброс координат для следующей регистрации


# --- Главная функция трекинга и распознавания ---
def tracking_loop():
    global GAME_FEATURES, TRACKING_OBJECT_NAME

    # Настройка IP-камеры (или локальной камеры)
    ip_camera_url = 'http://192.168.0.101:4747/video'
    cap = cv2.VideoCapture(ip_camera_url)

    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ ОШИБКА: Невозможно открыть камеру. Выход.")
            return

    # Параметры для обработчика мыши: [текущий кадр, (x1, y1), имя_фишки_для_регистрации]
    mouse_handler_params = [None, None, "New_Piece"]

    cv2.namedWindow('Game Field')
    cv2.setMouseCallback('Game Field', register_object_roi, mouse_handler_params)

    print("\n--- СИСТЕМА SIFT ЗАПУЩЕНА ---")
    print("Используйте консоль для ввода имени фишки (например, 'Naperstok').")
    print("Нажмите 'r' для регистрации, 'q' для выхода.")

    # Флаг для режима регистрации
    registration_mode = False

    while True:
        ret, frame = cap.read()
        if not ret: break

        mouse_handler_params[0] = frame.copy()

        # Находим SIFT признаки на текущем кадре
        kp_frame, des_frame = sift.detectAndCompute(frame, None)

        if des_frame is not None:

            # --- ФАЗА РАСПОЗНАВАНИЯ ---
            best_match_name = None
            max_good_matches = 0
            best_M = None
            h_template, w_template = 0, 0

            # Проходим по всем запомненным фишкам в памяти
            for name, feature_data in GAME_FEATURES.items():

                # Сопоставление дескрипторов (Match)
                matches = flann.knnMatch(feature_data["des"], des_frame, k=2)

                # Применение теста Ратио (Ratio Test) для отсеивания плохих совпадений
                good_matches = []
                for match_pair in matches:
                    # !!! ИСПРАВЛЕНИЕ: Проверка, что найдено 2 соседа
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:  # 0.75 - общепринятый порог
                            good_matches.append(m)

                # Если найдено достаточно хороших совпадений
                if len(good_matches) > 15 and "kp" in feature_data:  # Порог для "узнавания"

                    # Извлечение координат точек
                    src_pts = np.float32([feature_data["kp"][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    # Нахождение гомографии (преобразования) для определения положения объекта
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if M is not None and len(good_matches) > max_good_matches:
                        max_good_matches = len(good_matches)
                        best_match_name = name
                        best_M = M
                        h_template, w_template = feature_data["img_shape"]
                        feature_data["kp"] = kp_frame  # Обновляем kp для следующей итерации трекинга (опционально)

            # --- ВЫВОД РЕЗУЛЬТАТА ---
            if best_match_name is not None and best_M is not None:
                # Определение границ и центра объекта
                pts = np.float32(
                    [[0, 0], [0, h_template - 1], [w_template - 1, h_template - 1], [w_template - 1, 0]]).reshape(-1, 1,
                                                                                                                  2)
                dst = cv2.perspectiveTransform(pts, best_M)

                # Вывод ограничивающего контура и имени
                frame = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, best_match_name, (int(dst[0][0][0]), int(dst[0][0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Вычисление центра объекта для игровой логики
                center_x = int(np.mean(dst[:, 0, 0]))
                center_y = int(np.mean(dst[:, 0, 1]))
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

                # --- ВАША ИГРОВАЯ ЛОГИКА ---
                # print(f"Объект '{best_match_name}' найден! Координаты: ({center_x}, {center_y})")
                # ---------------------------

        # Вывод кадра
        cv2.imshow('Game Field', frame)

        # Обработка пользовательского ввода
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            # Вход в режим регистрации
            new_name = input("Введите имя для новой фишки (например, 'Kamyshek'): ")
            mouse_handler_params[2] = new_name
            registration_mode = True
            print(f"Режим регистрации активен. Выделите {new_name} мышью.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    tracking_loop()
"""
Конфигурация системы сканирования фишек
"""

# Настройки камеры
# Приоритетный способ подключения: 'rtsp', 'usb', 'http'
CAMERA_PRIMARY = 'usb'

CAMERA_RTSP_URL = 'rtsp://192.168.0.102:8080/h264_ulaw.sdp'  # IP Webcam (Android)
CAMERA_HTTP_URL = 'http://192.168.0.102:4747/video'           # DroidCam HTTP
CAMERA_USB_INDEX = 2  # 0 — встроенная, 1 — DroidCam USB, 2 — OBS Virtual Camera

# Разрешение и fps USB-камеры (реальная пропускная способность USB 2.0 ~480 Мбит/с):
#   MJPG выкл, 1920×1080 → ~1 fps   (raw YUY2, ~6 Гбит/с — не влезает)
#   MJPG выкл, 1280×720  → ~5 fps
#   MJPG вкл,  1920×1080 → ~10 fps  (сжатый поток)
#   MJPG вкл,  1280×720  → ~30 fps  ← рекомендуется для USB
#   OBS Virtual Camera   → 60 fps независимо от разрешения (ограничение не применяется)
CAMERA_WIDTH  = 1280  # Запрашиваемое разрешение; камера округлит до ближайшего
CAMERA_HEIGHT = 720
CAMERA_FPS    = 30    # Целевой fps (запрос к драйверу; может игнорироваться)
CAMERA_MJPG   = True  # MJPG-поток вместо raw YUY2: в 10–30× больше fps по USB

# Выбор детектора: 'sift' или 'akaze'
# При смене детектора нужно перерегистрировать все фишки (дескрипторы несовместимы)
DETECTOR_TYPE = 'sift'

# Настройки файлов
# Каждый детектор хранит свои признаки отдельно — дескрипторы несовместимы между SIFT и AKAZE
FEATURE_MEMORY_FILE = f'items/game_features_{DETECTOR_TYPE}.pkl'

# Настройки SIFT
SIFT_FEATURES = 500         # Максимальное количество признаков для извлечения

# Настройки AKAZE (используются только при DETECTOR_TYPE = 'akaze')
AKAZE_THRESHOLD = 0.001     # Порог детектора; ниже → больше признаков, медленнее

# Настройки CLAHE (предобработка перед SIFT — усиление локального контраста)
CLAHE_CLIP_LIMIT = 2.0      # Предел клипинга контраста; выше → сильнее эффект, но больше шума
CLAHE_TILE_GRID = (8, 8)    # Размер блоков; меньше → локальнее эффект

# Настройки сопоставления признаков
MATCH_RATIO_THRESHOLD = 0.80  # Порог для ratio test
MIN_MATCHES_THRESHOLD = 8     # Минимальное количество совпадений для распознавания
RANSAC_THRESHOLD = 5.0        # Порог для RANSAC в findHomography

# Настройки FLANN Matcher (SIFT — KD-tree, вещественные дескрипторы)
FLANN_INDEX_KDTREE = 1
FLANN_TREES = 5
FLANN_CHECKS = 50

# Настройки FLANN Matcher (AKAZE — LSH, бинарные дескрипторы)
FLANN_INDEX_LSH = 6
FLANN_LSH_TABLE_NUMBER = 6
FLANN_LSH_KEY_SIZE = 12
FLANN_LSH_MULTI_PROBE_LEVEL = 1

# Настройки регистрации
MIN_REGISTRATION_KEYPOINTS = 10  # Минимум ключевых точек для успешной регистрации
REGISTRATION_SHOTS = 5           # Снимков на одну регистрацию; держать фишку неподвижно,
                                 # слегка меняя наклон (~10°) между снимками
REGISTRATION_FEATURES_PER_SHOT = 150  # Признаков, сохраняемых с каждого снимка (топ по response).
                                      # Итого: SHOTS × PER_SHOT ≈ 750 при 5 снимках.
                                      # Больше → лучше покрытие ракурсов, медленнее детекция.
REGISTRATION_MODE = 'drag'            # Режим захвата снимков:
                                      #   'tilt'       — ROI фиксирован, наклонять объект вручную
                                      #   'click'      — кликнуть на объект; кроп roi_w×roi_h вокруг клика
                                      #   'drag'       — drag-выделение заново для каждого снимка
                                      #   'background' — автовыделение через вычитание фона (TODO: доинтегрировать UX)

# Профилирование (детальный breakdown proc time по операциям)
# Включить когда система стабильно работает; [P] в окне сохраняет baseline в CSV
PROFILING_ENABLED = False

# HSV-верификация цвета (постфильтр после SIFT — отклоняет ложные детекции по цвету)
# Требует перерегистрации фишек для записи hsv_profile в память
HSV_VERIFICATION = True   # включить/отключить
HSV_CROP_RATIO   = 0.5    # 0.5 = берём центральные 50% bbox по каждой оси (25% площади)
HSV_H_TOLERANCE  = 20     # допуск по оттенку (0–179, с учётом цикличности красного)
HSV_S_TOLERANCE  = 60     # допуск по насыщенности
HSV_V_TOLERANCE  = 60     # допуск по яркости (ключевой: чёрный V~40 vs красный V~130 → Δ=90)

# Настройки UI
WINDOW_NAME = 'Game Field'
DISPLAY_FPS = True
DISPLAY_SCALE = 0.75  # Масштаб окна отображения; обработка всегда на полном разрешении
THUMBNAIL_SIZE = 150       # Размер миниатюры объекта (пиксели)
THUMBNAIL_MAX_COLS = 4     # Максимальное число колонок в окне миниатюр
CAMERA_FLUSH_FRAMES = 5    # Кадров для сброса буфера после блокирующего окна


# Game Piece Scanner

Система компьютерного зрения для идентификации и трекинга игровых фишек на поле.
Стек: Python · OpenCV (SIFT/AKAZE, CLAHE, FLANN, Homography) · NumPy

---

## Требования

- Python 3.13
- Windows (протестировано на Windows 11)
- Камера: USB-вебкамера, DroidCam, или телефон через scrcpy + OBS Virtual Camera

---

## Установка

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

---

## Настройка камеры

### Вариант A — scrcpy + OBS Virtual Camera (рекомендуется)

1. Включи отладку по USB на телефоне, подключи по USB
2. Запусти scrcpy с камерой телефона:
   ```
   scrcpy --video-source=camera --camera-facing=back --no-audio
   ```
3. В OBS Studio: добавь источник **«Захват окна»** → выбери окно scrcpy
4. Нажми **«Запуск виртуальной камеры»** (панель Управление)
5. Найди индекс OBS-камеры:
   ```bash
   .venv/Scripts/python -c "
   import cv2
   for i in range(5):
       cap = cv2.VideoCapture(i)
       ret, frame = cap.read()
       if ret: print(i, int(cap.get(3)), 'x', int(cap.get(4)))
       cap.release()
   "
   ```
6. Укажи найденный индекс в `config/settings.py` → `CAMERA_USB_INDEX`

### Вариант B — DroidCam USB

1. Установи DroidCam на ПК и телефон, подключи по USB
2. В `config/settings.py`: раскомментируй строку с `cv2.CAP_DSHOW` в `camera_manager.py`
3. Укажи `CAMERA_USB_INDEX = 1` (обычно)

---

## Запуск

```bash
# bash / git bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python -u run_scanner.py 2>&1 | tee scanner.log
```

---

## Управление

| Клавиша / Кнопка | Действие |
|---|---|
| **R** | Зарегистрировать фишку (ввод имени → выделить мышью) |
| **L** | Список зарегистрированных фишек |
| **D** | Удалить фишку |
| **I** | Просмотр изображений зарегистрированных фишек |
| **Q** | Выход |
| **ESC** | Отменить текущий ввод / регистрацию |

---

## Ключевые настройки (`config/settings.py`)

| Параметр | По умолчанию | Описание |
|---|---|---|
| `DETECTOR_TYPE` | `'sift'` | Детектор признаков: `'sift'` или `'akaze'` |
| `CAMERA_WIDTH/HEIGHT` | `1280 / 720` | Разрешение захвата. При смене — перерегистрировать фишки |
| `CAMERA_MJPG` | `True` | MJPG-поток: в 10–30× больше fps по USB vs raw YUY2 |
| `CAMERA_FPS` | `30` | Целевой fps (запрос к драйверу) |
| `DISPLAY_SCALE` | `0.75` | Масштаб окна; обработка всегда на полном разрешении |
| `MATCH_RATIO_THRESHOLD` | `0.80` | Порог ratio test; ниже → строже, меньше ложных срабатываний |
| `MIN_MATCHES_THRESHOLD` | `8` | Мин. совпадений для распознавания фишки |

### Fps/разрешение по USB

| Режим | 1280×720 | 1920×1080 |
|---|---|---|
| Raw YUY2 (`CAMERA_MJPG=False`) | ~5 fps | ~1 fps |
| MJPG (`CAMERA_MJPG=True`) | ~30 fps | ~10 fps |
| OBS Virtual Camera | 60 fps | 60 fps |

---

## Файлы памяти

Признаки фишек хранятся в `items/game_features_{detector}.pkl`.
При смене `DETECTOR_TYPE` нужно перерегистрировать все фишки — дескрипторы несовместимы.

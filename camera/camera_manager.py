"""
Модуль управления камерой
Поддерживает RTSP, USB и HTTP-подключения с автоматическим перебором
"""

import cv2
from config import settings

# Фиксированный порядок резервных вариантов
_FALLBACK_ORDER = ['rtsp', 'usb', 'http']


class CameraManager:
    """Управление видеопотоком с камеры"""

    def __init__(self):
        self.cap = None
        self.is_initialized = False
        self.active_mode = None

    def initialize(self):
        """
        Открывает камеру, начиная с CAMERA_PRIMARY, затем перебирает остальные.

        Returns:
            bool: True если камера успешно открыта
        """
        order = [settings.CAMERA_PRIMARY] + [
            m for m in _FALLBACK_ORDER if m != settings.CAMERA_PRIMARY
        ]

        for mode in order:
            if self._try_open(mode):
                self.active_mode = mode
                self.is_initialized = True
                print(f"✅ Камера открыта через {mode}")
                return True
            print(f"⚠️ Способ '{mode}' недоступен, пробуем следующий...")

        print("❌ ОШИБКА: Не удалось открыть камеру ни одним способом")
        return False

    def _try_open(self, mode):
        """Попытка открыть камеру указанным способом."""
        try:
            if mode == 'rtsp':
                cap = cv2.VideoCapture(settings.CAMERA_RTSP_URL)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            elif mode == 'usb':
                cap = cv2.VideoCapture(settings.CAMERA_USB_INDEX)
                # DroidCam USB требует CAP_DSHOW — раскомментировать при использовании DroidCam:
                # cap = cv2.VideoCapture(settings.CAMERA_USB_INDEX, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if settings.CAMERA_MJPG:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, settings.CAMERA_FPS)
            elif mode == 'http':
                cap = cv2.VideoCapture(settings.CAMERA_HTTP_URL)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                return False

            if cap.isOpened():
                self.cap = cap
                return True

            cap.release()
            return False
        except Exception:
            return False

    def read_frame(self):
        """
        Чтение кадра с камеры.

        Returns:
            tuple: (success: bool, frame: np.ndarray) или (False, None) при ошибке
        """
        if not self.is_initialized or self.cap is None:
            return False, None
        return self.cap.read()

    def release(self):
        """Освобождение ресурсов камеры"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.is_initialized = False
            print("✅ Камера освобождена")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

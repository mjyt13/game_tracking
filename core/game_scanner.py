"""
Главный координатор системы сканирования фишек
Объединяет все модули: камера, сканер, память, трекер
"""

import cv2
import datetime
import numpy as np
from contextlib import nullcontext
from typing import Optional

from camera.camera_manager import CameraManager
from scanner.feature_extractor import FeatureExtractor
from scanner.object_detector import ObjectDetector
from memory.object_registry import ObjectRegistry
from memory.feature_storage import FeatureStorage
from tracker.object_tracker import ObjectTracker
from projector.projector_controller import ProjectorController
from metrics.metrics_tracker import MetricsTracker
from metrics.profiler import Profiler
from config import settings


class GameScanner:
    """Главный класс системы сканирования и трекинга фишек"""
    
    def __init__(self):
        """Инициализация всех компонентов системы"""
        # Компоненты системы
        self.camera = CameraManager()
        self.feature_extractor = FeatureExtractor()
        self.object_detector = ObjectDetector()
        self.storage = FeatureStorage()
        self.registry = ObjectRegistry(self.storage)
        self.tracker = ObjectTracker()
        self.projector = ProjectorController()
        
        # Состояние регистрации
        self.registration_mode = False
        self.registration_name = "New_Piece"
        self.registration_start = None
        self.mouse_handler_params = [None, None, None]  # [frame, (x1, y1), name]
        self._drag_current: tuple | None = None  # текущая позиция мыши при drag первого снимка
        
        # Метрики и UI
        self.metrics = MetricsTracker()
        self.profiler = Profiler() if settings.PROFILING_ENABLED else None
        self._button_rects = []   # вычисляются каждый кадр
        self._pending_key = None  # клик по кнопке → эмуляция клавиши

        # Текстовый ввод в оверлее (заменяет блокирующий input())
        self._text_input_mode = None   # "register" | "delete_select" | None
        self._text_input_buffer = ""
        self._delete_list = []

        # Масштаб отображения (обработка всегда на полном разрешении)
        self._display_scale = settings.DISPLAY_SCALE

        # Трекинг изменений детекций для логирования
        self._prev_active_ids: set = set()
        self._active_names: dict = {}  # object_id → name, для логирования исчезновений

        # Отложенная регистрация: callback → основной цикл (избегает реентрантного waitKey)
        self._pending_registration: tuple | None = None  # (roi_image, name, roi_coords)

        # Флаги
        self.running = False
    
    def _log(self, msg: str):
        """Вывод с временной меткой — попадает в stdout → scanner.log."""
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")


    def _mouse_callback(self, event, x, y, flags, param):
        # Координаты события — в пространстве отображения; переводим в пространство полного кадра
        s = self._display_scale
        x = int(x / s)
        y = int(y / s)
        """
        Обработчик событий мыши для регистрации объектов
        
        Args:
            event: Тип события мыши
            x, y: Координаты мыши
            flags: Флаги события
            param: Параметры [frame, (x1, y1), name]
        """
        # Клик мышью отменяет текстовый ввод в оверлее
        if self._text_input_mode is not None and event == cv2.EVENT_LBUTTONDOWN:
            self._text_input_mode = None
            self._text_input_buffer = ""
            print("❌ Ввод отменён.")
            return

        # Клики по кнопкам управления (вне режима регистрации)
        if not self.registration_mode and event == cv2.EVENT_LBUTTONDOWN:
            for btn in self._button_rects:
                x1, y1, x2, y2 = btn["rect"]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self._pending_key = btn["key"]
                    return

        if not self.registration_mode:
            return

        current_frame = param[0]

        if event == cv2.EVENT_LBUTTONDOWN and param[1] is None:
            print(f"\n📝 Начало регистрации '{param[2]}'. Кликните и перетащите для выделения объекта.")
            param[1] = (x, y)
            self._drag_current = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and param[1] is not None:
            self._drag_current = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and param[1] is not None:
            x1, y1 = param[1]
            x2, y2 = x, y
            
            # Нормализация координат
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Извлечение ROI
            roi_image = current_frame[y1:y2, x1:x2]

            if roi_image.size > 0:
                # Передаём в основной цикл — нельзя вызывать cv2.waitKey/imshow из callback
                self._pending_registration = (roi_image.copy(), param[2], (x1, y1, x2, y2))
            else:
                print("❌ Ошибка: выбрана нулевая область.")

            param[1] = None
            self._drag_current = None
            self.registration_mode = False
    
    @staticmethod
    def _compute_hsv_profile(roi_bgr: np.ndarray, crop_ratio: float = 0.5) -> np.ndarray | None:
        """Медиана HSV по центральному кропу ROI (crop_ratio=0.5 → центральные 25% площади)."""
        h, w = roi_bgr.shape[:2]
        if h < 4 or w < 4:
            return None
        cy, cx = h // 2, w // 2
        dh = max(1, int(h * crop_ratio / 2))
        dw = max(1, int(w * crop_ratio / 2))
        crop = roi_bgr[cy - dh:cy + dh, cx - dw:cx + dw]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return np.median(hsv.reshape(-1, 3), axis=0)  # [H, S, V]

    def _filter_by_color(self, frame_bgr: np.ndarray, detections: list, registered_objects: dict) -> list:
        """Отклонить детекции, чей HSV-профиль не совпадает с сохранённым при регистрации."""
        if not settings.HSV_VERIFICATION:
            return detections
        result = []
        for det in detections:
            stored = registered_objects.get(det["object_id"], {}).get("hsv_profile")
            if stored is None:  # нет профиля (старая регистрация) — пропускаем без фильтра
                result.append(det)
                continue
            corners = det["corners"]
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            dw = max(1, int((max(xs) - min(xs)) * settings.HSV_CROP_RATIO / 2))
            dh = max(1, int((max(ys) - min(ys)) * settings.HSV_CROP_RATIO / 2))
            fh, fw = frame_bgr.shape[:2]
            crop = frame_bgr[max(0, cy - dh):min(fh, cy + dh), max(0, cx - dw):min(fw, cx + dw)]
            if crop.size == 0:
                result.append(det)
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            current = np.median(hsv.reshape(-1, 3), axis=0)
            h_diff = min(abs(float(current[0]) - float(stored[0])),
                         180 - abs(float(current[0]) - float(stored[0])))
            if (h_diff                          <= settings.HSV_H_TOLERANCE and
                    abs(float(current[1]) - float(stored[1])) <= settings.HSV_S_TOLERANCE and
                    abs(float(current[2]) - float(stored[2])) <= settings.HSV_V_TOLERANCE):
                result.append(det)
        return result

    def _capture_shots(self, first_roi: np.ndarray, coords: tuple) -> list[np.ndarray]:
        """Захватить REGISTRATION_SHOTS кадров. Режим определяется settings.REGISTRATION_MODE."""
        x1, y1, x2, y2 = coords
        shots = [first_roi]
        total = settings.REGISTRATION_SHOTS
        mode = settings.REGISTRATION_MODE

        for i in range(1, total):
            if mode == 'click':
                self._log(f"  снимок {i + 1}/{total}: кликните на объект (ESC=стоп)")
                roi = self._capture_click_shot(x2 - x1, y2 - y1, i + 1, total)
            elif mode == 'drag':
                self._log(f"  снимок {i + 1}/{total}: выделите объект мышью (ESC=стоп)")
                roi = self._capture_drag_shot(i + 1, total)
            else:  # 'tilt'
                self._log(f"  снимок {i + 1}/{total}: наклоните ~10°, нажмите любую клавишу в окне (ESC=стоп)")
                roi = self._capture_tilt_shot(x1, y1, x2, y2, i + 1, total)

            if roi is None:
                self._log(f"  захват остановлен на снимке {i + 1} (ESC)")
                return shots
            if roi.size > 0:
                shots.append(roi)
                self._log(f"  снимок {i + 1} захвачен")

        return shots

    def _capture_tilt_shot(self, x1: int, y1: int, x2: int, y2: int,
                           shot_num: int, total: int) -> np.ndarray | None:
        """Живой просмотр → любая клавиша фиксирует кадр. ROI вырезается по coords."""
        while True:
            ret, preview = self.camera.read_frame()
            if ret:
                hint = f"Shot {shot_num}/{total}: tilt ~10 deg | any key = capture | ESC = stop"
                cv2.putText(preview, hint, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                if self._display_scale != 1.0:
                    dw = int(preview.shape[1] * self._display_scale)
                    dh = int(preview.shape[0] * self._display_scale)
                    preview = cv2.resize(preview, (dw, dh))
                cv2.imshow(settings.WINDOW_NAME, preview)
            k = cv2.waitKey(50) & 0xFF
            if k == 27:
                return None
            if k != 255:
                break
        ret, frame = self.camera.read_frame()
        if not ret:
            return None
        roi = frame[y1:y2, x1:x2]
        return roi.copy() if roi.size > 0 else None

    def _capture_click_shot(self, roi_w: int, roi_h: int,
                            shot_num: int, total: int) -> np.ndarray | None:
        """Клик на объект → кроп roi_w×roi_h вокруг точки клика."""
        state: dict = {'pt': None}

        def cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONUP:
                s = self._display_scale
                state['pt'] = (int(x / s), int(y / s))

        cv2.setMouseCallback(settings.WINDOW_NAME, cb)
        hint = f"Shot {shot_num}/{total}: click on object | ESC=stop"
        try:
            while state['pt'] is None:
                ret, preview = self.camera.read_frame()
                if ret:
                    cv2.putText(preview, hint, (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    if self._display_scale != 1.0:
                        dw = int(preview.shape[1] * self._display_scale)
                        dh = int(preview.shape[0] * self._display_scale)
                        preview = cv2.resize(preview, (dw, dh))
                    cv2.imshow(settings.WINDOW_NAME, preview)
                k = cv2.waitKey(50) & 0xFF
                if k == 27:
                    return None
            cx, cy = state['pt']
            ret, frame = self.camera.read_frame()
            if not ret:
                return None
            x1 = max(0, cx - roi_w // 2)
            y1 = max(0, cy - roi_h // 2)
            x2 = min(frame.shape[1], x1 + roi_w)
            y2 = min(frame.shape[0], y1 + roi_h)
            roi = frame[y1:y2, x1:x2]
            return roi.copy() if roi.size > 0 else None
        finally:
            cv2.setMouseCallback(settings.WINDOW_NAME, self._mouse_callback, self.mouse_handler_params)

    def _capture_drag_shot(self, shot_num: int, total: int) -> np.ndarray | None:
        """Drag-выделение для каждого снимка отдельно."""
        state: dict = {'start': None, 'cur': None, 'done': False}

        def cb(event, x, y, flags, param):
            s = self._display_scale
            rx, ry = int(x / s), int(y / s)
            if event == cv2.EVENT_LBUTTONDOWN:
                state['start'] = (rx, ry); state['cur'] = (rx, ry); state['done'] = False
            elif event == cv2.EVENT_MOUSEMOVE and state['start'] and not state['done']:
                state['cur'] = (rx, ry)
            elif event == cv2.EVENT_LBUTTONUP and state['start']:
                state['cur'] = (rx, ry); state['done'] = True

        cv2.setMouseCallback(settings.WINDOW_NAME, cb)
        hint = f"Shot {shot_num}/{total}: drag to select | ESC=stop"
        try:
            while not state['done']:
                ret, frame = self.camera.read_frame()
                if not ret:
                    continue
                preview = frame.copy()
                if state['start'] and state['cur']:
                    cv2.rectangle(preview, state['start'], state['cur'], (0, 255, 255), 2)
                cv2.putText(preview, hint, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                if self._display_scale != 1.0:
                    dw = int(preview.shape[1] * self._display_scale)
                    dh = int(preview.shape[0] * self._display_scale)
                    preview = cv2.resize(preview, (dw, dh))
                cv2.imshow(settings.WINDOW_NAME, preview)
                k = cv2.waitKey(50) & 0xFF
                if k == 27:
                    return None
            if not state['start'] or not state['cur']:
                return None
            x1 = min(state['start'][0], state['cur'][0])
            y1 = min(state['start'][1], state['cur'][1])
            x2 = max(state['start'][0], state['cur'][0])
            y2 = max(state['start'][1], state['cur'][1])
            if (x2 - x1) < 2 or (y2 - y1) < 2:
                return None
            ret, frame = self.camera.read_frame()
            if not ret:
                return None
            return frame[y1:y2, x1:x2].copy()
        finally:
            cv2.setMouseCallback(settings.WINDOW_NAME, self._mouse_callback, self.mouse_handler_params)

    def _capture_background_shots(self, name: str, n: int) -> list[np.ndarray]:
        """
        Режим 'background': захват N снимков с автовыделением через вычитание фона.
        Объект размещается в разных точках поля — без наклона, без клика на пиксель.

        ИНТЕГРАЦИЯ (TODO): вызывается вместо ROI-регистрации.
        Требует нового флага _pending_background в tracking_loop:
          - start_registration(): если REGISTRATION_MODE == 'background',
            установить _pending_background = name (не registration_mode = True)
          - tracking_loop(): обработать _pending_background → вызвать этот метод,
            передать результат напрямую в _register_object_from_roi(shots[0], name, None)
            с предварительным вызовом self.feature_extractor на каждый shot.
        """
        print(f"[background] Уберите все объекты с поля, нажмите любую клавишу...")
        while True:
            ret, preview = self.camera.read_frame()
            if ret:
                cv2.putText(preview, "Remove all objects, press any key", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2, cv2.LINE_AA)
                if self._display_scale != 1.0:
                    dw = int(preview.shape[1] * self._display_scale)
                    dh = int(preview.shape[0] * self._display_scale)
                    preview = cv2.resize(preview, (dw, dh))
                cv2.imshow(settings.WINDOW_NAME, preview)
            k = cv2.waitKey(50) & 0xFF
            if k == 27:
                return []
            if k != 255:
                break
        _, bg_frame = self.camera.read_frame()
        bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
        k_close = np.ones((15, 15), np.uint8)
        k_open = np.ones((5, 5), np.uint8)

        shots = []
        for i in range(n):
            self._log(f"  снимок {i + 1}/{n}: поместите '{name}' на новую позицию, нажмите любую клавишу (ESC=стоп)")
            while True:
                ret, preview = self.camera.read_frame()
                if ret:
                    hint = f"Shot {i + 1}/{n}: place '{name}' at new position | any key | ESC=stop"
                    cv2.putText(preview, hint, (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    if self._display_scale != 1.0:
                        dw = int(preview.shape[1] * self._display_scale)
                        dh = int(preview.shape[0] * self._display_scale)
                        preview = cv2.resize(preview, (dw, dh))
                    cv2.imshow(settings.WINDOW_NAME, preview)
                k = cv2.waitKey(50) & 0xFF
                if k == 27:
                    return shots
                if k != 255:
                    break
            ret, frame = self.camera.read_frame()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, bg_gray)
            _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                self._log(f"  снимок {i + 1}: объект не найден (пустая маска)")
                continue
            bx, by, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
            pad = 15
            x1 = max(0, bx - pad); y1 = max(0, by - pad)
            x2 = min(frame.shape[1], bx + bw + pad); y2 = min(frame.shape[0], by + bh + pad)
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                shots.append(roi.copy())
                self._log(f"  снимок {i + 1} захвачен (blob {bw}×{bh})")

        return shots

    def _register_object_from_roi(self, roi_image: np.ndarray, name: str, roi_coords: tuple = None):
        """Регистрация объекта из области интереса (один или несколько снимков)."""
        shots = (
            self._capture_shots(roi_image, roi_coords)
            if roi_coords and settings.REGISTRATION_SHOTS > 1
            else [roi_image]
        )

        per_shot = max(settings.MIN_REGISTRATION_KEYPOINTS,
                       settings.REGISTRATION_FEATURES_PER_SHOT)

        all_kp, all_des = [], []
        for shot in shots:
            kp, des = self.feature_extractor.extract_features(shot)
            if des is None or len(kp) == 0:
                continue
            # Берём per_shot лучших по силе отклика, чтобы суммарный размер не рос
            if len(kp) > per_shot:
                pairs = sorted(zip(kp, des), key=lambda x: x[0].response, reverse=True)
                kp, des = zip(*pairs[:per_shot])
                kp, des = list(kp), np.array(des)
            all_kp.extend(kp)
            all_des.append(des)

        if not all_des:
            self._log(f"ОШИБКА регистрации '{name}': ни один снимок не дал признаков")
            return

        combined_des = np.vstack(all_des)
        total_kp = len(all_kp)

        if total_kp < settings.MIN_REGISTRATION_KEYPOINTS:
            self._log(f"ОШИБКА регистрации '{name}': мало признаков ({total_kp})")
            return

        features = {"des": combined_des, "kp": all_kp, "img": roi_image,
                    "hsv_profile": self._compute_hsv_profile(roi_image)}
        object_id = self.registry.register_object(
            name=name,
            features=features,
            image_shape=roi_image.shape[:2],
            metadata={"source": "manual_registration", "shots": len(shots)},
        )

        obj_data = self.registry.get_object(object_id)
        if obj_data:
            obj_data["kp"] = all_kp

        self._log(f"РЕГИСТРАЦИЯ: '{name}' id={object_id[:8]} "
                  f"признаков={total_kp} снимков={len(shots)} per_shot={per_shot}")
    
    def _compute_button_rects(self, w: int, h: int) -> list:
        """Вычислить позиции кнопок управления для текущего размера кадра."""
        buttons = [
            {"label": "[R] Register", "key": ord("r")},
            {"label": "[L] List",     "key": ord("l")},
            {"label": "[D] Delete",   "key": ord("d")},
            {"label": "[I] Images",   "key": ord("i")},
            {"label": "[Q] Quit",     "key": ord("q")},
        ]
        margin = 8
        btn_h = 32
        btn_w = (w - margin * (len(buttons) + 1)) // len(buttons)
        y1, y2 = h - btn_h - 5, h - 5
        for i, btn in enumerate(buttons):
            x1 = margin + i * (btn_w + margin)
            btn["rect"] = (x1, y1, x1 + btn_w, y2)
        return buttons

    def _draw_ui(self, frame: np.ndarray):
        """Отрисовать оверлей с подсказкой и кнопки управления на кадре."""
        h, w = frame.shape[:2]

        # Полупрозрачный блок с инструкциями (правый верхний угол)
        lines = ["Controls:", "[R] Register", "[L] List", "[D] Delete", "[I] Images", "[Q] Quit"]
        box_w, box_h = 160, len(lines) * 20 + 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - box_w - 5, 5), (w - 5, box_h + 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        for i, txt in enumerate(lines):
            color = (200, 200, 200) if i > 0 else (120, 255, 120)
            cv2.putText(frame, txt, (w - box_w, 22 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Текстовый ввод в оверлее
        if self._text_input_mode is not None:
            prompt = "Name (Enter/ESC/click):" if self._text_input_mode == "register" \
                     else "Num or name (Enter/ESC/click):"
            display_text = f"{prompt} {self._text_input_buffer}_"
            (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            bx1, by1 = w // 2 - tw // 2 - 12, h // 2 - th - 12
            bx2, by2 = w // 2 + tw // 2 + 12, h // 2 + 12
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 200, 200), 1)
            cv2.putText(frame, display_text, (bx1 + 12, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        # Кнопки внизу кадра
        btn_colors = [(40, 100, 40), (40, 40, 100), (80, 40, 100), (40, 80, 100), (100, 40, 40)]
        self._button_rects = self._compute_button_rects(w, h)
        for btn, color in zip(self._button_rects, btn_colors):
            x1, y1, x2, y2 = btn["rect"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
            (tw, th), _ = cv2.getTextSize(btn["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(frame, btn["label"],
                        (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    def start_registration(self, name: str):
        """
        Начало режима регистрации нового объекта
        
        Args:
            name: Имя для нового объекта
        """
        self.registration_name = name
        self.registration_mode = True
        self.mouse_handler_params[2] = name
        self._log(f"РЕЖИМ РЕГИСТРАЦИИ: '{name}' — выделите объект мышью")
    
    def _show_thumbnails(self) -> None:
        """Показать окно с миниатюрами всех зарегистрированных объектов"""
        objects = self.registry.list_objects()
        if not objects:
            print("  (нет зарегистрированных объектов)")
            return

        thumb: int = settings.THUMBNAIL_SIZE
        pad: int = 10
        label_h: int = 22
        cols: int = min(settings.THUMBNAIL_MAX_COLS, len(objects))
        rows: int = (len(objects) + cols - 1) // cols
        cw: int = cols * thumb + (cols + 1) * pad
        ch: int = rows * (thumb + label_h) + (rows + 1) * pad
        canvas = np.zeros((ch, cw, 3), dtype=np.uint8)

        for i, meta in enumerate(objects):
            col, row = i % cols, i // cols
            x = pad + col * (thumb + pad)
            y = pad + row * (thumb + label_h + pad)

            obj = self.registry.get_object(meta["id"])
            img = obj.get("img") if obj else None
            if img is not None:
                canvas[y:y + thumb, x:x + thumb] = cv2.resize(img, (thumb, thumb))
            else:
                cv2.rectangle(canvas, (x, y), (x + thumb, y + thumb), (50, 50, 50), -1)
                cv2.putText(canvas, "no img", (x + 30, y + thumb // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

            cv2.putText(canvas, meta["name"][:20], (x, y + thumb + label_h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Registered Objects", canvas)
        # Опрашиваем с таймаутом — ловим и клавишу, и закрытие крестиком
        while True:
            k = cv2.waitKey(50) & 0xFF
            try:
                visible = cv2.getWindowProperty("Registered Objects", cv2.WND_PROP_VISIBLE)
            except cv2.error:
                visible = -1
            if k != 255 or visible < 1:
                break
        try:
            cv2.destroyWindow("Registered Objects")
        except cv2.error:
            pass
        for _ in range(settings.CAMERA_FLUSH_FRAMES):
            self.camera.read_frame()

    def _handle_key(self, key: int) -> bool:
        """
        Диспетчер клавиш с учётом активного режима.
        Возвращает True если нужно завершить цикл.
        """
        # Режим текстового ввода перехватывает все клавиши
        if self._text_input_mode is not None:
            if key == 27:  # ESC
                self._text_input_mode = None
                self._text_input_buffer = ""
                print("❌ Ввод отменён.")
            elif key == 13:  # Enter
                self._handle_text_input_confirm()
            elif key == 8:  # Backspace
                self._text_input_buffer = self._text_input_buffer[:-1]
            elif 32 <= key < 127:
                self._text_input_buffer += chr(key)
            return False

        # Режим выделения ROI — только ESC для отмены
        if self.registration_mode:
            if key == 27:  # ESC
                self.registration_mode = False
                self.mouse_handler_params[1] = None
                print("❌ Регистрация отменена.")
            return False

        # Обычные команды (недоступны пока активен любой другой режим)
        if key == ord('q'):
            print("\n🛑 Выход из системы...")
            return True
        elif key == ord('p') and self.profiler:
            self.profiler.save_baseline()
        elif key == ord('r'):
            self._text_input_mode = "register"
            self._text_input_buffer = ""
        elif key == ord('l'):
            objects = self.registry.list_objects()
            print("\n📋 Зарегистрированные объекты:")
            if objects:
                for i, obj in enumerate(objects, 1):
                    print(f"  {i}. {obj['name']} (ID: {obj['id'][:8]}..., Признаков: {obj['kp_num']})")
            else:
                print("  (нет зарегистрированных объектов)")
        elif key == ord('d'):
            objects = self.registry.list_objects()
            if not objects:
                print("  (нет зарегистрированных объектов)")
            else:
                print("\n🗑 Удаление фишки:")
                for i, obj in enumerate(objects, 1):
                    print(f"  {i}. {obj['name']} (ID: {obj['id'][:8]}..., Признаков: {obj['kp_num']})")
                self._delete_list = objects
                self._text_input_mode = "delete_select"
                self._text_input_buffer = ""
        elif key == ord('i'):
            self._show_thumbnails()
            self._pending_key = None  # игнорировать клики, накопленные во время просмотра
        return False

    def _handle_text_input_confirm(self) -> None:
        """Обработка подтверждения текстового ввода (Enter)"""
        value = self._text_input_buffer.strip()
        mode = self._text_input_mode
        self._text_input_mode = None
        self._text_input_buffer = ""

        if mode == "register":
            if value:
                self.start_registration(value)
            else:
                print("❌ Имя не может быть пустым.")
        elif mode == "delete_select":
            deleted = False
            # попытка по номеру
            try:
                idx = int(value) - 1
                if 0 <= idx < len(self._delete_list):
                    obj = self._delete_list[idx]
                    self.registry.delete_object(obj["id"])
                    self._log(f"УДАЛЕНИЕ: '{obj['name']}' id={obj['id'][:8]}")
                    deleted = True
                else:
                    print(f"❌ Номер вне диапазона: {value}")
            except ValueError:
                pass
            # попытка по имени
            if not deleted:
                for obj in self._delete_list:
                    if obj["name"] == value:
                        self.registry.delete_object(obj["id"])
                        self._log(f"УДАЛЕНИЕ: '{obj['name']}' id={obj['id'][:8]}")
                        deleted = True
                        break
            if not deleted and value:
                print(f"❌ Фишка '{value}' не найдена.")

    def tracking_loop(self):
        """Главный цикл трекинга и обработки кадров"""
        # Инициализация камеры
        self._log(f"СТАРТ системы  детектор={settings.DETECTOR_TYPE}  память={settings.FEATURE_MEMORY_FILE}")
        if not self.camera.initialize():
            self._log("ОШИБКА: не удалось инициализировать камеру")
            return
        
        # Создание окна и привязка обработчика мыши
        cv2.namedWindow(settings.WINDOW_NAME)
        cv2.setMouseCallback(settings.WINDOW_NAME, self._mouse_callback, self.mouse_handler_params)
        
        self.running = True
        frame_count = 0

        while self.running:
            proc_start = self.metrics.tick()

            # Чтение кадра
            p = self.profiler
            with (p.measure('camera') if p else nullcontext()):
                ret, frame = self.camera.read_frame()
            if not ret:
                print("⚠️ Не удалось прочитать кадр. Выход.")
                break

            frame_count += 1
            self.mouse_handler_params[0] = frame.copy()

            # Отложенная регистрация — запускается здесь, а не в callback
            if self._pending_registration is not None:
                roi_image, name, roi_coords = self._pending_registration
                self._pending_registration = None
                self._register_object_from_roi(roi_image, name, roi_coords)


            # Извлечение признаков из текущего кадра
            with (p.measure('extract') if p else nullcontext()):
                kp_frame, des_frame = self.feature_extractor.extract_features(frame)

            # Визуализация
            display_frame = frame.copy()

            if des_frame is not None and len(des_frame) > 0:
                # Получение всех зарегистрированных объектов
                registered_objects = self.registry.get_all_for_detection()

                if registered_objects:
                    # Детекция объектов
                    with (p.measure('detect') if p else nullcontext()):
                        detections = self.object_detector.detect_objects(
                            kp_frame, des_frame, registered_objects
                        )

                    detections = self._filter_by_color(frame, detections, registered_objects)

                    # Обновление трекера и метрик
                    with (p.measure('tracker') if p else nullcontext()):
                        self.tracker.update(detections)
                    self.metrics.update_detections(detections)

                    # Логирование изменений через трекер (дебаунс 30 кадров — без шума)
                    for d in detections:
                        self._active_names[d["object_id"]] = d["object_name"]
                    tracker_ids = {o.object_id for o in self.tracker.get_active_objects()}
                    for oid in tracker_ids - self._prev_active_ids:
                        self._log(f"ПОЯВИЛАСЬ: '{self._active_names.get(oid, oid[:8])}'")
                    for oid in self._prev_active_ids - tracker_ids:
                        self._log(f"ПРОПАЛА:   '{self._active_names.get(oid, oid[:8])}'")
                    self._prev_active_ids = tracker_ids
                    
                    # Визуализация детекций
                    for detection in detections:
                        object_id: str = detection["object_id"]
                        object_name: str = detection["object_name"]
                        center: tuple[int, int] = detection["center"]
                        corners: list[tuple[int, int]] = detection["corners"]
                        confidence: float = detection["confidence"]
                        matches_count: int = detection["matches_count"]

                        # Сглаженный центр из трекера
                        smooth_center: tuple[int, int] = (
                            self.tracker.get_object_position(object_id, smoothed=True) or center
                        )

                        # Сдвигаем углы bounding box к сглаженному центру
                        # corners — список кортежей (x,y), reshape даёт (4,1,2) для cv2.polylines
                        corners_int: np.ndarray = np.int32(corners).reshape(-1, 1, 2)
                        raw_cx: int = int(np.mean(corners_int[:, 0, 0]))
                        raw_cy: int = int(np.mean(corners_int[:, 0, 1]))
                        dx: int = smooth_center[0] - raw_cx
                        dy: int = smooth_center[1] - raw_cy
                        smooth_corners: np.ndarray = (
                            corners_int + np.array([[[dx, dy]]], dtype=np.int32)
                        )

                        cv2.polylines(display_frame, [smooth_corners], True, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.circle(display_frame, smooth_center, 5, (0, 0, 255), -1)

                        label = f"{object_name} ({confidence:.2f}, {matches_count})"
                        cv2.putText(
                            display_frame, label,
                            (smooth_corners[0][0][0], smooth_corners[0][0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                        )
                        
                        # Вывод в консоль (опционально, можно отключить для производительности)
                        # print(f"Объект '{object_name}' (ID: {object_id[:8]}...) найден! Центр: {center}, Уверенность: {confidence:.2f}")
            
            # Информация о режиме регистрации
            if self.registration_mode:
                cv2.putText(
                    display_frame, f"REGISTRATION MODE: {self.registration_name}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                )
                cv2.putText(
                    display_frame, "Click+drag to select  ESC=cancel",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
                )
                if self.mouse_handler_params[1] is not None and self._drag_current is not None:
                    cv2.rectangle(display_frame, self.mouse_handler_params[1],
                                  self._drag_current, (0, 255, 255), 2)
            
            # Обновление метрик и отрисовка UI
            active_count = len(self.tracker.get_active_objects())
            total_registered = len(self.registry.list_objects())
            self.metrics.record_proc(proc_start)
            self.metrics.camera_mode = self.camera.active_mode or "—"
            self.metrics.registered_count = total_registered

            with (p.measure('draw') if p else nullcontext()):
                self._draw_ui(display_frame)
            if self._display_scale != 1.0:
                dw = int(display_frame.shape[1] * self._display_scale)
                dh = int(display_frame.shape[0] * self._display_scale)
                display_frame = cv2.resize(display_frame, (dw, dh))
            cv2.imshow(settings.WINDOW_NAME, display_frame)
            self.metrics.render()

            # Обработка клавиатуры и кликов по кнопкам
            key = cv2.waitKey(1) & 0xFF
            if self._pending_key is not None:
                key = self._pending_key
                self._pending_key = None

            if p:
                p.next_frame()
            if self._handle_key(key):
                break
        
        # Очистка
        self.camera.release()
        cv2.destroyAllWindows()
        self._log("СТОП системы")
    
    def shutdown(self):
        """Корректное завершение работы системы"""
        self.running = False
        self.camera.release()
        self.registry.save()


def main():
    """Точка входа в программу"""
    scanner = GameScanner()
    try:
        scanner.tracking_loop()
    except KeyboardInterrupt:
        print("\n⚠️ Прервано пользователем")
    finally:
        scanner.shutdown()


if __name__ == "__main__":
    main()


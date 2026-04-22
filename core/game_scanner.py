"""
Главный координатор системы сканирования фишек
Объединяет все модули: камера, сканер, память, трекер
"""

import cv2
import numpy as np
from typing import Optional

from camera.camera_manager import CameraManager
from scanner.feature_extractor import FeatureExtractor
from scanner.object_detector import ObjectDetector
from memory.object_registry import ObjectRegistry
from memory.feature_storage import FeatureStorage
from tracker.object_tracker import ObjectTracker
from projector.projector_controller import ProjectorController
from metrics.metrics_tracker import MetricsTracker
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
        
        # Метрики и UI
        self.metrics = MetricsTracker()
        self._button_rects = []   # вычисляются каждый кадр
        self._pending_key = None  # клик по кнопке → эмуляция клавиши

        # Текстовый ввод в оверлее (заменяет блокирующий input())
        self._text_input_mode = None   # "register" | "delete_select" | None
        self._text_input_buffer = ""
        self._delete_list = []

        # Масштаб отображения (обработка всегда на полном разрешении)
        self._display_scale = settings.DISPLAY_SCALE

        # Флаги
        self.running = False
    
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
        
        elif event == cv2.EVENT_LBUTTONUP and param[1] is not None:
            x1, y1 = param[1]
            x2, y2 = x, y
            
            # Нормализация координат
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Извлечение ROI
            roi_image = current_frame[y1:y2, x1:x2]
            
            if roi_image.size > 0:
                name = param[2]
                self._register_object_from_roi(roi_image, name)
            else:
                print("❌ Ошибка: выбрана нулевая область.")
            
            param[1] = None  # Сброс координат
            self.registration_mode = False
    
    def _register_object_from_roi(self, roi_image: np.ndarray, name: str):
        """
        Регистрация объекта из области интереса
        
        Args:
            roi_image: Изображение области интереса
            name: Имя объекта
        """
        # Извлечение признаков
        keypoints, descriptors = self.feature_extractor.extract_features(roi_image)
        
        if descriptors is not None and len(keypoints) >= settings.MIN_REGISTRATION_KEYPOINTS:
            # Регистрация в реестре
            features = {
                "des": descriptors,
                "kp": keypoints,
                "img": roi_image
            }
            
            object_id = self.registry.register_object(
                name=name,
                features=features,
                image_shape=roi_image.shape[:2],
                metadata={"source": "manual_registration"}
            )
            
            # Обновление данных для детектора (ключевые точки нужны в памяти)
            obj_data = self.registry.get_object(object_id)
            if obj_data:
                obj_data["kp"] = keypoints
            
            print(f"✅ Фишка '{name}' успешно зарегистрирована! ID: {object_id}, Признаков: {len(keypoints)}")
            cv2.imshow(f"Registered: {name}", roi_image)
        else:
            print(f"❌ Ошибка: Не удалось найти достаточно признаков SIFT на объекте '{name}'.")
            print(f"   Найдено ключевых точек: {len(keypoints) if keypoints else 0}")
    
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
        print(f"📝 Режим регистрации активен. Выделите объект '{name}' мышью в окне 'Game Field'.")
    
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
                    self.registry.delete_object(self._delete_list[idx]["id"])
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
                        deleted = True
                        break
            if not deleted and value:
                print(f"❌ Фишка '{value}' не найдена.")

    def tracking_loop(self):
        """Главный цикл трекинга и обработки кадров"""
        # Инициализация камеры
        if not self.camera.initialize():
            print("❌ Не удалось инициализировать камеру. Выход.")
            return
        
        # Создание окна и привязка обработчика мыши
        cv2.namedWindow(settings.WINDOW_NAME)
        cv2.setMouseCallback(settings.WINDOW_NAME, self._mouse_callback, self.mouse_handler_params)
        
        self.running = True
        frame_count = 0

        while self.running:
            proc_start = self.metrics.tick()

            # Чтение кадра
            ret, frame = self.camera.read_frame()
            if not ret:
                print("⚠️ Не удалось прочитать кадр. Выход.")
                break
            
            frame_count += 1
            self.mouse_handler_params[0] = frame.copy()
            
            # Извлечение признаков из текущего кадра
            kp_frame, des_frame = self.feature_extractor.extract_features(frame)
            
            # Визуализация
            display_frame = frame.copy()
            
            if des_frame is not None and len(des_frame) > 0:
                # Получение всех зарегистрированных объектов
                registered_objects = self.registry.get_all_for_detection()
                
                if registered_objects:
                    # Детекция объектов
                    detections = self.object_detector.detect_objects(
                        kp_frame, des_frame, registered_objects
                    )
                    
                    # Обновление трекера и метрик
                    self.tracker.update(detections)
                    self.metrics.update_detections(detections)
                    
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
            
            # Обновление метрик и отрисовка UI
            active_count = len(self.tracker.get_active_objects())
            total_registered = len(self.registry.list_objects())
            self.metrics.record_proc(proc_start)
            self.metrics.camera_mode = self.camera.active_mode or "—"
            self.metrics.registered_count = total_registered

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

            if self._handle_key(key):
                break
        
        # Очистка
        self.camera.release()
        cv2.destroyAllWindows()
        print("✅ Система завершена")
    
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


"""
Главный координатор системы сканирования фишек
Объединяет все модули: камера, сканер, память, трекер
"""

import cv2
import datetime
import time
import numpy as np
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

from camera.camera_manager import CameraManager
from scanner.feature_extractor import FeatureExtractor
from scanner.object_detector import ObjectDetector
from scanner.background_subtractor import MOG2BackgroundLearner
from scanner.temporal_filter import TemporalStabilityFilter
from memory.object_registry import ObjectRegistry
from memory.feature_storage import FeatureStorage
from tracker.object_tracker import ObjectTracker
from projector.projector_controller import ProjectorController
from metrics.metrics_tracker import MetricsTracker
from metrics.profiler import Profiler
from config import settings
from config import game_rules
from game.turn_manager import TurnManager
from game.field_calibration import FieldCalibration
from game.game_engine import GameEngine
from game.game_state_manager import GameStateManager
from game.events import Player
from server import GameEventServer


class GameScanner:
    """Главный класс системы сканирования и трекинга фишек"""
    
    def __init__(self):
        """Инициализация всех компонентов системы"""
        # Компоненты системы
        self.camera = CameraManager()
        self._no_camera = False        # режим без камеры (гибрид): кадр-заглушка, игра идёт от симуляции
        self._placeholder = None
        self.feature_extractor = FeatureExtractor()
        self.object_detector = ObjectDetector()
        self.storage = FeatureStorage()
        self.registry = ObjectRegistry(self.storage)
        self.tracker = ObjectTracker(
            max_history=settings.TRACKER_MAX_HISTORY,
            inactive_threshold=settings.TRACKER_INACTIVE_FRAMES,
        )
        self.projector = ProjectorController()
        self.temporal_filter = TemporalStabilityFilter()
        self._web_reg_name: str | None = None
        self._web_reg_shots: list = []

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
        self._pending_background_name: str | None = None  # имя объекта для background-регистрации
        self._pending_load_save: str | None = None  # путь сохранения для загрузки

        # Очередь ходов
        self.turns = TurnManager()
        self._pending_player_name: str | None = None  # буфер между двумя шагами ввода игрока

        # Калибровка поля
        self.field = FieldCalibration(game_rules.PATH_TYPE)
        self._calibration_mode: bool = False
        self._calibration_corners: list[tuple[int, int]] = []
        self._grid_visible: bool = True  # видимость сетки

        # Игровой движок
        self.engine = GameEngine()
        self._chip_cells: dict[str, int] = {}  # chip_id → последняя известная клетка
        self._chip_cell_debounce: dict[str, tuple[int, int]] = {}  # chip_id → (cell, frame_count)
        self._turn_profile: dict | None = None

        # Сохранение состояния игры
        self.state_manager = GameStateManager()

        # HTTP сервер для событий
        self.http_server = GameEventServer()
        self.http_server.start()
        self._sync_chips_http()

        # Флаги
        self.running = False
    
    def _log(self, msg: str):
        """Вывод с временной меткой — попадает в stdout → scanner.log."""
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")

    def _sync_http_game_state(self) -> None:
        """Обновить текущее состояние игры для HTML/API."""
        turn_entry = self.turns.current()
        winner = self.engine.state.winner
        state = {
            "active": self.engine.state.active,
            "turn_state": self.engine.turn_state,
            "expected_cell": self.engine.expected_cell,
            "rule_target": self.engine.rule_target,
            "dice_roll": self.engine._last_dice_roll,
            "board_size": self.engine.state.board_size,
            "winner": winner.name if winner else None,
            "path_type": self.field.path_type,
            "current_turn": dict(turn_entry) if turn_entry else None,
            "ready_for_next": self.engine.is_active and self.engine.can_advance(),
            "sim_enabled": settings.SIMULATION_ENABLED,
            "players": [
                {
                    "name": p.name,
                    "chip_id": p.chip_id,
                    "chip_name": p.chip_name,
                    "cell": p.cell,
                    "skip_turns": p.skip_turns,
                }
                for p in self.engine.state.players
            ],
            # очередь ожидающих (до старта игры) — для экрана ожидания и админа
            "queue": [
                {"player": e["player"], "chip_name": e["chip_name"], "chip_id": e["chip_id"]}
                for e in self.turns.list_entries()
            ],
        }
        self.http_server.update_game_state(state)

    def _build_cell_colors(self) -> dict:
        """Сопоставить номер клетки → цвет заливки (BGR) по правилам поля + финиш."""
        colors: dict[int, tuple] = {}
        for num, rule in game_rules.CELL_RULES.items():
            c = settings.CELL_EFFECT_COLORS.get(rule.get("effect"))
            if c:
                colors[num] = c
        # Финишная клетка — последняя в маршруте
        if self.field.is_calibrated() and self.field.path_manager:
            finish_color = settings.CELL_EFFECT_COLORS.get("FINISH")
            if finish_color:
                colors[self.field.path_manager.get_board_size()] = finish_color
        return colors

    def _sync_chips_http(self) -> None:
        """Обновить списки фишек и миниатюры для /api/chips, /join и админа."""
        all_chips = self.registry.list_objects()
        assigned = {e["chip_id"] for e in self.turns.list_entries()}
        available = [{"id": c["id"], "name": c["name"]} for c in all_chips if c["id"] not in assigned]
        self.http_server.update_available_chips(available)
        self.http_server.update_all_chips([{"id": c["id"], "name": c["name"]} for c in all_chips])

        # Миниатюры фишек (JPEG) для веб-интерфейса
        images: dict[str, bytes] = {}
        for c in all_chips:
            obj = self.registry.get_object(c["id"])
            img = obj.get("img") if obj else None
            if img is not None and img.size > 0:
                thumb = cv2.resize(img, (settings.THUMBNAIL_SIZE, settings.THUMBNAIL_SIZE))
                ok, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    images[c["id"]] = buf.tobytes()
        self.http_server.update_chip_images(images)

    def _reset_if_finished(self) -> None:
        """Если прошлая игра завершена (есть победитель) — сбросить движок,
        чтобы старая надпись GAME OVER не висела во время подготовки новой игры."""
        if self.engine.state.winner and not self.engine.state.active:
            self.engine = GameEngine()
            self._chip_cells.clear()
            self._chip_cell_debounce.clear()
            self._turn_profile = None
            self._sync_http_game_state()

    def _end_game(self) -> None:
        """Стоп партии: сброс движка (без остановки приложения)."""
        self.engine = GameEngine()
        self._chip_cells.clear()
        self._chip_cell_debounce.clear()
        self._turn_profile = None
        self.turns.unfocus()
        self._log("СТОП ПАРТИИ: движок сброшен, система продолжает работу")
        self._sync_http_game_state()

    def _delete_chip_by_value(self, value: str) -> None:
        """Удалить фишку по id или имени (для веб-админа)."""
        objects = self.registry.list_objects()
        target = next((o for o in objects if o["id"] == value or o["name"] == value), None)
        if target:
            self.registry.delete_object(target["id"])
            self._log(f"УДАЛЕНИЕ [HTTP]: '{target['name']}' id={target['id'][:8]}")
            self._sync_chips_http()
            self._sync_http_game_state()
        else:
            self._log(f"⚠️ Фишка '{value}' не найдена для удаления")

    def _calibrate_from_web(self, corners_norm: list, cols: int, rows: int, frame) -> None:
        """Калибровка поля по 4 нормированным углам [x,y] из веба + размер сетки."""
        fh, fw = frame.shape[:2]
        try:
            corners_px = [(int(x * fw), int(y * fh)) for x, y in corners_norm]
        except (TypeError, ValueError):
            self._log("⚠️ Некорректные углы калибровки из веба")
            return
        if len(corners_px) != 4 or cols < 1 or rows < 1:
            self._log("⚠️ Калибровка из веба отклонена (нужно 4 угла, размер ≥1)")
            return
        self.field.set_corners(corners_px, cols, rows)
        self.field.save("items/field_calibration.json")
        self._grid_visible = True
        self._log(f"ПОЛЕ [HTTP]: откалибровано {cols}×{rows}, сохранено")
        self.http_server.emit_event("NOTICE", {"message": f"Поле откалибровано: {cols}×{rows}", "level": "info"})
        self._sync_http_game_state()

    def _set_path_from_web(self, ptype: str) -> None:
        """Сменить порядок обхода клеток (linear/snake/spiral) из веб-админа."""
        if self.engine.is_active:
            msg = "Нельзя менять маршрут во время партии. Остановите партию ([Стоп])."
            self._log(f"⚠️ {msg}")
            self.http_server.emit_event("NOTICE", {"message": msg, "level": "error"})
            return
        if not self.field.is_calibrated():
            msg = "Сначала откалибруйте поле, затем меняйте маршрут."
            self._log(f"⚠️ {msg}")
            self.http_server.emit_event("NOTICE", {"message": msg, "level": "error"})
            return
        self.field.set_path_type(ptype)
        self.field.save("items/field_calibration.json")
        self._log(f"МАРШРУТ [HTTP]: порядок клеток → {ptype}")
        self.http_server.emit_event("NOTICE", {"message": f"Маршрут поля: {ptype}", "level": "info"})
        self._sync_http_game_state()

    def _remove_player_by_value(self, value: str) -> None:
        """Удалить игрока из очереди по имени/номеру (для веб-админа). Синхронизирует движок."""
        chip_id = self.turns.remove(value)
        if not chip_id:
            self._log(f"⚠️ Игрок '{value}' не найден в очереди")
            return
        self._log(f"ОЧЕРЕДЬ [HTTP]: игрок удалён ('{value}')")
        self._sync_chips_http()
        if self.engine.is_active:
            player = self.engine.state.get_player(chip_id)
            if player:
                self.engine.state.players.remove(player)
                self._log(f"ENGINE: игрок '{player.name}' удалён из состояния")
            if self.engine._active_chip_id == chip_id:
                self.engine._active_chip_id = None
                self.engine._pending_extra_turn = None
                if self.engine.turn_state in ("WAITING_MOVE", "AWAITING_RULE"):
                    self.engine.turn_state = "TURN_DONE"
                self._log("⚠️ Удалённый игрок был активным. Очередь готова к продвижению.")
            self._turn_profile = None
        self._sync_http_game_state()

    def _forward_engine_event(self, event) -> None:
        """Отправить событие GameEngine в HTTP сервер"""
        if event:
            self.http_server.emit_event(event.type, event.data)
            # Лог значимых игровых событий в stdout (для анализа игрового цикла, раздел 4.4)
            if event.type in ("TURN_START", "RULE_TRIGGERED", "GAME_OVER", "CHIP_MOVED"):
                self._log(f"EVENT {event.type}: {event.data}")
            self._sync_http_game_state()

    def _start_turn_profile(self, chip_id: str, player_name: str, target_cell: int, phase: str) -> None:
        """Начать замер детекции активной фишки для текущей фазы хода."""
        if phase not in ("WAITING_MOVE", "AWAITING_RULE"):
            self._turn_profile = None
            return
        self._turn_profile = {
            "chip_id": chip_id,
            "player": player_name,
            "target_cell": target_cell,
            "phase": phase,
            "started_perf": time.perf_counter(),
            "started_ts": datetime.datetime.now(),
            "first_cell_perf": None,
            "first_cell_ts": None,
            "first_cell": None,
        }

    def _mark_first_cell_seen(self, chip_id: str, cell: int) -> None:
        """Зафиксировать первый кадр, где активная фишка обнаружена в целевой клетке."""
        profile = self._turn_profile
        if not profile or profile["chip_id"] != chip_id or profile["first_cell_ts"] is not None:
            return
        if cell != profile["target_cell"]:
            return
        profile["first_cell_perf"] = time.perf_counter()
        profile["first_cell_ts"] = datetime.datetime.now()
        profile["first_cell"] = cell
        self._log(
            f"TURN_TIMING target_cell player='{profile['player']}' phase={profile['phase']} "
            f"cell={cell} target={profile['target_cell']} at {profile['first_cell_ts'].strftime('%H:%M:%S.%f')[:-3]}"
        )

    def _finish_turn_profile(self, chip_id: str, cell: int, accepted_state: str) -> None:
        """Завершить замер, когда FSM приняла клетку и сменила состояние."""
        profile = self._turn_profile
        if not profile or profile["chip_id"] != chip_id:
            return
        confirmed_perf = time.perf_counter()
        confirmed_ts = datetime.datetime.now()
        first_perf = profile["first_cell_perf"] or confirmed_perf
        first_ts = profile["first_cell_ts"] or confirmed_ts
        self._log(
            f"TURN_TIMING accepted player='{profile['player']}' phase={profile['phase']} "
            f"cell={cell} target={profile['target_cell']} target_first={first_ts.strftime('%H:%M:%S.%f')[:-3]} "
            f"accepted={confirmed_ts.strftime('%H:%M:%S.%f')[:-3]} delta={(confirmed_perf - first_perf):.3f}s "
            f"state={accepted_state}->{self.engine.turn_state}"
        )

        if self.engine.turn_state == "AWAITING_RULE":
            self._start_turn_profile(chip_id, profile["player"], self.engine.rule_target, "AWAITING_RULE")
        else:
            self._turn_profile = None

    def _advance_turn(self) -> bool:
        """Передать ход следующему игроку (или выдать экстра-ход). True если ход начат.
        Вызывается из [N] и из симуляции."""
        if self.engine._pending_extra_turn:
            chip_id = self.engine._pending_extra_turn
            self.engine._pending_extra_turn = None
            self._log("EXTRA_TURN: игрок получает ещё один ход")
            event = self.engine.begin_turn(chip_id)
            self._forward_engine_event(event)
            player = self.engine.state.get_player(chip_id)
            if player:
                self._start_turn_profile(chip_id, player.name, self.engine.expected_cell, self.engine.turn_state)
            self._chip_cells.pop(chip_id, None)
            self._chip_cell_debounce.pop(chip_id, None)
            return True
        entry = self.turns.advance()
        if not entry:
            print("⚠️ Очередь пуста. Добавьте игроков через [T].")
            return False
        self._log(f"ХОД: '{entry['player']}' → фишка '{entry['chip_name']}'")
        if self.engine.is_active:
            event = self.engine.begin_turn(entry["chip_id"])
            self._forward_engine_event(event)
            self._start_turn_profile(
                entry["chip_id"], entry["player"], self.engine.expected_cell, self.engine.turn_state
            )
            self._chip_cells.pop(entry["chip_id"], None)
            self._chip_cell_debounce.pop(entry["chip_id"], None)
        return True

    def _simulate_step(self) -> bool:
        """Один шаг симуляции: продвинуть FSM без физической детекции.
        IDLE/TURN_DONE → следующий ход; WAITING_MOVE/AWAITING_RULE → «фишка дошла до целевой клетки»."""
        if not settings.SIMULATION_ENABLED:
            return False
        if not self.engine.is_active:
            return False
        st = self.engine.turn_state
        if st in ("IDLE", "TURN_DONE"):
            if not self.engine.can_advance():
                return False
            started = self._advance_turn()
            self._sync_http_game_state()
            return started
        chip = self.engine._active_chip_id
        if chip is None:
            return False
        if st == "WAITING_MOVE":
            target = self.engine.expected_cell
        elif st == "AWAITING_RULE":
            target = self.engine.rule_target
        else:
            return False
        accepted_state = self.engine.turn_state
        events = self.engine.on_chip_at_cell(chip, target)
        for event in events:
            self._forward_engine_event(event)
        if events:
            self._chip_cells[chip] = target
            self._finish_turn_profile(chip, target, accepted_state)
        self._sync_http_game_state()
        return bool(events)

    def _simulate_auto(self) -> None:
        """Автопрогон партии до победителя без детекции (с предохранителем от зацикливания)."""
        if not settings.SIMULATION_ENABLED:
            return
        self._log("СИМУЛЯЦИЯ: автопрогон партии")
        for _ in range(1000):
            if not self.engine.is_active:
                break
            if not self._simulate_step():
                break
        self._sync_http_game_state()

    def _log_pre_game_summary(self) -> None:
        """Показать сводку перед стартом, не двигая очередь ходов."""
        entries = self.turns.list_entries()
        if not entries:
            self._log("ИГРА НЕ ЗАПУЩЕНА: очередь пуста. Добавьте игроков через [T], затем [S].")
            return

        if self.field.is_calibrated():
            board_size = self.field.grid_cols * self.field.grid_rows
            field_info = f"{self.field.grid_cols}×{self.field.grid_rows} ({board_size} клеток)"
        else:
            field_info = f"не откалибровано (будет BOARD_SIZE={game_rules.BOARD_SIZE})"

        players = ", ".join(f"{e['player']}[{e['chip_name']}]" for e in entries)
        rules = []
        for cell, rule in sorted(game_rules.CELL_RULES.items()):
            effect = rule.get("effect", "?")
            distance = rule.get("distance")
            suffix = f" {distance}" if distance else ""
            rules.append(f"{cell}:{effect}{suffix}")
        rules_info = ", ".join(rules) if rules else "нет"

        self._log(
            f"ИГРА НЕ ЗАПУЩЕНА: нажмите [S] для старта. "
            f"Игроки: {players}. Поле: {field_info}. Путь: {game_rules.PATH_TYPE}. Правила: {rules_info}"
        )

    @staticmethod
    def _putText_stroked(frame, text, pos, font, font_scale, color, thickness=1):
        """Draw text with dark stroke for visibility on variable background."""
        cv2.putText(frame, text, pos, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)


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

        # Сбор угловых точек в режиме калибровки
        if self._calibration_mode and event == cv2.EVENT_LBUTTONDOWN:
            self._calibration_corners.append((x, y))
            print(f"  Угол {len(self._calibration_corners)}/4: ({x}, {y})")
            if len(self._calibration_corners) == 4:
                self._calibration_mode = False
                self._text_input_mode = "field_grid"
                self._text_input_buffer = ""
                print("  Введите размер сетки: ширина x высота (например, 8x5):")
            return

        # Клики по кнопкам управления (вне режима регистрации и калибровки)
        if not self.registration_mode and not self._calibration_mode and event == cv2.EVENT_LBUTTONDOWN:
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
                h, w = roi.shape[:2]
                self._log(f"  снимок {i + 1} захвачен (blob {w}×{h})")

        return shots

    def _capture_tilt_shot(self, x1: int, y1: int, x2: int, y2: int,
                           shot_num: int, total: int) -> np.ndarray | None:
        """Живой просмотр → любая клавиша фиксирует кадр. ROI вырезается по coords."""
        while True:
            ret, preview = self.camera.read_frame()
            if ret:
                hint = f"Shot {shot_num}/{total}: tilt ~10 deg | any key = capture | ESC = stop"
                self._putText_stroked(preview, hint, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
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
                    self._putText_stroked(preview, hint, (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
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
                self._putText_stroked(preview, hint, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
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

    def _capture_bg_shots(self, name: str, n: int) -> list[np.ndarray]:
        """
        Unified background registration for 'background' and 'background_mog2' modes.
        Phase 1: user clears field → press any key.
        Phase 2: learn background (single absdiff frame or MOG2 multi-frame training with counter).
        Phase 3: capture N object shots with auto-detection.
        """
        mode = settings.REGISTRATION_MODE

        # Phase 1: wait for user to clear the field
        self._log(f"[{mode}] Уберите все объекты с поля, нажмите любую клавишу...")
        while True:
            ret, preview = self.camera.read_frame()
            if ret:
                self._putText_stroked(preview, "Remove all objects, press any key",
                                      (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
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

        # Phase 2: learn background
        k_close = np.ones((15, 15), np.uint8)
        k_open = np.ones((5, 5), np.uint8)
        if mode == 'background':
            _, bg_frame = self.camera.read_frame()
            bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
            mog2 = None
        else:  # background_mog2
            total_bg = settings.MOG2_BG_FRAMES
            mog2 = MOG2BackgroundLearner()
            bg_frames = []
            while len(bg_frames) < total_bg:
                ret, frame = self.camera.read_frame()
                if not ret:
                    continue
                bg_frames.append(frame)
                count = len(bg_frames)
                preview = frame.copy()
                hint = f"Wait... capturing background {count}/{total_bg}"
                self._putText_stroked(preview, hint, (20, 50),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                if self._display_scale != 1.0:
                    dw = int(preview.shape[1] * self._display_scale)
                    dh = int(preview.shape[0] * self._display_scale)
                    preview = cv2.resize(preview, (dw, dh))
                cv2.imshow(settings.WINDOW_NAME, preview)
                cv2.waitKey(1)
            mog2.train_on_frames(bg_frames)
            self._log(f"MOG2: обучение завершено ({total_bg} кадров)")
            bg_gray = None

        # Phase 3: capture object shots
        # while-loop (not for) so oversized blobs retry the same shot number
        fh_max = None  # populated on first frame read
        fw_max = None
        shots = []
        i = 0
        while i < n:
            self._log(f"  снимок {i + 1}/{n}: поместите '{name}' на новую позицию, нажмите любую клавишу (ESC=стоп)")
            while True:
                ret, preview = self.camera.read_frame()
                if ret:
                    hint = f"Shot {i + 1}/{n}: place '{name}' at new position | any key | ESC=stop"
                    self._putText_stroked(preview, hint, (20, 50),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
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

            if fh_max is None:
                fh_max = int(frame.shape[0] * settings.MOG2_MAX_BLOB_RATIO)
                fw_max = int(frame.shape[1] * settings.MOG2_MAX_BLOB_RATIO)

            if mode == 'background':
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
                if bw > fw_max or bh > fh_max:
                    self._log(f"  снимок {i + 1}: blob слишком большой ({bw}×{bh}) — возможно тень или сдвиг фона, повторите")
                    continue
                pad = 15
                x1 = max(0, bx - pad); y1 = max(0, by - pad)
                x2 = min(frame.shape[1], bx + bw + pad); y2 = min(frame.shape[0], by + bh + pad)
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    shots.append(roi.copy())
                    self._log(f"  снимок {i + 1} захвачен (blob {bw}×{bh})")
                    i += 1
            else:  # background_mog2
                mask = mog2.detect_foreground(frame)
                if mask is None:
                    self._log(f"  снимок {i + 1}: ошибка маски MOG2")
                    continue
                result = mog2.extract_roi_from_mask(frame, mask)
                if result is None:
                    self._log(f"  снимок {i + 1}: объект не обнаружен (пустая маска MOG2)")
                    continue
                roi, (x1, y1, x2, y2) = result
                w, h = x2 - x1, y2 - y1
                if w > fw_max or h > fh_max:
                    self._log(f"  снимок {i + 1}: blob слишком большой ({w}×{h}) — весь кадр засчитан foreground, повторите")
                    continue
                if roi.size > 0:
                    shots.append(roi.copy())
                    self._log(f"  снимок {i + 1} захвачен MOG2 (blob {w}×{h})")
                    i += 1

        return shots

    def _register_object_from_roi(self, roi_image: np.ndarray, name: str,
                                   roi_coords: tuple = None, pre_shots: list = None):
        """Регистрация объекта из области интереса (один или несколько снимков)."""
        if pre_shots is not None:
            shots = pre_shots
        else:
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
            {"label": "[D] Delete",   "key": ord("d")},
            {"label": "[I] Images",   "key": ord("i")},
            {"label": "[S] Start",    "key": ord("s")},
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

        # Панель команд и кнопки управления — только если включён оверлей cv2
        if not settings.CV2_SHOW_OVERLAY:
            self._button_rects = []
            return

        # Полупрозрачный блок с инструкциями (правый верхний угол)
        lines = ["Controls:", "[R] Register", "[L] List", "[D] Delete", "[I] Images",
                 "[T] Add player", "[X] Remove player", "[S] Start game",
                 "[N] Next turn", "[A] All chips",
                 "[C] Cal/Clear field", "[G] Toggle grid", "[Q] Quit"]
        box_w, box_h = 185, len(lines) * 20 + 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - box_w - 5, 5), (w - 5, box_h + 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        for i, txt in enumerate(lines):
            color = (200, 200, 200) if i > 0 else (120, 255, 120)
            self._putText_stroked(frame, txt, (w - box_w, 22 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Текстовый ввод в оверлее
        if self._text_input_mode is not None:
            prompt = ("Name (Enter/ESC/click):"            if self._text_input_mode == "register"
                      else "Num or name (Enter/ESC/click):" if self._text_input_mode == "delete_select"
                      else "Player name (Enter/ESC):"       if self._text_input_mode == "turn_player"
                      else "Chip num/name (Enter/ESC):"     if self._text_input_mode == "turn_chip"
                      else "Remove player num/name (Enter/ESC):" if self._text_input_mode == "remove_player"
                      else "Grid cols x rows, e.g. 8x5 (Enter/ESC):")
            display_text = f"{prompt} {self._text_input_buffer}_"
            (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            bx1, by1 = w // 2 - tw // 2 - 12, h // 2 - th - 12
            bx2, by2 = w // 2 + tw // 2 + 12, h // 2 + 12
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 200, 200), 1)
            self._putText_stroked(frame, display_text, (bx1 + 12, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        # Кнопки внизу кадра
        btn_colors = [(40, 100, 40), (40, 40, 100), (80, 40, 100), (40, 80, 100), (100, 40, 40)]
        self._button_rects = self._compute_button_rects(w, h)
        for btn, color in zip(self._button_rects, btn_colors):
            x1, y1, x2, y2 = btn["rect"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
            (tw, th), _ = cv2.getTextSize(btn["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            self._putText_stroked(frame, btn["label"],
                        (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    def start_registration(self, name: str):
        if settings.REGISTRATION_MODE in ('background', 'background_mog2'):
            self._pending_background_name = name
            self._log(f"РЕЖИМ РЕГИСТРАЦИИ ({settings.REGISTRATION_MODE}): '{name}'")
        else:
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
                self._putText_stroked(canvas, "no img", (x + 30, y + thumb // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

            self._putText_stroked(canvas, meta["name"][:20], (x, y + thumb + label_h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

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

        # Режим калибровки — только ESC для отмены
        if self._calibration_mode:
            if key == 27:
                self._calibration_mode = False
                self._calibration_corners = []
                print("❌ Калибровка отменена.")
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
        elif key == ord('t'):
            self._reset_if_finished()
            self._text_input_mode = "turn_player"
            self._text_input_buffer = ""
        elif key == ord('s'):
            entries = self.turns.list_entries()
            if not entries:
                msg = "Очередь пуста. Добавьте игроков через [T]."
                print(f"⚠️ {msg}")
                self.http_server.emit_event("NOTICE", {"message": msg, "level": "error"})
            elif not settings.ALLOW_SINGLE_PLAYER and len(entries) < 2:
                msg = f"Нужно 2+ игроков для старта (сейчас {len(entries)}). Добавьте ещё через [T]."
                print(f"⚠️ {msg}")
                self.http_server.emit_event("NOTICE", {"message": msg, "level": "error"})
            else:
                players = [Player(e["player"], e["chip_id"], e["chip_name"]) for e in entries]
                board_size = (self.field.grid_cols * self.field.grid_rows
                              if self.field.is_calibrated() else None)
                event = self.engine.start_game(players, board_size=board_size)
                self._forward_engine_event(event)
                self._chip_cells.clear()
                self._chip_cell_debounce.clear()
                self.turns.unfocus()
                self._turn_profile = None
                self._log("ИГРА ГОТОВА: нажмите [N] или Next Turn для первого хода")
                self._sync_http_game_state()
                # turns НЕ стартует здесь — первый [N] правильно начнёт с игрока 0
        elif key == ord('x'):
            entries = self.turns.list_entries()
            if not entries:
                print("⚠️ Нет игроков в очереди.")
            else:
                print("\n🚫 Удаление игрока из очереди:")
                for i, e in enumerate(entries, 1):
                    print(f"  {i}. {e['player']} ({e['chip_name']})")
                self._delete_list = entries
                self._text_input_mode = "remove_player"
                self._text_input_buffer = ""
        elif key == ord('n'):
            if not self.engine.is_active:
                self.turns.unfocus()
                self._log_pre_game_summary()
                self._sync_http_game_state()
                return False

            if self.engine.is_active and not self.engine.can_advance():
                print("⚠️ Дождитесь выполнения хода перед передачей очереди.")
                return False

            self._advance_turn()
        elif key == ord('a'):
            self.turns.unfocus()
            self._log("Режим: все фишки")
        elif key == ord('c'):
            if self.field.is_calibrated():
                self.field = FieldCalibration()
                self._grid_visible = True
                self._log("✓ Калибровка удалена. Готово к новой калибровке.")
            else:
                self._calibration_mode = True
                self._calibration_corners = []
                self._log("CALIBRATION: click 4 corners in order (Top-Left → Top-Right → Bottom-Right → Bottom-Left)")
        elif key == ord('g'):
            self._grid_visible = not self._grid_visible
            state = "видна" if self._grid_visible else "скрыта"
            self._log(f"Сетка: {state}")
        elif key == ord('u'):
            if self.engine.is_active:
                filepath = self.state_manager.save(self.engine, self.turns.list_entries(), self.field)
                self._log(f"✓ Состояние сохранено: {Path(filepath).name}")
            else:
                print("⚠️ Игра не активна. Начните игру через [S].")
        elif key == ord('v'):
            saves = self.state_manager.list_saves()
            if not saves:
                print("⚠️ Сохранённых игр нет.")
            else:
                print("\n💾 Доступные сохранения:")
                for i, save in enumerate(saves, 1):
                    players = ", ".join(save["players"])
                    print(f"  {i}. {save['filename']} — {save['timestamp']}")
                    print(f"     Игроки: {players} (доска: {save['board_size']})")
                # Загрузить первое сохранение для быстрого доступа
                self._pending_load_save = saves[0]["path"]
                print(f"\n💡 Для загрузки наберите в консоли: game.load_state('{saves[0]['path']}')")
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
            if deleted:
                self._sync_chips_http()
            elif value:
                print(f"❌ Фишка '{value}' не найдена.")

        elif mode == "turn_player":
            if not value:
                print("❌ Имя игрока не может быть пустым.")
                return
            objects = self.registry.list_objects()
            if not objects:
                print("❌ Нет зарегистрированных фишек.")
                return
            self._pending_player_name = value
            self._delete_list = objects
            print("\n🎮 Выберите фишку для игрока:")
            for i, obj in enumerate(objects, 1):
                print(f"  {i}. {obj['name']}")
            self._text_input_mode = "turn_chip"
            self._text_input_buffer = ""

        elif mode == "turn_chip":
            player_name = self._pending_player_name or ""
            self._pending_player_name = None
            if not value:
                return
            chip = None
            try:
                idx = int(value) - 1
                if 0 <= idx < len(self._delete_list):
                    chip = self._delete_list[idx]
            except ValueError:
                for obj in self._delete_list:
                    if obj["name"] == value:
                        chip = obj
                        break
            if chip:
                self.turns.add(player_name, chip["id"], chip["name"])
                self._log(f"ОЧЕРЕДЬ: игрок '{player_name}' → фишка '{chip['name']}'")
                self._sync_chips_http()
                self._sync_http_game_state()
            else:
                print(f"❌ Фишка '{value}' не найдена.")

        elif mode == "remove_player":
            if not value:
                return
            chip_id = self.turns.remove(value)
            if chip_id:
                self._log(f"ОЧЕРЕДЬ: игрок удалён ('{value}')")
                self._sync_chips_http()
                # Синхронизировать с GameEngine — удалить из состояния игры
                if self.engine.is_active:
                    player = self.engine.state.get_player(chip_id)
                    if player:
                        self.engine.state.players.remove(player)
                        self._log(f"ENGINE: игрок '{player.name}' удалён из состояния")

                    # Если удаляемый игрок — текущий активный, сбросить состояние
                    if self.engine._active_chip_id == chip_id:
                        self.engine._active_chip_id = None
                        self.engine._pending_extra_turn = None
                        # Позволить адванцировать очередь, установив TURN_DONE
                        if self.engine.turn_state in ("WAITING_MOVE", "AWAITING_RULE"):
                            self.engine.turn_state = "TURN_DONE"
                        self._log(f"⚠️ Удалённый игрок был активным. Очередь готова к продвижению.")
                    self._turn_profile = None
                self._sync_http_game_state()
            else:
                print(f"❌ Игрок '{value}' не найден в очереди.")

        elif mode == "field_grid":
            import re
            parts = re.split(r'[xхX×,\s]+', value.strip())
            if len(parts) == 2:
                try:
                    cols, rows = int(parts[0]), int(parts[1])
                    if cols > 0 and rows > 0 and len(self._calibration_corners) == 4:
                        self.field.set_corners(self._calibration_corners, cols, rows)
                        self.field.save("items/field_calibration.json")
                        self._log(f"ПОЛЕ: {cols}×{rows} клеток, сохранено")
                    else:
                        print("❌ Некорректный размер или нет угловых точек.")
                except ValueError:
                    print(f"❌ Не могу разобрать '{value}', ожидается 'ШxВ' (например, 8x5).")
            else:
                print(f"❌ Ожидается 'ШxВ' (например, 8x5), получено: '{value}'.")
            self._calibration_corners = []

    def _get_placeholder(self):
        """Кадр-заглушка для режима без камеры (создаётся один раз)."""
        if self._placeholder is None:
            f = np.zeros((settings.CAMERA_HEIGHT, settings.CAMERA_WIDTH, 3), dtype=np.uint8)
            f[:] = (35, 35, 40)
            cv2.putText(f, "No camera - simulation mode",
                        (40, settings.CAMERA_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (200, 200, 200), 2, cv2.LINE_AA)
            self._placeholder = f
        return self._placeholder.copy()

    def tracking_loop(self):
        """Главный цикл трекинга и обработки кадров"""
        # Инициализация камеры
        self._log(f"СТАРТ системы  детектор={settings.DETECTOR_TYPE}  память={settings.FEATURE_MEMORY_FILE}")
        if not self.camera.initialize():
            self._no_camera = True
            self._log("Камера недоступна — режим без камеры (веб + симуляция)")
        if settings.HEADLESS or self._no_camera:
            settings.SIMULATION_ENABLED = True  # без камеры ход двигается только симуляцией
        
        # Загрузка калибровки поля (если есть)
        field_info = ""
        if self.field.load("items/field_calibration.json"):
            field_info = f" | field: {self.field.grid_cols}x{self.field.grid_rows}"

        self._log(f"SESSION START - detector: {settings.DETECTOR_TYPE}, SIFT: {settings.SIFT_FEATURES}{field_info}")

        # Создание окна и привязка обработчика мыши (в headless GUI нет)
        if not settings.HEADLESS:
            cv2.namedWindow(settings.WINDOW_NAME)
            cv2.setMouseCallback(settings.WINDOW_NAME, self._mouse_callback, self.mouse_handler_params)
        
        self.running = True
        frame_count = 0
        start_time = datetime.datetime.now()

        while self.running:
            proc_start = self.metrics.tick()

            # Чтение кадра
            p = self.profiler
            with (p.measure('camera') if p else nullcontext()):
                if self._no_camera:
                    ret, frame = True, self._get_placeholder()
                else:
                    ret, frame = self.camera.read_frame()
            if not ret:
                # камера отвалилась на ходу — продолжаем по заглушке, не выходим
                self._no_camera = True
                ret, frame = True, self._get_placeholder()

            frame_count += 1
            self.mouse_handler_params[0] = frame.copy()

            # Отложенная регистрация — запускается здесь, а не в callback
            if self._pending_registration is not None:
                roi_image, name, roi_coords = self._pending_registration
                self._pending_registration = None
                h, w = roi_image.shape[:2]
                self._log(f"  снимок 1/{settings.REGISTRATION_SHOTS} захвачен (blob {w}×{h})")
                self._register_object_from_roi(roi_image, name, roi_coords)
                self.temporal_filter.reset()
                self._sync_chips_http()

            if self._pending_background_name is not None:
                name = self._pending_background_name
                self._pending_background_name = None
                shots = self._capture_bg_shots(name, settings.REGISTRATION_SHOTS)
                if shots:
                    self._register_object_from_roi(shots[0], name, pre_shots=shots)
                    self.temporal_filter.reset()
                    self._sync_chips_http()
                else:
                    self._log(f"ОТМЕНА регистрации '{name}': нет снимков")

            # Извлечение признаков из текущего кадра
            with (p.measure('extract') if p else nullcontext()):
                kp_frame, des_frame = self.feature_extractor.extract_features(frame)
                if settings.TEMPORAL_FILTER_ENABLED:
                    kp_frame, des_frame = self.temporal_filter.filter(kp_frame, des_frame)

            # Визуализация
            display_frame = frame.copy()
            with (p.measure('field') if p else nullcontext()):
                if self._grid_visible:
                    self.field.draw_grid(display_frame)
                    self.field.draw_cell_fills(display_frame, self._build_cell_colors(),
                                               settings.CELL_FILL_ALPHA)
                    self.field.draw_cell_numbers(display_frame, settings.CELL_NUMBER_COLOR,
                                                 settings.CELL_NUMBER_POSITION)

            if des_frame is not None and len(des_frame) > 0:
                # Получение всех зарегистрированных объектов
                registered_objects = self.registry.get_all_for_detection()

                if registered_objects:
                    # Фокусированная детекция: только фишка текущего игрока
                    active_id = self.turns.active_chip_id
                    if active_id and active_id in registered_objects:
                        registered_objects = {active_id: registered_objects[active_id]}

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

                    # Сборка карты клеток для логирования
                    cell_map = {}  # object_id → (cell_number, row+1, col+1)
                    for det in detections:
                        cell_pos = self.field.pixel_to_cell(det["center"][0], det["center"][1])
                        cell_num = self.field.pixel_to_cell_number(det["center"][0], det["center"][1])
                        if cell_num and cell_pos:
                            row, col = cell_pos
                            cell_map[det["object_id"]] = (cell_num, row+1, col+1)

                    tracker_ids = {o.object_id for o in self.tracker.get_active_objects()}
                    for oid in tracker_ids - self._prev_active_ids:
                        name = self._active_names.get(oid, oid[:8])
                        if oid in cell_map:
                            cell_num, row, col = cell_map[oid]
                            cell_info = f" at cell {cell_num} ({row},{col})"
                        else:
                            cell_info = " (outside field)"
                        self._log(f"APPEARED: '{name}'{cell_info}")
                    for oid in self._prev_active_ids - tracker_ids:
                        self._log(f"DISAPPEARED: '{self._active_names.get(oid, oid[:8])}'")
                    self._prev_active_ids = tracker_ids

                    active_for_metrics = []
                    for obj in self.tracker.get_active_objects():
                        pos = obj.get_current_position()
                        cell_num = self.field.pixel_to_cell_number(pos[0], pos[1]) if pos else None
                        active_for_metrics.append({
                            "name": self._active_names.get(obj.object_id, obj.object_id[:8]),
                            "cell": cell_num,
                        })
                    self.metrics.update_active_objects(active_for_metrics)
                    
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

                        if settings.CV2_SHOW_BBOX:
                            cv2.polylines(display_frame, [smooth_corners], True, (0, 255, 0), 2, cv2.LINE_AA)
                            cv2.circle(display_frame, smooth_center, 5, (0, 0, 255), -1)

                        cell_num = self.field.pixel_to_cell_number(center[0], center[1])
                        cell_pos = self.field.pixel_to_cell(center[0], center[1])
                        if cell_num and cell_pos:
                            row, col = cell_pos
                            cell_str = f" [cell {cell_num} ({row+1},{col+1})]"
                            self._mark_first_cell_seen(object_id, cell_num)
                            if self.engine.is_active and cell_num != self._chip_cells.get(object_id):
                                # Дебаунс: требуем 3 consecutive frames на одной клетке перед фиксацией
                                cached_cell, cached_count = self._chip_cell_debounce.get(object_id, (None, 0))
                                if cached_cell == cell_num:
                                    cached_count += 1
                                else:
                                    cached_count = 1
                                self._chip_cell_debounce[object_id] = (cell_num, cached_count)

                                if cached_count >= 3:
                                    # Зафиксировать движение
                                    self._chip_cells[object_id] = cell_num
                                    accepted_state = self.engine.turn_state
                                    events = self.engine.on_chip_at_cell(object_id, cell_num)
                                    for event in events:
                                        self._forward_engine_event(event)
                                    if events:
                                        self._finish_turn_profile(object_id, cell_num, accepted_state)
                                    # Сброс дебаунса после фиксации
                                    del self._chip_cell_debounce[object_id]
                        else:
                            cell_str = ""
                        if settings.CV2_SHOW_BBOX:
                            label = f"{object_name}{cell_str} ({confidence:.2f}, {matches_count})"
                            self._putText_stroked(
                                display_frame, label,
                                (smooth_corners[0][0][0], smooth_corners[0][0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                            )
                        
                        # Вывод в консоль (опционально, можно отключить для производительности)
                        # print(f"Объект '{object_name}' (ID: {object_id[:8]}...) найден! Центр: {center}, Уверенность: {confidence:.2f}")
            
            # Стартовое сообщение (первые 5 секунд, только когда нет активного хода).
            # Игровые тексты дублируются в вебе через API, поэтому в кадр пишем
            # их только при включённом оверлее cv2 (settings.CV2_SHOW_OVERLAY).
            if (settings.CV2_SHOW_OVERLAY
                    and (datetime.datetime.now() - start_time).total_seconds() < 5.0
                    and not self.turns.is_focused()
                    and not self._calibration_mode
                    and not self.registration_mode):
                self._putText_stroked(display_frame, "Mode: all chips",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 180, 180), 2)

            # Информация о режиме калибровки
            if self._calibration_mode:
                n = len(self._calibration_corners)
                labels_short = ["TL", "TR", "BR", "BL"]
                labels_full = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
                self._putText_stroked(display_frame,
                            f"CALIBRATION: click corner {n+1}/4 ({labels_full[n]})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
                for i, pt in enumerate(self._calibration_corners):
                    cv2.circle(display_frame, pt, 8, (255, 200, 0), -1)
                    self._putText_stroked(display_frame, labels_short[i], (pt[0] + 10, pt[1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

            # Победитель / готовность / текущий ход — игровые тексты в кадре только при
            # включённом оверлее cv2; в веб-интерфейс та же информация идёт через API.
            if settings.CV2_SHOW_OVERLAY:
                if self.engine.state.winner:
                    self._putText_stroked(
                        display_frame,
                        f"GAME OVER — {self.engine.state.winner.name} wins!",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 100), 2
                    )
                elif self.engine.is_active and not self.turns.current():
                    self._putText_stroked(
                        display_frame,
                        "GAME READY — press [N] or Next Turn",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2
                    )
                # Информация о текущем ходе + статус FSM
                elif turn_entry := self.turns.current():
                    self._putText_stroked(
                        display_frame,
                        f"TURN: {turn_entry['player']}  [{turn_entry['chip_name']}]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2
                    )
                    fsm = self.engine.status_msg()
                    if fsm:
                        self._putText_stroked(
                            display_frame, fsm[0],
                            (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, fsm[1], 2
                        )

            # Информация о режиме регистрации
            if self.registration_mode:
                self._putText_stroked(
                    display_frame, f"REGISTRATION MODE: {self.registration_name}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                )
                self._putText_stroked(
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
            with (p.measure('show') if p else nullcontext()):
                if self._display_scale != 1.0:
                    dw = int(display_frame.shape[1] * self._display_scale)
                    dh = int(display_frame.shape[0] * self._display_scale)
                    display_frame = cv2.resize(display_frame, (dw, dh))
                if not settings.HEADLESS:
                    cv2.imshow(settings.WINDOW_NAME, display_frame)
                    self.metrics.render()

            # Кадр → MJPEG стрим (encode после resize чтобы слать уменьшенный кадр)
            with (p.measure('mjpeg') if p else nullcontext()):
                _, jpeg = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                self.http_server.update_stream_frame(jpeg.tobytes())

            # HTTP команды → эмуляция клавиши (приоритет ниже реальной клавиши)
            for cmd in self.http_server.drain_commands():
                if self._pending_key is not None:
                    continue
                if cmd == "next":
                    self._pending_key = ord('n')
                elif cmd == "start":
                    self._pending_key = ord('s')
                elif cmd == "grid":
                    self._pending_key = ord('g')
                elif cmd == "quit":
                    self._pending_key = ord('q')
                elif cmd == "endgame":
                    self._end_game()
                elif cmd == "overlay":
                    settings.CV2_SHOW_OVERLAY = not settings.CV2_SHOW_OVERLAY
                    state = "видна" if settings.CV2_SHOW_OVERLAY else "скрыта"
                    self._log(f"Панель команд cv2: {state}")
                elif cmd == "sim_step":
                    self._simulate_step()
                elif cmd == "sim_auto":
                    self._simulate_auto()
                elif cmd == "sim_toggle":
                    settings.SIMULATION_ENABLED = not settings.SIMULATION_ENABLED
                    self._log(f"Симуляция (форс ходов): {'вкл' if settings.SIMULATION_ENABLED else 'выкл'}")
                    self._sync_http_game_state()

            # HTTP админ-команды (удаление фишки / игрока, калибровка)
            for acmd in self.http_server.drain_admin():
                if acmd["action"] == "delete_chip":
                    self._delete_chip_by_value(acmd["value"])
                elif acmd["action"] == "remove_player":
                    self._remove_player_by_value(acmd["value"])
                elif acmd["action"] == "calibrate":
                    self._calibrate_from_web(acmd["corners"], acmd["cols"], acmd["rows"], frame)
                elif acmd["action"] == "set_path":
                    self._set_path_from_web(acmd["value"])

            # HTTP веб-регистрация
            for cmd in self.http_server.drain_register_commands():
                if cmd["action"] == "start":
                    self._web_reg_name = cmd["name"]
                    self._web_reg_shots = []
                    self._log(f"WEB REGISTER: '{cmd['name']}' ({settings.REGISTRATION_SHOTS} снимков)")
                    self.http_server.update_reg_status({
                        "active": True, "shot_num": 0,
                        "total": settings.REGISTRATION_SHOTS, "name": cmd["name"], "done": False,
                    })
                elif cmd["action"] == "shot" and self._web_reg_name:
                    fh, fw = frame.shape[:2]
                    x1 = max(0, int(cmd["x1"] * fw))
                    y1 = max(0, int(cmd["y1"] * fh))
                    x2 = min(fw, int(cmd["x2"] * fw))
                    y2 = min(fh, int(cmd["y2"] * fh))
                    if x2 > x1 + 4 and y2 > y1 + 4:
                        roi = frame[y1:y2, x1:x2].copy()
                        if roi.size > 0:
                            self._web_reg_shots.append(roi)
                            sn = len(self._web_reg_shots)
                            total = settings.REGISTRATION_SHOTS
                            self._log(f"WEB REGISTER: снимок {sn}/{total}")
                            if sn >= total:
                                name = self._web_reg_name
                                shots = self._web_reg_shots
                                self._web_reg_name = None
                                self._web_reg_shots = []
                                self._register_object_from_roi(shots[0], name, pre_shots=shots)
                                self.temporal_filter.reset()
                                self._sync_chips_http()
                                self.http_server.update_reg_status({
                                    "active": False, "shot_num": sn, "total": total, "name": name, "done": True,
                                })
                            else:
                                self.http_server.update_reg_status({
                                    "active": True, "shot_num": sn,
                                    "total": total, "name": self._web_reg_name, "done": False,
                                })
                elif cmd["action"] == "cancel":
                    self._web_reg_name = None
                    self._web_reg_shots = []
                    self.http_server.update_reg_status({
                        "active": False, "shot_num": 0,
                        "total": settings.REGISTRATION_SHOTS, "name": "", "done": False,
                    })

            # HTTP join → добавить игрока в очередь
            for join in self.http_server.drain_joins():
                jname, jchip_id = join["name"], join["chip_id"]
                chip = next((c for c in self.registry.list_objects() if c["id"] == jchip_id), None)
                if chip:
                    self._reset_if_finished()
                    self.turns.add(jname, chip["id"], chip["name"])
                    self._log(f"ОЧЕРЕДЬ [HTTP]: игрок '{jname}' → фишка '{chip['name']}'")
                    self._sync_chips_http()
                    self._sync_http_game_state()

            # Обработка клавиатуры и кликов по кнопкам (в headless окна нет → клавиатуры нет)
            if settings.HEADLESS:
                time.sleep(0.01)
                key = 255
            else:
                key = cv2.waitKey(1) & 0xFF
            if self._pending_key is not None:
                key = self._pending_key
                self._pending_key = None

            if p:
                p.set_counts(total_registered, active_count)
                p.next_frame()
            if self._handle_key(key):
                break
        
        # Очистка (финальный лог в shutdown())
        self.camera.release()
        if not settings.HEADLESS:
            cv2.destroyAllWindows()

    def shutdown(self):
        """Корректное завершение работы системы"""
        # Сводка качества детекции за сессию (раздел 4.2)
        for line in self.metrics.session_summary():
            self._log(line)
        # Сводка разбивки времени по операциям (раздел 4.3)
        if self.profiler:
            self.profiler._print_summary()
        self._log("SESSION END")
        self.running = False
        self.http_server.stop()
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


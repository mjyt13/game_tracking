import json
import queue
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Optional

from server._state import _State
from server._pages import ADMIN_HTML, JOIN_HTML, GAME_HTML, RESULT_HTML
from server import _net
from config import settings
from config import messages as _msg_module


class GameEventHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/":
            self._send(ADMIN_HTML, "text/html; charset=utf-8")
        elif self.path == "/join":
            self._send(JOIN_HTML, "text/html; charset=utf-8")
        elif self.path == "/game":
            self._send(GAME_HTML, "text/html; charset=utf-8")
        elif self.path == "/result":
            self._send(RESULT_HTML, "text/html; charset=utf-8")
        elif self.path == "/stream":
            self._mjpeg_stream()
        elif self.path == "/api/chips":
            self._json({"status": "ok", "chips": _State.available_chips})
        elif self.path == "/api/chips/all":
            self._json({"status": "ok", "chips": _State.all_chips})
        elif self.path.startswith("/api/chips/") and self.path.endswith("/image"):
            chip_id = self.path[len("/api/chips/"):-len("/image")]
            img = _State.chip_images.get(chip_id)
            if img:
                self._send(img, "image/jpeg")
            else:
                self._json({"error": "no image"}, 404)
        elif self.path == "/api/messages":
            lang = settings.LANGUAGE if settings.LANGUAGE in _msg_module.MESSAGES else 'en'
            self._json({"messages": _msg_module.MESSAGES[lang]})
        elif self.path == "/api/register/status":
            self._json({"status": "ok", "reg": _State.reg_status})
        elif self.path == "/api/game/events":
            self._json({"status": "ok",
                        "events": _State.events[-20:],
                        "game_state": _State.game_state})
        elif self.path == "/api/game/state":
            self._json({"status": "ok", "state": _State.game_state})
        elif self.path == "/api/qr":
            self._send_qr()
        elif self.path == "/api/netinfo":
            ips = _net.get_all_ips()
            self._json({"ip": ips[0] if ips else None, "ips": ips, "port": settings.HTTP_PORT})
        elif self.path == "/api/health":
            self._json({"status": "ok"})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/api/game/next":
            _State.command_queue.put("next")
            self._json({"status": "queued"})
        elif self.path == "/api/game/start":
            _State.command_queue.put("start")
            self._json({"status": "queued"})
        elif self.path == "/api/game/grid":
            _State.command_queue.put("grid")
            self._json({"status": "queued"})
        elif self.path == "/api/game/overlay":
            _State.command_queue.put("overlay")
            self._json({"status": "queued"})
        elif self.path == "/api/game/path":
            self._handle_set_path()
        elif self.path == "/api/game/endgame":
            _State.command_queue.put("endgame")
            self._json({"status": "queued"})
        elif self.path == "/api/game/stop":
            _State.command_queue.put("quit")
            self._json({"status": "queued"})
        elif self.path == "/api/game/join":
            self._handle_join()
        elif self.path == "/api/chips/delete":
            self._handle_admin_data("delete_chip", "id")
        elif self.path == "/api/players/remove":
            self._handle_admin_data("remove_player", "value")
        elif self.path == "/api/calibrate":
            self._handle_calibrate()
        elif self.path == "/api/admin/login":
            self._handle_login()
        elif self.path == "/api/register/start":
            self._handle_reg_start()
        elif self.path == "/api/register/shot":
            self._handle_reg_shot()
        elif self.path == "/api/register/cancel":
            _State.reg_queue.put({"action": "cancel"})
            self._json({"status": "queued"})
        else:
            self._json({"error": "not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    # ---- helpers ----

    def _send_qr(self):
        """QR-код со ссылкой на /join (для подключения игроков и слайдов презентации)."""
        try:
            import io, segno
        except ImportError:
            self._json({"error": "segno not installed (pip install segno)"}, 503)
            return
        ip = _net.get_primary_ip()
        if not ip:
            self._json({"error": "no LAN ip"}, 503)
            return
        url = f"http://{ip}:{settings.HTTP_PORT}/join"
        buf = io.BytesIO()
        segno.make(url, error='m').save(buf, kind="png", scale=6, border=2)
        self._send(buf.getvalue(), "image/png")

    def _handle_login(self):
        """Проверка пароля ведущего (простая авторизация для демо)."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            pw = str(body.get("password", ""))
        except Exception:
            self._json({"error": "bad request"}, 400)
            return
        self._json({"ok": pw == settings.ADMIN_PASSWORD})

    def _handle_set_path(self):
        """Сменить порядок обхода клеток поля (linear/snake/spiral)."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            ptype = str(body.get("type", "")).strip()
        except Exception:
            self._json({"error": "bad request"}, 400)
            return
        if ptype not in ("linear", "snake", "spiral"):
            self._json({"error": "type must be linear/snake/spiral"}, 400)
            return
        _State.admin_queue.put({"action": "set_path", "value": ptype})
        self._json({"status": "queued"})

    def _handle_calibrate(self):
        """Калибровка поля по 4 нормированным углам + размер сетки."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            corners = body["corners"]
            cols = int(body["cols"])
            rows = int(body["rows"])
            assert len(corners) == 4 and cols > 0 and rows > 0
        except Exception:
            self._json({"error": "bad request"}, 400)
            return
        _State.admin_queue.put({"action": "calibrate", "corners": corners, "cols": cols, "rows": rows})
        self._json({"status": "queued"})

    def _handle_admin_data(self, action: str, field: str):
        """Принять админ-команду с одним полем (delete_chip/remove_player)."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            value = str(body.get(field, "")).strip()
        except Exception:
            self._json({"error": "bad request"}, 400)
            return
        if not value:
            self._json({"error": f"{field} required"}, 400)
            return
        _State.admin_queue.put({"action": action, "value": value})
        self._json({"status": "queued"})

    def _handle_reg_start(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            name = str(body.get("name", "")).strip()
        except Exception:
            self._json({"error": "bad request"}, 400)
            return
        if not name:
            self._json({"error": "Name required"}, 400)
            return
        _State.reg_queue.put({"action": "start", "name": name})
        self._json({"status": "queued"})

    def _handle_reg_shot(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            x1 = float(body["x1"]); y1 = float(body["y1"])
            x2 = float(body["x2"]); y2 = float(body["y2"])
        except Exception:
            self._json({"error": "bad request"}, 400)
            return
        _State.reg_queue.put({"action": "shot", "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        self._json({"status": "queued"})

    def _handle_join(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            name = str(body.get("name", "")).strip()
            chip_id = str(body.get("chip_id", "")).strip()
        except Exception:
            self._json({"error": "bad request"}, 400)
            return

        if not name:
            self._json({"error": "Name is required"}, 400)
            return
        if not chip_id:
            self._json({"error": "chip_id is required"}, 400)
            return

        if _State.game_state.get("active"):
            self._json({"error": "Game already in progress"}, 409)
            return

        # Validate chip is still available
        available_ids = {c["id"] for c in _State.available_chips}
        if chip_id not in available_ids:
            self._json({"error": "Chip not available (already taken or not registered)"}, 409)
            return

        _State.join_queue.put({"name": name, "chip_id": chip_id})
        self._json({"status": "ok"})

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send(self, body: bytes, content_type: str, code: int = 200):
        # Клиент (вкладка браузера) может закрыть соединение во время ответа —
        # это нормально для polling/стрима, глушим шумную трассировку.
        try:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

    def _json(self, data: dict, code: int = 200):
        self._send(json.dumps(data).encode(), "application/json", code)

    def _mjpeg_stream(self):
        """Держит соединение открытым, отправляет JPEG кадры по мере их обновления."""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self._cors()
        self.end_headers()
        last_frame = b""
        try:
            while True:
                frame = _State.stream_frame
                if not frame:
                    continue
                if frame is last_frame:
                    continue
                last_frame = frame
                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                )
                try:
                    self.wfile.write(header + frame + b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
        except Exception:
            pass

    def log_message(self, format, *args):
        pass


class GameEventServer:
    """HTTP-сервер для событий, стрима и управления игрой."""

    def __init__(self, host: str = "0.0.0.0", port: int | None = None):
        self.host = host
        self.port = port if port is not None else settings.HTTP_PORT
        self.server: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self._zc = None
        self._zc_info = None

    def start(self) -> None:
        if self.running:
            return
        self.server = ThreadingHTTPServer((self.host, self.port), GameEventHandler)
        self.running = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        # Определить адреса и вывести их в лог
        ips = _net.get_all_ips()
        print(f"[HTTP] Сервер запущен на порту {self.port}")
        if ips:
            print(f"[HTTP] Адрес для устройств в сети: http://{ips[0]}:{self.port}/")
            if len(ips) > 1:
                print("[HTTP] Если не открывается — попробуйте другой адрес:")
                for ip in ips[1:]:
                    print(f"         http://{ip}:{self.port}/")
        else:
            print("[HTTP] ⚠️ Не удалось определить IP. Узнайте вручную: ipconfig → IPv4")

        # mDNS-имя (опционально)
        if settings.MDNS_ENABLED:
            primary = ips[0] if ips else None
            self._zc, self._zc_info = _net.start_mdns(
                settings.MDNS_HOSTNAME, self.port, primary
            )
            if self._zc:
                print(f"[HTTP] mDNS: http://{settings.MDNS_HOSTNAME}.local:{self.port}/")
            else:
                print("[HTTP] mDNS не запущен (нет пакета zeroconf — pip install zeroconf)")

    def stop(self) -> None:
        _net.stop_mdns(self._zc, self._zc_info)
        self._zc = None
        self._zc_info = None
        if self.server:
            self.running = False
            self.server.shutdown()
            self.server.server_close()
            print("[HTTP] Сервер остановлен")

    def emit_event(self, event_type: str, data: dict) -> None:
        _State.events.append({"type": event_type, "data": data})

    def update_game_state(self, state: dict) -> None:
        _State.game_state = state

    def update_stream_frame(self, jpeg_bytes: bytes) -> None:
        _State.stream_frame = jpeg_bytes

    def update_available_chips(self, chips: list) -> None:
        """Обновить список незанятых фишек для /api/chips и /join."""
        _State.available_chips = chips

    def update_all_chips(self, chips: list) -> None:
        """Обновить полный список зарегистрированных фишек для админа."""
        _State.all_chips = chips

    def update_chip_images(self, images: dict) -> None:
        """Обновить миниатюры фишек (chip_id -> JPEG bytes) для /api/chips/<id>/image."""
        _State.chip_images = images

    def drain_commands(self) -> list:
        """Вернуть накопленные команды (например 'next')."""
        cmds = []
        while not _State.command_queue.empty():
            try:
                cmds.append(_State.command_queue.get_nowait())
            except queue.Empty:
                break
        return cmds

    def drain_joins(self) -> list:
        """Вернуть накопленные запросы на присоединение [{name, chip_id}]."""
        joins = []
        while not _State.join_queue.empty():
            try:
                joins.append(_State.join_queue.get_nowait())
            except queue.Empty:
                break
        return joins

    def drain_register_commands(self) -> list:
        """Вернуть накопленные команды веб-регистрации."""
        cmds = []
        while not _State.reg_queue.empty():
            try:
                cmds.append(_State.reg_queue.get_nowait())
            except queue.Empty:
                break
        return cmds

    def drain_admin(self) -> list:
        """Вернуть накопленные админ-команды (delete_chip, remove_player)."""
        cmds = []
        while not _State.admin_queue.empty():
            try:
                cmds.append(_State.admin_queue.get_nowait())
            except queue.Empty:
                break
        return cmds

    def update_reg_status(self, status: dict) -> None:
        """Обновить статус веб-регистрации для /api/register/status."""
        _State.reg_status = status

"""
Простой HTTP сервер для выдачи событий игры.
Запускается в отдельном потоке, слушает на localhost:5000
"""

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional


class GameEventHandler(BaseHTTPRequestHandler):
    """HTTP обработчик для /api/game/events"""

    _shared_state = {"events": [], "game_state": {}}

    def do_GET(self):
        """Обработка GET запросов"""
        if self.path == "/api/game/events":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            response = {
                "status": "ok",
                "events": self._shared_state["events"][-10:],  # последние 10 событий
                "game_state": self._shared_state["game_state"],
            }
            self.wfile.write(json.dumps(response).encode())
        elif self.path == "/api/game/state":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            response = {"status": "ok", "state": self._shared_state["game_state"]}
            self.wfile.write(json.dumps(response).encode())
        elif self.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            self.send_response(404)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "not found"}).encode())

    def do_OPTIONS(self):
        """Поддержка CORS preflight"""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        """Подавить стандартный логирование"""
        pass


class GameEventServer:
    """Управление HTTP сервером для событий"""

    def __init__(self, host: str = "localhost", port: int = 5000):
        self.host = host
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False

    def start(self) -> None:
        """Запустить сервер в отдельном потоке"""
        if self.running:
            return

        self.server = HTTPServer((self.host, self.port), GameEventHandler)
        self.running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        print(f"[HTTP] Сервер запущен на http://{self.host}:{self.port}")

    def _run_server(self) -> None:
        """Запуск HTTP сервера (в отдельном потоке)"""
        self.server.serve_forever()

    def stop(self) -> None:
        """Остановить сервер"""
        if self.server:
            self.running = False
            self.server.shutdown()
            self.server.server_close()
            print("[HTTP] Сервер остановлен")

    def emit_event(self, event_type: str, data: dict) -> None:
        """Добавить событие в очередь для передачи через HTTP"""
        event = {"type": event_type, "data": data}
        GameEventHandler._shared_state["events"].append(event)

    def update_game_state(self, state: dict) -> None:
        """Обновить состояние игры для выдачи через /api/game/state"""
        GameEventHandler._shared_state["game_state"] = state

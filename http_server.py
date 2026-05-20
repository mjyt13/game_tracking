"""
HTTP сервер: события игры, MJPEG-стрим, HTML статус-страница, команды управления.
Запускается в отдельном потоке. Доступен по адресу http://<ip>:5000
"""

import json
import queue
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Optional

# ---------- HTML страница статуса (встроена) ----------

_STATUS_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Game Status</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: #111; color: #ddd; }
    #panel {
      position: sticky; top: 0; z-index: 10;
      padding: 10px 14px; background: #111; border-bottom: 1px solid #333;
    }
    #feed { width: 100%; display: block; }
    .row { margin: 6px 0; font-size: 15px; }
    #turn { font-size: 18px; }
    .label { color: #777; font-size: 13px; }
    .player { color: #4fc; font-weight: bold; }
    .chip { color: #ffd166; font-weight: bold; }
    .cell { color: #fc4; }
    .fsm { color: #4af; }
    .rule { color: #f84; }
    #btns { margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }
    button {
      padding: 10px 20px; font-size: 15px; cursor: pointer;
      background: #222; color: #ddd; border: 1px solid #444; border-radius: 6px;
    }
    button:active { background: #444; }
  </style>
</head>
<body>
  <div id="panel">
    <div class="row" id="turn"></div>
    <div class="row" id="players"><span class="label">Players: loading...</span></div>
    <div id="btns">
      <button onclick="cmd('next')">&#9654; Next Turn</button>
    </div>
  </div>
  <img id="feed" src="/stream" alt="Game feed">
  <script>
    function update() {
      fetch('/api/game/state').then(r => r.json()).then(data => {
        const s = data.state || {};
        if (!s.active) {
          const winner = s.winner ? `GAME OVER: <span class="player">${s.winner}</span> wins` : 'Game not started';
          document.getElementById('turn').innerHTML = `<span class="fsm">${winner}</span>`;
          document.getElementById('players').innerHTML = '';
          return;
        }
        const pl = (s.players || []).map(p =>
          `<span class="player">${p.name}</span>&nbsp;<span class="chip">${p.chip_name || ''}</span>&nbsp;<span class="label">cell</span>&nbsp;<span class="cell">${p.cell}</span>`
        ).join('&emsp;');
        document.getElementById('players').innerHTML = pl;
        const cur = s.current_turn;
        let t = cur
          ? `Turn: <span class="player">${cur.player}</span>&nbsp;<span class="chip">[${cur.chip_name}]</span>`
          : '<span class="fsm">Game ready: press Next Turn</span>';
        if (s.turn_state) t += `&nbsp;<span class="fsm">${s.turn_state}</span>`;
        if (s.expected_cell) t += `&nbsp;→&nbsp;cell&nbsp;<span class="cell">${s.expected_cell}</span>`;
        if (s.rule_target)   t += `&nbsp;<span class="rule">rule&nbsp;→&nbsp;${s.rule_target}</span>`;
        document.getElementById('turn').innerHTML = t;
      }).catch(() => {});
    }
    function cmd(name) {
      fetch('/api/game/' + name, { method: 'POST' }).catch(() => {});
    }
    setInterval(update, 1000);
    update();
  </script>
</body>
</html>
""".encode("utf-8")


# ---------- Общее состояние (класс-уровня, GIL-safe для чтения/записи bytes) ----------

class _State:
    events: list = []
    game_state: dict = {}
    stream_frame: bytes = b""          # последний JPEG кадр
    command_queue: queue.Queue = queue.Queue()


# ---------- HTTP обработчик ----------

class GameEventHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/":
            self._send(_STATUS_HTML, "text/html; charset=utf-8")
        elif self.path == "/stream":
            self._mjpeg_stream()
        elif self.path == "/api/game/events":
            self._json({"status": "ok",
                        "events": _State.events[-20:],
                        "game_state": _State.game_state})
        elif self.path == "/api/game/state":
            self._json({"status": "ok", "state": _State.game_state})
        elif self.path == "/api/health":
            self._json({"status": "ok"})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/api/game/next":
            _State.command_queue.put("next")
            self._json({"status": "queued"})
        else:
            self._json({"error": "not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    # ---- helpers ----

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send(self, body: bytes, content_type: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

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
                if frame is last_frame:        # кадр не изменился — не слать дубль
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
        """Подавить стандартный логирование"""
        pass


# ---------- Публичный API ----------

class GameEventServer:
    """Управление HTTP сервером для событий"""
    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        self.host = host
        self.port = port
        self.server: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False

    def start(self) -> None:
        """Запустить сервер в отдельном потоке"""
        if self.running:
            return
        self.server = ThreadingHTTPServer((self.host, self.port), GameEventHandler)
        self.running = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f"[HTTP] Сервер запущен на http://{self.host}:{self.port}")

    def stop(self) -> None:
        """Остановить сервер"""
        if self.server:
            self.running = False
            self.server.shutdown()
            self.server.server_close()
            print("[HTTP] Сервер остановлен")

    def emit_event(self, event_type: str, data: dict) -> None:
        """Добавить событие в очередь для передачи через HTTP"""
        _State.events.append({"type": event_type, "data": data})

    def update_game_state(self, state: dict) -> None:
        """Обновить состояние игры для выдачи через /api/game/state"""
        _State.game_state = state

    def update_stream_frame(self, jpeg_bytes: bytes) -> None:
        """Обновить кадр для MJPEG стрима. Вызывать после отрисовки оверлеев."""
        _State.stream_frame = jpeg_bytes

    def drain_commands(self) -> list[str]:
        """Вернуть все накопленные команды от клиентов (например 'next')."""
        cmds = []
        while not _State.command_queue.empty():
            try:
                cmds.append(_State.command_queue.get_nowait())
            except queue.Empty:
                break
        return cmds

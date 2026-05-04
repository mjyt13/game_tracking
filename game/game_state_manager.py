"""
Сохранение и загрузка состояния игры в JSON.
Позволяет паузировать/возобновлять игру между сессиями.
"""

import json
import datetime
from pathlib import Path
from game.events import Player
from game.game_engine import GameEngine


class GameStateManager:
    """Управление сохранением/загрузкой состояния игры"""

    def __init__(self, save_dir: str = "game_saves"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def save(self, engine: GameEngine, turns_entries: list[dict], field_calibration=None, filename: str | None = None) -> str:
        """Сохранить состояние игры в JSON. Вернуть путь файла."""
        if filename is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"game_{ts}.json"

        filepath = self.save_dir / filename

        # Подготовить данные состояния
        state_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "board_size": engine.state.board_size,
            "turn_state": engine.turn_state,
            "expected_cell": engine.expected_cell,
            "rule_target": engine.rule_target,
            "players": [
                {
                    "name": p.name,
                    "chip_id": p.chip_id,
                    "chip_name": p.chip_name,
                    "cell": p.cell,
                    "skip_turns": p.skip_turns,
                }
                for p in engine.state.players
            ],
            "turns_entries": turns_entries,
            "field_calibration": None,
        }

        # Сохранить калибровку поля если есть
        if field_calibration and field_calibration.is_calibrated():
            state_data["field_calibration"] = {
                "grid_cols": field_calibration.grid_cols,
                "grid_rows": field_calibration.grid_rows,
                "corners": field_calibration.corners,
                "H": field_calibration.H.tolist() if hasattr(field_calibration.H, 'tolist') else None,
                "H_inv": field_calibration.H_inv.tolist() if hasattr(field_calibration.H_inv, 'tolist') else None,
            }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def load(self, filepath: str) -> dict:
        """Загрузить состояние игры из JSON. Вернуть словарь с игровыми данными."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Save file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            state_data = json.load(f)

        return state_data

    def list_saves(self) -> list[dict]:
        """Список всех сохранённых игр."""
        saves = []
        for f in sorted(self.save_dir.glob("game_*.json"), reverse=True):
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    saves.append({
                        "filename": f.name,
                        "path": str(f),
                        "timestamp": data.get("timestamp"),
                        "board_size": data.get("board_size"),
                        "players": [p["name"] for p in data.get("players", [])],
                    })
            except Exception:
                pass
        return saves

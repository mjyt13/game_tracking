"""
Правила поля: клетка → эффект. Загружаются из config/settings.py.
"""
from game.events import Effect
from config import game_rules


class RulesEngine:
    def __init__(self, board_size: int | None = None):
        self.board_size = board_size if board_size is not None else game_rules.BOARD_SIZE
        self._rules: dict[int, Effect] = {
            cell: Effect(type=rule["effect"], distance=rule.get("distance", 0))
            for cell, rule in game_rules.CELL_RULES.items()
        }

    def get_effect(self, cell: int) -> Effect | None:
        return self._rules.get(cell)

    def is_finish(self, cell: int) -> bool:
        return cell >= self.board_size

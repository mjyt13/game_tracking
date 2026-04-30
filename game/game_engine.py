"""
Игровой движок: FSM управления ходом.
Состояния: IDLE → WAITING_MOVE → (AWAITING_RULE) → TURN_DONE → WAITING_MOVE ...
"""
import datetime

from game.events import (
    Player, Event,
    GAME_START, GAME_OVER, TURN_START, CHIP_MOVED, RULE_TRIGGERED,
    SKIP_TURN, EXTRA_TURN, MOVE_FORWARD, MOVE_BACK,
    IDLE, WAITING_MOVE, AWAITING_RULE, TURN_DONE,
)
from game.game_state import GameState
from game.rules_engine import RulesEngine
from config import game_rules


class GameEngine:
    def __init__(self):
        self.state = GameState(game_rules.BOARD_SIZE)
        self.rules = RulesEngine()
        self._log: list[Event] = []
        self.turn_state: str = IDLE
        self.expected_cell: int = 0   # куда должна встать фишка в WAITING_MOVE
        self.rule_target: int = 0     # куда должна встать фишка в AWAITING_RULE
        self._active_chip_id: str | None = None

    def start_game(self, players: list[Player], board_size: int | None = None) -> Event:
        bs = board_size if board_size is not None else game_rules.BOARD_SIZE
        self.state = GameState(bs)
        self.rules = RulesEngine(bs)
        for p in players:
            self.state.add_player(p)
        self.state.active = True
        self.turn_state = IDLE
        self._active_chip_id = None
        return self._emit(GAME_START, {"players": [p.name for p in players], "board_size": bs})

    def begin_turn(self, chip_id: str) -> Event:
        """Начало хода. Вызывается при [N] после TurnManager.advance()."""
        player = self.state.get_player(chip_id)
        if player is None:
            return self._emit(TURN_START, {"player": chip_id[:8]})

        self._active_chip_id = chip_id

        if player.skip_turns > 0:
            player.skip_turns -= 1
            self.turn_state = TURN_DONE
            return self._emit(TURN_START, {
                "player": player.name, "skipped": True,
                "skip_turns_left": player.skip_turns,
            })

        self.turn_state = WAITING_MOVE
        self.expected_cell = player.cell + 1
        return self._emit(TURN_START, {
            "player": player.name, "skipped": False,
            "current_cell": player.cell, "expected_cell": self.expected_cell,
        })

    def on_chip_at_cell(self, chip_id: str, cell: int) -> list[Event]:
        """Вызывается из tracking_loop при смене клетки. Обрабатывает переходы FSM."""
        if not self.state.active or chip_id != self._active_chip_id:
            return []

        player = self.state.get_player(chip_id)
        if player is None:
            return []

        fired: list[Event] = []

        if self.turn_state == WAITING_MOVE:
            if cell != self.expected_cell:
                return []  # неверная клетка — UI покажет инструкцию

            old_cell = player.cell
            self.state.move_player(player, cell)
            fired.append(self._emit(CHIP_MOVED, {"player": player.name, "from": old_cell, "to": cell}))

            if self.rules.is_finish(cell):
                self.state.winner = player
                self.state.active = False
                self.turn_state = IDLE
                fired.append(self._emit(GAME_OVER, {"winner": player.name}))
                return fired

            effect = self.rules.get_effect(cell)
            if effect is None:
                self.turn_state = TURN_DONE
            elif effect.type in (SKIP_TURN, EXTRA_TURN):
                self.state.apply_effect(player, effect)
                fired.append(self._emit(RULE_TRIGGERED, {"player": player.name, "cell": cell, "effect": effect.type}))
                self.turn_state = TURN_DONE
            elif effect.type in (MOVE_FORWARD, MOVE_BACK):
                delta = effect.distance if effect.type == MOVE_FORWARD else -effect.distance
                self.rule_target = max(1, min(player.cell + delta, self.state.board_size))
                fired.append(self._emit(RULE_TRIGGERED, {
                    "player": player.name, "cell": cell,
                    "effect": effect.type, "distance": effect.distance, "target": self.rule_target,
                }))
                self.turn_state = AWAITING_RULE

        elif self.turn_state == AWAITING_RULE:
            if cell != self.rule_target:
                return []  # неверная клетка

            old_cell = player.cell
            self.state.move_player(player, cell)
            fired.append(self._emit(CHIP_MOVED, {
                "player": player.name, "from": old_cell, "to": cell, "rule_executed": True,
            }))
            self.turn_state = TURN_DONE

        return fired

    def status_msg(self) -> tuple[str, tuple[int, int, int]] | None:
        """Текст и цвет для отображения статуса FSM на кадре. None если нечего показывать."""
        if not self.state.active:
            return None
        if self.turn_state == WAITING_MOVE:
            return f"→ Move chip to cell {self.expected_cell}", (0, 220, 255)
        if self.turn_state == AWAITING_RULE:
            return f"→ Rule: move chip to cell {self.rule_target}", (0, 140, 255)
        if self.turn_state == TURN_DONE:
            return "→ Press [N] for next turn", (100, 255, 100)
        return None

    def can_advance(self) -> bool:
        """True если разрешено переходить к следующему ходу."""
        return self.turn_state in (IDLE, TURN_DONE)

    @property
    def is_active(self) -> bool:
        return self.state.active

    def get_log(self) -> list[Event]:
        return list(self._log)

    def _emit(self, event_type: str, data: dict | None = None) -> Event:
        e = Event(type=event_type, data=data or {})
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] EVENT {e.type}: {e.data}")
        self._log.append(e)
        return e

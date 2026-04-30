"""
Константы событий и эффектов игрового движка. Сущности Player и Effect.
"""
from dataclasses import dataclass, field

# --- Состояния хода (FSM) ---
IDLE          = "IDLE"
WAITING_MOVE  = "WAITING_MOVE"
AWAITING_RULE = "AWAITING_RULE"
TURN_DONE     = "TURN_DONE"

# --- Типы событий ---
GAME_START      = "GAME_START"
GAME_OVER       = "GAME_OVER"
TURN_START      = "TURN_START"
CHIP_MOVED      = "CHIP_MOVED"      # сканер зафиксировал перемещение фишки
RULE_TRIGGERED  = "RULE_TRIGGERED"  # сработало правило клетки

# --- Типы эффектов ---
SKIP_TURN       = "SKIP_TURN"
EXTRA_TURN      = "EXTRA_TURN"
MOVE_FORWARD    = "MOVE_FORWARD"
MOVE_BACK       = "MOVE_BACK"


@dataclass
class Player:
    name:       str
    chip_id:    str
    chip_name:  str
    cell:       int = 0   # текущая клетка (0 = вне поля / не начал)
    skip_turns: int = 0   # ходов к пропуску


@dataclass
class Effect:
    type:     str         # SKIP_TURN | EXTRA_TURN | MOVE_FORWARD | MOVE_BACK
    distance: int = 0     # для MOVE_FORWARD / MOVE_BACK


@dataclass
class Event:
    type: str
    data: dict = field(default_factory=dict)

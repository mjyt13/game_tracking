"""
Правила игрового поля. Редактируй здесь — константы эффектов импортированы,
опечатка приведёт к NameError, а не к тихому игнорированию.
"""
from game.events import SKIP_TURN, EXTRA_TURN, MOVE_FORWARD, MOVE_BACK

BOARD_SIZE = 12  # дефолт; при старте игры переопределяется как grid_cols × grid_rows поля

CELL_RULES: dict[int, dict] = {
    # номер_клетки: {"effect": <SKIP_TURN|EXTRA_TURN|MOVE_FORWARD|MOVE_BACK>, "distance": N}
    5:  {"effect": SKIP_TURN},
    9:  {"effect": MOVE_BACK,    "distance": 3},
    2: {"effect": MOVE_FORWARD, "distance": 2},
    3:  {"effect": EXTRA_TURN},
}

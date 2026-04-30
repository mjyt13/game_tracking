"""
Состояние игры: игроки, позиции, победитель.
"""
from game.events import Player, Effect, MOVE_FORWARD, MOVE_BACK, SKIP_TURN, EXTRA_TURN


class GameState:
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.players:   list[Player] = []
        self.active:    bool = False
        self.winner:    Player | None = None

    def add_player(self, player: Player) -> None:
        self.players.append(player)

    def get_player(self, chip_id: str) -> Player | None:
        return next((p for p in self.players if p.chip_id == chip_id), None)

    def move_player(self, player: Player, to_cell: int) -> None:
        player.cell = max(1, min(to_cell, self.board_size))

    def apply_effect(self, player: Player, effect: Effect) -> None:
        if effect.type == MOVE_FORWARD:
            self.move_player(player, player.cell + effect.distance)
        elif effect.type == MOVE_BACK:
            self.move_player(player, player.cell - effect.distance)
        elif effect.type == SKIP_TURN:
            player.skip_turns += 1
        elif effect.type == EXTRA_TURN:
            player.skip_turns = max(0, player.skip_turns - 1)

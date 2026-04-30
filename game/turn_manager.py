"""
Очередь ходов: связывает игроков с зарегистрированными фишками.
"""


class TurnManager:
    def __init__(self):
        self._entries: list[dict] = []  # {"player": str, "chip_id": str, "chip_name": str}
        self._index: int = 0
        self._focused: bool = False

    def add(self, player_name: str, chip_id: str, chip_name: str) -> None:
        self._entries.append({"player": player_name, "chip_id": chip_id, "chip_name": chip_name})

    def remove_by_chip(self, chip_id: str) -> None:
        self._entries = [e for e in self._entries if e["chip_id"] != chip_id]
        if self._entries:
            self._index = self._index % len(self._entries)
        else:
            self._index = 0
            self._focused = False

    def remove(self, value: str) -> bool:
        """Удалить игрока по 1-based номеру в очереди или по имени. True если нашёл."""
        chip_id = None
        try:
            idx = int(value) - 1
            if 0 <= idx < len(self._entries):
                chip_id = self._entries[idx]["chip_id"]
        except ValueError:
            pass
        if chip_id is None:
            for e in self._entries:
                if e["player"] == value:
                    chip_id = e["chip_id"]
                    break
        if chip_id:
            self.remove_by_chip(chip_id)
            return True
        return False

    def start(self) -> dict | None:
        """Начать с первого игрока."""
        if not self._entries:
            return None
        self._focused = True
        self._index = 0
        return self.current()

    def advance(self) -> dict | None:
        """Следующий игрок. Запускает очередь если ещё не активна."""
        if not self._entries:
            return None
        if not self._focused:
            self._focused = True
            self._index = 0
        else:
            self._index = (self._index + 1) % len(self._entries)
        return self.current()

    def current(self) -> dict | None:
        if not self._focused or not self._entries:
            return None
        return self._entries[self._index]

    @property
    def active_chip_id(self) -> str | None:
        entry = self.current()
        return entry["chip_id"] if entry else None

    def unfocus(self) -> None:
        """Снять фокус — детектировать все фишки."""
        self._focused = False

    def is_focused(self) -> bool:
        return self._focused

    def list_entries(self) -> list[dict]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

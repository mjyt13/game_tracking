"""
Управление путём на игровом поле.
Поддерживает линейный, змейка и спиральный порядок клеток.
"""


class PathManager:
    """Определяет порядок прохождения клеток на поле"""

    def __init__(self, path_type: str = "linear", grid_cols: int = 3, grid_rows: int = 4):
        """
        path_type: 'linear' | 'snake' | 'spiral'
        grid_cols, grid_rows: размеры поля
        """
        self.path_type = path_type
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.order = self._compute_order()

    def _compute_order(self) -> list[tuple[int, int]]:
        """Вычислить порядок обхода клеток (row, col) для типа пути."""
        if self.path_type == "linear":
            return self._linear_order()
        elif self.path_type == "snake":
            return self._snake_order()
        elif self.path_type == "spiral":
            return self._spiral_order()
        else:
            return self._linear_order()

    def _linear_order(self) -> list[tuple[int, int]]:
        """Линейный порядок: слева направо, сверху вниз."""
        order = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                order.append((row, col))
        return order

    def _snake_order(self) -> list[tuple[int, int]]:
        """Змейка: четные строки слева направо, нечетные справа налево."""
        order = []
        for row in range(self.grid_rows):
            if row % 2 == 0:
                # Четная строка: слева направо
                for col in range(self.grid_cols):
                    order.append((row, col))
            else:
                # Нечетная строка: справа налево
                for col in range(self.grid_cols - 1, -1, -1):
                    order.append((row, col))
        return order

    def _spiral_order(self) -> list[tuple[int, int]]:
        """Спираль: от края к центру по часовой стрелке."""
        order = []
        top, bottom = 0, self.grid_rows - 1
        left, right = 0, self.grid_cols - 1

        while top <= bottom and left <= right:
            # Верхняя строка (слева направо)
            for col in range(left, right + 1):
                order.append((top, col))
            top += 1

            # Правый столбец (сверху вниз)
            for row in range(top, bottom + 1):
                order.append((row, right))
            right -= 1

            # Нижняя строка (справа налево)
            if top <= bottom:
                for col in range(right, left - 1, -1):
                    order.append((bottom, col))
                bottom -= 1

            # Левый столбец (снизу вверх)
            if left <= right:
                for row in range(bottom, top - 1, -1):
                    order.append((row, left))
                left += 1

        return order

    def cell_number_to_position(self, cell_num: int) -> tuple[int, int] | None:
        """Преобразовать номер клетки (1-indexed) в позицию (row, col)."""
        if 1 <= cell_num <= len(self.order):
            return self.order[cell_num - 1]
        return None

    def position_to_cell_number(self, row: int, col: int) -> int | None:
        """Преобразовать позицию (row, col) в номер клетки (1-indexed)."""
        try:
            idx = self.order.index((row, col))
            return idx + 1
        except ValueError:
            return None

    def get_board_size(self) -> int:
        """Вернуть общее число клеток."""
        return len(self.order)

"""
Профайлер операций трекинг-цикла.
Включается через settings.PROFILING_ENABLED.
Сохраняет базовые показатели в metrics/profile_baseline.csv при нажатии [P].
"""

import csv
import time
from collections import deque
from contextlib import contextmanager


class Profiler:
    # field — рисование сетки/заливки клеток/номеров; draw — панель команд cv2;
    # show — resize+imshow+окно метрик; mjpeg — JPEG-кодирование кадра для стрима
    SECTIONS = ['camera', 'extract', 'detect', 'tracker', 'field', 'draw', 'show', 'mjpeg']

    def __init__(self, window: int = 60, output_file: str = 'metrics/profile_baseline.csv'):
        self._times: dict[str, deque] = {s: deque(maxlen=window) for s in self.SECTIONS}
        self._frame: int = 0
        self._output_file = output_file
        self.registered: int = 0   # число зарегистрированных фишек
        self.active: int = 0       # число одновременно детектируемых фишек

    def set_counts(self, registered: int, active: int) -> None:
        """Сообщить профилировщику число объектов для вывода в сводке (раздел 4.3)."""
        self.registered = registered
        self.active = active

    @contextmanager
    def measure(self, section: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._times[section].append((time.perf_counter() - t0) * 1000)

    def next_frame(self):
        self._frame += 1
        if self._frame % 60 == 0:
            self._print_summary()

    def _averages(self) -> dict[str, float]:
        return {s: (sum(q) / len(q) if q else 0.0) for s, q in self._times.items()}

    def _print_summary(self):
        avgs = self._averages()
        total = sum(avgs.values())
        fps = 1000.0 / total if total else 0.0
        print(f"\n[profiler] frame={self._frame}  total={total:.1f} ms  (~{fps:.1f} fps)  "
              f"объектов: рег={self.registered} актив={self.active}")
        for s in self.SECTIONS:
            pct = avgs[s] / total * 100 if total else 0
            print(f"  {s:<10} {avgs[s]:6.1f} ms  ({pct:.0f}%)")

    def save_baseline(self):
        """Сохранить текущие средние в CSV (базовая линия для оптимизации)."""
        avgs = self._averages()
        total = sum(avgs.values())
        rows = [{'section': s, 'avg_ms': f'{avgs[s]:.2f}',
                 'pct': f'{avgs[s]/total*100:.1f}' if total else '0'}
                for s in self.SECTIONS]
        rows.append({'section': 'TOTAL', 'avg_ms': f'{total:.2f}', 'pct': '100'})

        with open(self._output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['section', 'avg_ms', 'pct'])
            writer.writeheader()
            writer.writerows(rows)

        print(f"[profiler] baseline saved → {self._output_file}")
        self._print_summary()

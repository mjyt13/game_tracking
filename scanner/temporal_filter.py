"""
Temporal stability filter for SIFT keypoints.

Stable keypoints persist at the same spatial position across multiple frames —
typical of static objects (game chips). Transient keypoints appear at new
locations each frame — typical of animated or changing backgrounds.

Algorithm:
  - Divide the frame into a grid of cell_size × cell_size pixel cells.
  - Quantize each keypoint's (x, y) position to its grid cell.
  - Maintain a rolling history of N frame cell-sets.
  - A keypoint in the current frame is "stable" if its grid cell appeared
    in at least min_hits of the previous frames.
  - Transient keypoints are dropped before FLANN matching.

Fallback guarantees:
  - During warm-up (history shorter than min_hits): all keypoints pass through.
  - If the filter would drop all keypoints: all keypoints pass through.

Integration:
  Wired in tracking_loop (game_scanner.py) after extract_features(), before
  detect_objects(). Controlled by TEMPORAL_FILTER_ENABLED in settings.py.
  reset() called after every registration to avoid stale history.

Parameters live in config/settings.py (TEMPORAL_FILTER_*).
"""

import numpy as np
import cv2
from config import settings


class TemporalStabilityFilter:
    """
    Grid-based temporal keypoint filter. O(N) per frame.

    Quantizes keypoint positions to cells of cell_size × cell_size pixels.
    Keeps only keypoints whose grid cell was occupied in at least min_hits
    of the last window_frames frames.
    """

    def __init__(self):
        self._cell: int = settings.TEMPORAL_FILTER_CELL_SIZE
        self._window: int = settings.TEMPORAL_FILTER_FRAMES
        self._min_hits: int = settings.TEMPORAL_FILTER_MIN_HITS
        # Rolling history: each entry is a set of (col, row) grid cells occupied
        self._history: list[set[tuple[int, int]]] = []

    def filter(
        self,
        keypoints: list[cv2.KeyPoint],
        descriptors: np.ndarray,
    ) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        """
        Return (keypoints, descriptors) with transient keypoints removed.
        keypoints[i] and descriptors[i] must correspond (standard OpenCV layout).
        """
        if not keypoints or descriptors is None:
            self._push(set())
            return keypoints, descriptors

        # Quantize current keypoints to grid cells
        cells: list[tuple[int, int]] = [
            (int(kp.pt[0] / self._cell), int(kp.pt[1] / self._cell))
            for kp in keypoints
        ]
        cur_set = set(cells)

        # Compute stable cells from history accumulated so far (before this frame)
        stable: set[tuple[int, int]] = set()
        if len(self._history) >= self._min_hits:
            for cell in cur_set:
                hits = sum(1 for past in self._history if cell in past)
                if hits >= self._min_hits:
                    stable.add(cell)

        self._push(cur_set)

        # Warm-up: not enough history yet — pass all through
        if len(self._history) <= self._min_hits:
            return keypoints, descriptors

        # Fallback: filter would remove everything — pass all through
        if not stable:
            return keypoints, descriptors

        mask = np.array([c in stable for c in cells], dtype=bool)
        return [kp for kp, keep in zip(keypoints, mask) if keep], descriptors[mask]

    def reset(self) -> None:
        """Clear history. Call after registration or significant scene change."""
        self._history.clear()

    def stats(self) -> dict:
        """Return diagnostics for logging / profiling."""
        total_past = sum(len(s) for s in self._history)
        return {
            "history_frames": len(self._history),
            "avg_cells_per_frame": total_past / max(1, len(self._history)),
        }

    def _push(self, cell_set: set) -> None:
        self._history.append(cell_set)
        if len(self._history) > self._window:
            self._history.pop(0)

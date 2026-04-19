from collections import deque

import numpy as np

import config
from cell_types import CellType, MUTABLE_CYCLE, PASSABLE_TYPES


class CityGrid:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed
        self.cells: np.ndarray = np.zeros((size, size), dtype=np.int8)
        self._noise_rng: np.random.Generator = np.random.default_rng(seed ^ 0xDEAD)
        self._noise_seed_base = seed ^ 0xDEAD

    def generate(self) -> None:
        self._cells_snapshot: np.ndarray | None = None
        for attempt in range(100):
            current_seed = self.seed + attempt
            rng = np.random.default_rng(current_seed)
            total = self.size * self.size
            cells = np.full(total, int(CellType.ROAD), dtype=np.int8)

            n_obs = int(total * config.FRAC_OBSTACLE)
            n_light = int(total * config.FRAC_LIGHT)
            n_heavy = int(total * config.FRAC_HEAVY)
            n_c20 = int(total * config.FRAC_CARP20)
            n_c50 = int(total * config.FRAC_CARP50)

            indices = rng.permutation(total)
            cursor = 0

            def _place(ct: CellType, count: int, offset: int) -> int:
                cells[indices[offset: offset + count]] = int(ct)
                return offset + count

            cursor = _place(CellType.OBSTACLE, n_obs, cursor)
            cursor = _place(CellType.LIGHT_TRAFFIC, n_light, cursor)
            cursor = _place(CellType.HEAVY_TRAFFIC, n_heavy, cursor)
            cursor = _place(CellType.CARPOOLER_20, n_c20, cursor)
            cursor = _place(CellType.CARPOOLER_50, n_c50, cursor)

            grid = cells.reshape((self.size, self.size))
            sr, sc = config.START
            er, ec = config.END
            grid[sr, sc] = int(CellType.START)
            grid[er, ec] = int(CellType.END)

            if self._bfs_reachable(grid, sr, sc, er, ec):
                self.cells = grid
                self._cells_snapshot = grid.copy()
                self.seed = current_seed
                return

        raise RuntimeError("Failed to generate a passable grid after 100 retries.")

    def _bfs_reachable(self, grid: np.ndarray, sr: int, sc: int, er: int, ec: int) -> bool:
        visited = np.zeros((self.size, self.size), dtype=bool)
        queue: deque = deque()
        queue.append((sr, sc))
        visited[sr, sc] = True
        while queue:
            r, c = queue.popleft()
            if r == er and c == ec:
                return True
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size and not visited[nr, nc]:
                    if CellType(grid[nr, nc]) != CellType.OBSTACLE:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
        return False

    def get_cell(self, r: int, c: int) -> CellType:
        return CellType(self.cells[r, c])

    def apply_noise(self, step: int) -> tuple[int, int] | None:  # noqa: F821
        if self._noise_rng.random() >= config.P_NOISE:
            return None
        candidates = []
        sr, sc = config.START
        er, ec = config.END
        for r in range(self.size):
            for c in range(self.size):
                ct = CellType(self.cells[r, c])
                if (r == sr and c == sc) or (r == er and c == ec):
                    continue
                if ct == CellType.OBSTACLE:
                    continue
                if ct in MUTABLE_CYCLE:
                    candidates.append((r, c))
        if not candidates:
            return None
        idx = int(self._noise_rng.integers(0, len(candidates)))
        r, c = candidates[idx]
        ct = CellType(self.cells[r, c])
        cycle_idx = MUTABLE_CYCLE.index(ct)
        next_ct = MUTABLE_CYCLE[(cycle_idx + 1) % len(MUTABLE_CYCLE)]
        self.cells[r, c] = int(next_ct)
        return (r, c)

    def reset_noise_rng(self) -> None:
        self._noise_rng = np.random.default_rng(self._noise_seed_base)
        if self._cells_snapshot is not None:
            self.cells = self._cells_snapshot.copy()

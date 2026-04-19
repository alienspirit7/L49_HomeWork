import heapq
import time
from collections import deque

import numpy as np

import config
from cell_types import CellType, REWARD_MAP, PASSABLE_TYPES

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class BellmanAgent:
    def __init__(self, grid_size: int, gamma: float) -> None:
        self.size = grid_size
        self.gamma = gamma
        self.V: np.ndarray = np.zeros((grid_size, grid_size), dtype=float)
        self._last_ops: int = 0

    def _reward(self, ct: CellType) -> float:
        if ct == CellType.UNKNOWN:
            return REWARD_MAP[CellType.ROAD]
        return REWARD_MAP.get(ct, REWARD_MAP[CellType.ROAD])

    def _is_passable(self, ct: CellType) -> bool:
        return ct in PASSABLE_TYPES

    def reset(self, belief_map: np.ndarray) -> None:
        self.V = np.zeros((self.size, self.size), dtype=float)
        er, ec = config.END
        self.V[er, ec] = 0.0  # terminal absorbing state; reward for entering END is captured in neighbours' Bellman updates
        while True:
            delta = 0.0
            for r in range(self.size):
                for c in range(self.size):
                    ct = CellType(belief_map[r, c])
                    if not self._is_passable(ct):
                        continue
                    if r == er and c == ec:
                        continue
                    best = float('-inf')
                    for dr, dc in ACTIONS:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.size and 0 <= nc < self.size:
                            nct = CellType(belief_map[nr, nc])
                            if self._is_passable(nct):
                                val = self._reward(nct) + self.gamma * self.V[nr, nc]
                                if val > best:
                                    best = val
                    if best > float('-inf'):
                        old = self.V[r, c]
                        self.V[r, c] = best
                        delta = max(delta, abs(old - best))
            if delta < config.BELLMAN_THETA:
                break

    def update_belief(self, belief_map: np.ndarray, changed_cells: list[tuple[int, int]]) -> int:
        heap: list = []
        er, ec = config.END
        for r, c in changed_cells:
            for dr, dc in ACTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    heapq.heappush(heap, (-abs(self.V[nr, nc]), nr, nc))
        ops = 0
        t_start = time.perf_counter()
        visited = set()
        while heap and ops < config.BELLMAN_MAX_SWEEP_OPS:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            if elapsed_ms > config.BELLMAN_SWEEP_TIME_MS:
                break
            _, r, c = heapq.heappop(heap)
            if (r, c) in visited:
                continue
            visited.add((r, c))
            ct = CellType(belief_map[r, c])
            if not self._is_passable(ct) or (r == er and c == ec):
                ops += 1
                continue
            best = float('-inf')
            for dr, dc in ACTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    nct = CellType(belief_map[nr, nc])
                    if self._is_passable(nct):
                        val = self._reward(nct) + self.gamma * self.V[nr, nc]
                        if val > best:
                            best = val
            if best > float('-inf'):
                self.V[r, c] = best
            ops += 1
        self._last_ops = ops
        return ops

    def select_action(self, r: int, c: int, fog: 'FogMap', grid: 'CityGrid') -> int:  # noqa: F821
        best_action = -1
        best_val = float('-inf')
        for a, (dr, dc) in enumerate(ACTIONS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                ct = grid.get_cell(nr, nc)
                if self._is_passable(ct):
                    val = self._reward(ct) + self.gamma * self.V[nr, nc]
                    if val > best_val:
                        best_val = val
                        best_action = a
        if best_action != -1:
            return best_action
        return self._bfs_action(r, c, grid)

    def _bfs_action(self, r: int, c: int, grid: 'CityGrid') -> int:  # noqa: F821
        er, ec = config.END
        queue: deque = deque()
        queue.append((r, c, -1))
        visited = {(r, c)}
        while queue:
            cr, cc, first_action = queue.popleft()
            for a, (dr, dc) in enumerate(ACTIONS):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in visited:
                    ct = grid.get_cell(nr, nc)
                    if self._is_passable(ct):
                        fa = a if first_action == -1 else first_action
                        if nr == er and nc == ec:
                            return fa
                        visited.add((nr, nc))
                        queue.append((nr, nc, fa))
        return 0

    def get_v_stats(self) -> str:
        er, ec = config.END
        return f"V={self.V[er, ec]:.1f} ops={self._last_ops}"

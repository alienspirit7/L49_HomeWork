import numpy as np

import config
from cell_types import CellType, PASSABLE_TYPES

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class QLearningAgent:
    def __init__(
        self,
        grid_size: int,
        alpha: float,
        gamma: float,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay: float,
    ) -> None:
        self.size = grid_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q: np.ndarray = np.zeros((grid_size, grid_size, 4), dtype=float)
        self._epsilon: float = epsilon_start
        self._first_reset: bool = True
        self._rng = np.random.default_rng(config.NOISE_SEED)

    def reset_episode(self) -> None:
        if self._first_reset:
            self._epsilon = self.epsilon_start
            self._first_reset = False

    def _is_passable(self, ct: CellType) -> bool:
        return ct in PASSABLE_TYPES

    def select_action(self, r: int, c: int, fog: 'FogMap', grid: 'CityGrid') -> int:  # noqa: F821
        valid_actions = []
        for a, (dr, dc) in enumerate(ACTIONS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if fog.is_revealed(nr, nc):
                    ct = grid.get_cell(nr, nc)
                    if ct != CellType.OBSTACLE:
                        valid_actions.append(a)
                else:
                    valid_actions.append(a)
        if not valid_actions:
            return int(self._rng.integers(0, 4))
        if self._rng.random() < self._epsilon:
            idx = int(self._rng.integers(0, len(valid_actions)))
            return valid_actions[idx]
        q_vals = [self.Q[r, c, a] for a in valid_actions]
        best_idx = int(np.argmax(q_vals))
        return valid_actions[best_idx]

    def update(self, r: int, c: int, action: int, reward: float, r2: int, c2: int) -> None:
        best_next = float(np.max(self.Q[r2, c2, :]))
        td_error = reward + self.gamma * best_next - self.Q[r, c, action]
        self.Q[r, c, action] += self.alpha * td_error
        self._epsilon = max(self.epsilon_min, self._epsilon * self.epsilon_decay)

    def get_epsilon(self) -> float:
        return self._epsilon

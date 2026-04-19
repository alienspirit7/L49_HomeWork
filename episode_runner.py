from __future__ import annotations
from dataclasses import dataclass

import numpy as np

import config
from cell_types import CellType, REWARD_MAP

# Action index → (delta_row, delta_col): 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


@dataclass
class FrameResult:
    position: tuple[int, int]
    revealed_cells: list[tuple[int, int]]
    noise_cell: tuple[int, int] | None
    collected_cell: tuple[int, int] | None
    reward_delta: float
    cumulative_reward: float
    step: int
    done: bool
    failed: bool
    sweep_ops: int
    path: list[tuple[int, int]]


class EpisodeRunner:
    def __init__(self, grid, fog, agent, agent_type: str) -> None:
        self.grid = grid
        self.fog = fog
        self.agent = agent
        self.agent_type = agent_type
        self._pos: tuple[int, int] = config.START
        self._reward: float = 0.0
        self._step: int = 0
        self._path: list[tuple[int, int]] = []
        self._belief: np.ndarray = np.full(
            (config.GRID_SIZE, config.GRID_SIZE), -1, dtype=np.int8
        )

    def reset(self, run_index: int = 0) -> None:
        self._pos = config.START
        self._reward = 0.0
        self._step = 0
        self._path = [config.START]
        self.fog.reset()
        if run_index > 0:
            self.grid.reset_noise_rng()
        belief = np.full((config.GRID_SIZE, config.GRID_SIZE), -1, dtype=np.int8)
        sr, sc = config.START
        er, ec = config.END
        belief[sr, sc] = int(self.grid.get_cell(sr, sc))
        belief[er, ec] = int(self.grid.get_cell(er, ec))
        self._belief = belief
        if self.agent_type == 'bellman':
            self.agent.reset(self._belief)
        else:
            self.agent.reset_episode()

    def tick(self) -> FrameResult:
        r, c = self._pos

        # Phase 0: reveal fog
        newly_revealed = self.fog.reveal_around(r, c, config.REVEAL_RADIUS)
        for nr, nc in newly_revealed:
            self._belief[nr, nc] = int(self.grid.get_cell(nr, nc))
        sweep_ops = 0
        if newly_revealed and self.agent_type == 'bellman':
            sweep_ops = self.agent.update_belief(self._belief, newly_revealed)

        # Phase 1: apply noise
        noise_cell = self.grid.apply_noise(self._step)
        if noise_cell is not None:
            nr, nc = noise_cell
            if self.fog.is_revealed(nr, nc):
                self._belief[nr, nc] = int(self.grid.get_cell(nr, nc))
                if self.agent_type == 'bellman':
                    sweep_ops += self.agent.update_belief(self._belief, [noise_cell])

        # Phase 2: action
        action = self.agent.select_action(r, c, self.fog, self.grid)
        dr, dc = _DELTA[action]
        new_r = max(0, min(config.GRID_SIZE - 1, r + dr))
        new_c = max(0, min(config.GRID_SIZE - 1, c + dc))

        cell = self.grid.get_cell(new_r, new_c)
        collected_cell = None
        if cell == CellType.OBSTACLE:
            # Blocked — stay in place, apply wall penalty
            new_r, new_c = r, c
            reward = -5.0
        else:
            reward = REWARD_MAP.get(cell, -1)
            if cell in (CellType.CARPOOLER_20, CellType.CARPOOLER_50):
                # Carpooler picked up — convert cell to ROAD so it can't be farmed
                self.grid.cells[new_r, new_c] = int(CellType.ROAD)
                self._belief[new_r, new_c] = int(CellType.ROAD)
                collected_cell = (new_r, new_c)
        new_pos = (new_r, new_c)

        if self.agent_type == 'qlearning':
            self.agent.update(r, c, action, reward, new_r, new_c)

        self._pos = new_pos
        self._reward += reward
        self._step += 1
        self._path.append(new_pos)

        done = new_pos == config.END
        failed = (not done) and self._step >= config.MAX_STEPS

        return FrameResult(
            position=new_pos,
            revealed_cells=newly_revealed,
            noise_cell=noise_cell,
            collected_cell=collected_cell,
            reward_delta=reward,
            cumulative_reward=self._reward,
            step=self._step,
            done=done,
            failed=failed,
            sweep_ops=sweep_ops,
            path=list(self._path),
        )

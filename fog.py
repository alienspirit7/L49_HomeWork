import numpy as np


class FogMap:
    def __init__(self, size: int, start: tuple, end: tuple) -> None:
        self.size = size
        self.start = start
        self.end = end
        self.revealed: np.ndarray = np.zeros((size, size), dtype=bool)
        self._pre_reveal()

    def _pre_reveal(self) -> None:
        sr, sc = self.start
        er, ec = self.end
        self.revealed[sr, sc] = True
        self.revealed[er, ec] = True

    def reveal_around(self, r: int, c: int, radius: int) -> list[tuple[int, int]]:
        newly: list[tuple[int, int]] = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if not self.revealed[nr, nc]:
                        self.revealed[nr, nc] = True
                        newly.append((nr, nc))
        return newly

    def is_revealed(self, r: int, c: int) -> bool:
        return bool(self.revealed[r, c])

    def reset(self) -> None:
        self.revealed[:] = False
        self._pre_reveal()

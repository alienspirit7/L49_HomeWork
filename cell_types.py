from enum import IntEnum


class CellType(IntEnum):
    UNKNOWN = -1
    ROAD = 0
    LIGHT_TRAFFIC = 1
    HEAVY_TRAFFIC = 2
    OBSTACLE = 3
    CARPOOLER_20 = 4
    CARPOOLER_50 = 5
    START = 6
    END = 7


REWARD_MAP = {
    CellType.ROAD: -1,
    CellType.LIGHT_TRAFFIC: -5,
    CellType.HEAVY_TRAFFIC: -10,
    CellType.OBSTACLE: float('-inf'),
    CellType.CARPOOLER_20: +20,
    CellType.CARPOOLER_50: +50,
    CellType.START: -1,
    CellType.END: +1000,
    CellType.UNKNOWN: -1,  # optimistic assumption
}

MUTABLE_CYCLE = [
    CellType.ROAD,
    CellType.LIGHT_TRAFFIC,
    CellType.HEAVY_TRAFFIC,
    CellType.CARPOOLER_20,
    CellType.CARPOOLER_50,
]

PASSABLE_TYPES = frozenset([
    CellType.ROAD, CellType.LIGHT_TRAFFIC, CellType.HEAVY_TRAFFIC,
    CellType.CARPOOLER_20, CellType.CARPOOLER_50, CellType.START,
    CellType.END, CellType.UNKNOWN,
])

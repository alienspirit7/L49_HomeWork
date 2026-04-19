"""icons.py — Patch-based cell icons and car sprite (no emoji, cross-platform)."""
from __future__ import annotations
import matplotlib.patches as mp
import matplotlib.pyplot as plt


def _mini_car(ax, cx: float, cy: float, scale: float, body_color: str) -> list:
    """Draw a simplified car at (cx, cy) using patches. Returns artist list."""
    arts = []
    w, h = 0.28 * scale, 0.13 * scale
    roof_w, roof_h = 0.16 * scale, 0.09 * scale
    wheel_r = 0.038 * scale
    arts.append(ax.add_patch(mp.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.01", facecolor=body_color,
        edgecolor='#111111', linewidth=0.4, zorder=4)))
    arts.append(ax.add_patch(mp.FancyBboxPatch(
        (cx - roof_w / 2, cy + h / 2 - roof_h * 0.3), roof_w, roof_h,
        boxstyle="round,pad=0.01", facecolor=body_color,
        edgecolor='#111111', linewidth=0.3, zorder=4)))
    for wx in (cx - w * 0.30, cx + w * 0.30):
        arts.append(ax.add_patch(mp.Circle(
            (wx, cy - h / 2 - wheel_r * 0.5), wheel_r,
            facecolor='#222222', edgecolor='none', zorder=5)))
    return arts


def _person(ax, cx: float, cy: float, scale: float) -> list:
    """Draw a stick-figure head + body hint. Returns artist list."""
    arts = []
    r = 0.10 * scale
    arts.append(ax.add_patch(mp.Circle(
        (cx, cy + 0.12 * scale), r,
        facecolor='#eecc88', edgecolor='#555533', linewidth=0.5, zorder=4)))
    arts.append(ax.add_patch(mp.FancyBboxPatch(
        (cx - 0.07 * scale, cy - 0.08 * scale), 0.14 * scale, 0.18 * scale,
        boxstyle="round,pad=0.01", facecolor='#eecc88',
        edgecolor='#555533', linewidth=0.4, zorder=4)))
    return arts


def draw_cell_icon(ax, r: int, c: int, cell_type_int: int, grid_size: int) -> list:
    """Draw icon patches for a revealed cell. Returns list of artists."""
    from cell_types import CellType
    cx = c + 0.5
    cy = grid_size - 1 - r + 0.5
    ct = cell_type_int

    if ct == int(CellType.LIGHT_TRAFFIC):
        a1 = _mini_car(ax, cx - 0.22, cy, 0.70, '#222233')
        a2 = _mini_car(ax, cx + 0.22, cy, 0.70, '#222233')
        return a1 + a2

    if ct == int(CellType.HEAVY_TRAFFIC):
        a1 = _mini_car(ax, cx - 0.30, cy, 0.60, '#220000')
        a2 = _mini_car(ax, cx,        cy, 0.60, '#220000')
        a3 = _mini_car(ax, cx + 0.30, cy, 0.60, '#220000')
        return a1 + a2 + a3

    if ct in (int(CellType.CARPOOLER_20), int(CellType.CARPOOLER_50)):
        amount = 20 if ct == int(CellType.CARPOOLER_20) else 50
        arts = _person(ax, cx, cy, 1.0)
        arts.append(ax.text(cx, cy - 0.22, f'${amount}',
                            ha='center', va='center', fontsize=6,
                            color='white', fontweight='bold', zorder=6))
        return arts

    if ct == int(CellType.START):
        arts = [ax.text(cx, cy, 'S', ha='center', va='center',
                        fontsize=10, color='#111122', fontweight='bold', zorder=4)]
        return arts

    if ct == int(CellType.END):
        arts = [ax.text(cx, cy, 'E', ha='center', va='center',
                        fontsize=10, color='#111122', fontweight='bold', zorder=4)]
        return arts

    return []


def make_car_marker(ax, grid_size: int) -> list:
    """Create a car marker: colored diamond + outline. Returns [patch, patch]."""
    arts = [
        ax.add_patch(mp.RegularPolygon(
            (-10, -10), 4, radius=0.28,
            orientation=0.785, facecolor='#00ffcc',
            edgecolor='white', linewidth=1.2, zorder=15)),
        ax.text(-10, -10, '▶', ha='center', va='center',
                fontsize=6, color='#111122', zorder=16),
    ]
    return arts


def move_car_marker(arts: list, r: int, c: int, grid_size: int) -> None:
    """Move the car marker to cell (r, c)."""
    cx = c + 0.5
    cy = grid_size - 1 - r + 0.5
    arts[0].xy = (cx, cy)
    arts[1].set_position((cx, cy))

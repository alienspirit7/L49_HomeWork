from __future__ import annotations
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import config
from episode_runner import FrameResult
from ui_layout import UIComponents
from icons import draw_cell_icon, make_car_marker, move_car_marker

_COLORS = {-1: config.COLOR_FOG, 0: config.COLOR_ROAD, 1: config.COLOR_LIGHT,
           2: config.COLOR_HEAVY, 3: config.COLOR_OBSTACLE, 4: config.COLOR_CARP20,
           5: config.COLOR_CARP50, 6: config.COLOR_START, 7: config.COLOR_END}

_FRAME_INTERVAL_MS = 50   # fixed ~20fps; speed controls steps-per-frame


def _cell_color(ct: int) -> str:
    return _COLORS.get(ct, config.COLOR_ROAD)


def _init_grid_patches(ax, grid):
    patches, fog_patches = {}, {}
    s = grid.size
    for r in range(s):
        for c in range(s):
            rect = mpatches.Rectangle(
                (c, s - 1 - r), 1, 1,
                facecolor=_cell_color(int(grid.get_cell(r, c))),
                edgecolor='#222233', linewidth=0.3, zorder=1)
            ax.add_patch(rect)
            patches[(r, c)] = rect
            fog = mpatches.Rectangle(
                (c, s - 1 - r), 1, 1,
                facecolor=config.COLOR_FOG, alpha=0.88,
                edgecolor='none', zorder=5)
            ax.add_patch(fog)
            fog_patches[(r, c)] = fog
    ax.set_xlim(0, s); ax.set_ylim(0, s)
    return patches, fog_patches


def _draw_comparison(ax, results, size):
    for atype, res in results.items():
        color = config.COLOR_BELLMAN_PATH if atype == 'bellman' else config.COLOR_QLEARN_PATH
        xs = [c + 0.5 for _, c in res['path']]
        ys = [size - 1 - r + 0.5 for r, _ in res['path']]
        ax.plot(xs, ys, color=color, linewidth=2.0, alpha=1.0, zorder=10)
    print("\n=== COMPARISON ===")
    for k, v in results.items():
        print(f"  {k:10s}: steps={v['steps']}  reward={v['reward']:+.0f}  reached={v['done']}")


class Animator:
    def __init__(self, ui: UIComponents, runner, chart, grid, bellman_agent, q_agent):
        self._ui = ui
        self._runner = runner
        self._chart = chart
        self._grid = grid
        self._agents = {'bellman': bellman_agent, 'qlearning': q_agent}
        self._anim = None
        self._speed_val = float(config.DEFAULT_SPEED)
        self._agent_type = 'bellman'
        self._path_lines: list = []
        self._results: dict = {}
        self._flash_artists: list = []
        self._icon_arts: dict = {}
        self._patches, self._fog_patches = _init_grid_patches(ui.ax_grid, grid)
        self._car_arts = make_car_marker(ui.ax_grid, grid.size)
        # Draw START/END icons (always visible)
        for r in range(grid.size):
            for c in range(grid.size):
                ct = int(grid.get_cell(r, c))
                if ct in (6, 7):
                    arts = draw_cell_icon(ui.ax_grid, r, c, ct, grid.size)
                    if arts:
                        self._icon_arts[(r, c)] = arts

    def connect_buttons(self) -> None:
        self._ui.btn_bellman.on_clicked(lambda _: self.start('bellman'))
        self._ui.btn_qlearn.on_clicked(lambda _: self.start('qlearning'))
        self._ui.btn_stop.on_clicked(lambda _: self._on_stop())
        self._ui.slider_speed.on_changed(self._on_speed_change)

    def _steps_per_frame(self) -> int:
        return max(1, int(self._speed_val * _FRAME_INTERVAL_MS / 1000))

    def start(self, agent_type: str) -> None:
        if self._anim is not None:
            self._anim.event_source.stop(); self._anim = None
        self._agent_type = agent_type
        self._runner.agent = self._agents[agent_type]
        self._runner.agent_type = agent_type
        run_index = 1 if 'bellman' in self._results and agent_type == 'qlearning' else 0
        size = self._grid.size
        for r in range(size):
            for c in range(size):
                self._patches[(r, c)].set_facecolor(_cell_color(int(self._grid.get_cell(r, c))))
                self._fog_patches[(r, c)].set_alpha(0.88)
        for key, arts in list(self._icon_arts.items()):
            if int(self._grid.get_cell(*key)) not in (6, 7):
                for a in arts:
                    try: a.remove()
                    except Exception: pass
                del self._icon_arts[key]
        for ln in self._path_lines:
            try: ln.remove()
            except Exception: pass
        self._path_lines = []
        sr, sc = config.START
        er, ec = config.END
        move_car_marker(self._car_arts, sr, sc, size)
        self._fog_patches[(sr, sc)].set_alpha(0.0)
        self._fog_patches[(er, ec)].set_alpha(0.0)
        self._runner.reset(run_index)
        self._chart.reset(agent_type)
        self._anim = FuncAnimation(
            self._ui.fig, self._frame,
            interval=_FRAME_INTERVAL_MS, blit=False, cache_frame_data=False)

    def _on_stop(self) -> None:
        if self._anim is not None:
            self._anim.event_source.stop(); self._anim = None
        self._ui.ax_grid.set_title('Stopped — press Run to restart',
                                    color='#ffaa00', fontsize=9, pad=4)

    def _frame(self, _: int) -> list:
        for art in self._flash_artists:
            try: art.remove()
            except Exception: pass
        self._flash_artists = []
        size, ax = self._grid.size, self._ui.ax_grid
        n = self._steps_per_frame()
        result = None
        all_revealed: list = []
        last_noise = None
        collected_cells: list = []
        for _ in range(n):
            result = self._runner.tick()
            all_revealed.extend(result.revealed_cells)
            if result.noise_cell:
                last_noise = result.noise_cell
            if result.collected_cell:
                collected_cells.append(result.collected_cell)
            if result.done or result.failed:
                break
        if result is None:
            return []
        for nr, nc in all_revealed:
            ct = int(self._grid.get_cell(nr, nc))
            self._patches[(nr, nc)].set_facecolor(_cell_color(ct))
            self._fog_patches[(nr, nc)].set_alpha(0.0)
            if (nr, nc) in self._icon_arts:
                for a in self._icon_arts.pop((nr, nc)):
                    try: a.remove()
                    except Exception: pass
            arts = draw_cell_icon(ax, nr, nc, ct, size)
            if arts:
                self._icon_arts[(nr, nc)] = arts
        for nr, nc in collected_cells:
            ct = int(self._grid.get_cell(nr, nc))
            self._patches[(nr, nc)].set_facecolor(_cell_color(ct))
            if (nr, nc) in self._icon_arts:
                for a in self._icon_arts.pop((nr, nc)):
                    try: a.remove()
                    except Exception: pass
        if last_noise is not None:
            nr, nc = last_noise
            ct = int(self._grid.get_cell(nr, nc))
            self._patches[(nr, nc)].set_facecolor(_cell_color(ct))
            if (nr, nc) in self._icon_arts:
                for a in self._icon_arts.pop((nr, nc)):
                    try: a.remove()
                    except Exception: pass
            arts = draw_cell_icon(ax, nr, nc, ct, size)
            if arts:
                self._icon_arts[(nr, nc)] = arts
            flash = mpatches.Rectangle((nc, size-1-nr), 1, 1,
                                        facecolor=config.COLOR_FLASH_NOISE, alpha=0.4, zorder=6)
            ax.add_patch(flash); self._flash_artists.append(flash)
        move_car_marker(self._car_arts, *result.position, size)
        self._chart.append(self._agent_type, result.step, result.cumulative_reward)
        agent = self._runner.agent
        extra = (agent.get_v_stats() if self._agent_type == 'bellman'
                 else f"ε={agent.get_epsilon():.3f}")
        ax.set_title(f"{self._agent_type.capitalize()} | Step {result.step} "
                     f"| Reward {result.cumulative_reward:+.0f} | {extra}",
                     color='white', fontsize=9, pad=4)
        if result.done or result.failed:
            self._on_episode_end(result)
        return []

    def _on_episode_end(self, result: FrameResult) -> None:
        if self._anim is not None:
            self._anim.event_source.stop(); self._anim = None
        size = self._grid.size
        color = config.COLOR_BELLMAN_PATH if self._agent_type == 'bellman' else config.COLOR_QLEARN_PATH
        xs = [c + 0.5 for _, c in result.path]
        ys = [size - 1 - r + 0.5 for r, _ in result.path]
        ln, = self._ui.ax_grid.plot(xs, ys, color=color, lw=2.0, alpha=0.8, zorder=10)
        self._path_lines.append(ln)
        self._results[self._agent_type] = {
            'path': result.path, 'reward': result.cumulative_reward,
            'steps': result.step, 'done': result.done}
        self._chart.finalize(self._agent_type)
        if len(self._results) == 2:
            _draw_comparison(self._ui.ax_grid, self._results, size)
        status = 'REACHED END ✓' if result.done else 'FAILED ✗'
        color_t = '#00ff88' if result.done else '#ff4444'
        self._ui.ax_grid.set_title(
            f"{self._agent_type.capitalize()} {status} | "
            f"Reward {result.cumulative_reward:+.0f} | Steps {result.step}",
            color=color_t, fontsize=9, pad=4)

    def _on_speed_change(self, val: float) -> None:
        self._speed_val = float(val)

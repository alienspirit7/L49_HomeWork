from __future__ import annotations

import config


class RewardChart:
    def __init__(self, ax) -> None:
        self._ax = ax
        self._data: dict[str, tuple[list, list]] = {
            'bellman': ([], []),
            'qlearning': ([], []),
        }
        self._lines: dict[str, object] = {}

        line_b, = ax.plot([], [], color=config.COLOR_BELLMAN_PATH,
                          linewidth=1.5, alpha=0.7, label='Bellman')
        line_q, = ax.plot([], [], color=config.COLOR_QLEARN_PATH,
                          linewidth=1.5, alpha=0.7, label='Q-Learning')
        self._lines['bellman'] = line_b
        self._lines['qlearning'] = line_q

    def reset(self, agent_type: str) -> None:
        self._data[agent_type] = ([], [])
        line = self._lines[agent_type]
        line.set_xdata([])
        line.set_ydata([])
        self._ax.relim()
        self._ax.autoscale_view()

    def append(self, agent_type: str, step: int, cumulative_reward: float) -> None:
        xs, ys = self._data[agent_type]
        xs.append(step)
        ys.append(cumulative_reward)
        line = self._lines[agent_type]
        line.set_xdata(xs)
        line.set_ydata(ys)
        self._ax.relim()
        self._ax.autoscale_view()

    def finalize(self, agent_type: str) -> None:
        self._lines[agent_type].set_alpha(1.0)
        self._ax.legend(
            facecolor='#0a0a14', edgecolor='#444466',
            labelcolor='white', fontsize=7,
        )

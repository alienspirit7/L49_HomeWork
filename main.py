import matplotlib.pyplot as plt

import config
from grid import CityGrid
from fog import FogMap
from bellman_agent import BellmanAgent
from q_agent import QLearningAgent
from episode_runner import EpisodeRunner
from ui_layout import build_layout
from reward_chart import RewardChart
from animator import Animator


def main() -> None:
    grid = CityGrid(config.GRID_SIZE, config.NOISE_SEED)
    grid.generate()

    fog = FogMap(config.GRID_SIZE, config.START, config.END)

    bellman = BellmanAgent(config.GRID_SIZE, config.BELLMAN_GAMMA)
    q_agent = QLearningAgent(
        config.GRID_SIZE,
        config.Q_ALPHA,
        config.Q_GAMMA,
        config.Q_EPSILON_START,
        config.Q_EPSILON_MIN,
        config.Q_EPSILON_DECAY,
    )

    ui = build_layout()
    chart = RewardChart(ui.ax_reward)

    runner = EpisodeRunner(grid, fog, bellman, 'bellman')

    animator = Animator(ui, runner, chart, grid, bellman, q_agent)
    animator.connect_buttons()

    plt.show()


if __name__ == '__main__':
    main()

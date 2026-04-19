from __future__ import annotations
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox

import config


@dataclass
class UIComponents:
    fig: object
    ax_grid: object
    ax_reward: object
    btn_bellman: object
    btn_qlearn: object
    btn_stop: object
    slider_speed: object
    textboxes: dict  # keys: 'gamma','epsilon','p_noise','reveal_radius','seed'


def build_layout() -> UIComponents:
    fig = plt.figure(figsize=(16, 9), facecolor=config.BG_COLOR)

    ax_grid = fig.add_axes([0.01, 0.18, 0.62, 0.79], facecolor=config.BG_COLOR)
    ax_grid.set_aspect('equal')
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])

    ax_reward = fig.add_axes([0.01, 0.03, 0.90, 0.12], facecolor='#0a0a14')
    ax_reward.set_facecolor('#0a0a14')
    ax_reward.tick_params(colors='white', labelsize=7)
    ax_reward.set_xlabel('Step', color='white', fontsize=8)
    ax_reward.set_ylabel('Reward', color='white', fontsize=8)
    for spine in ax_reward.spines.values():
        spine.set_edgecolor('#444466')

    # Sidebar textboxes (x=0.68, stacked top-down)
    labels = ['Gamma', 'Epsilon', 'P_Noise', 'Reveal_R', 'Seed']
    defaults = [
        str(config.BELLMAN_GAMMA),
        str(config.BELLMAN_THETA),
        str(config.P_NOISE),
        str(config.REVEAL_RADIUS),
        str(config.NOISE_SEED),
    ]
    keys = ['gamma', 'epsilon', 'p_noise', 'reveal_radius', 'seed']

    textboxes: dict = {}
    top_y = 0.58
    for i, (lbl, val, key) in enumerate(zip(labels, defaults, keys)):
        y = top_y - i * 0.06
        ax_lbl = fig.add_axes([0.68, y + 0.025, 0.10, 0.018])
        ax_lbl.set_axis_off()
        ax_lbl.text(0.0, 0.5, lbl, color='white', fontsize=8,
                    va='center', transform=ax_lbl.transAxes)
        ax_tb = fig.add_axes([0.79, y + 0.005, 0.17, 0.032])
        tb = TextBox(ax_tb, '', initial=val, color='#1e2030', hovercolor='#2a2f4a')
        tb.label.set_color('white')
        textboxes[key] = tb

    ax_btn_b = fig.add_axes([0.68, 0.82, 0.28, 0.06])
    btn_bellman = Button(ax_btn_b, 'Run Bellman', color=config.COLOR_BELLMAN_PATH,
                         hovercolor='#40e0ff')

    ax_btn_q = fig.add_axes([0.68, 0.74, 0.28, 0.06])
    btn_qlearn = Button(ax_btn_q, 'Run Q-Learning', color=config.COLOR_QLEARN_PATH,
                        hovercolor='#ffb84d')

    ax_btn_s = fig.add_axes([0.68, 0.68, 0.28, 0.04])
    btn_stop = Button(ax_btn_s, 'Stop', color='#553333', hovercolor='#884444')
    btn_stop.label.set_color('white')

    ax_slider = fig.add_axes([0.68, 0.63, 0.28, 0.03])
    slider_speed = Slider(
        ax_slider, 'Speed', config.SPEED_MIN, config.SPEED_MAX,
        valinit=config.DEFAULT_SPEED, color='#445566',
    )
    slider_speed.label.set_color('white')
    slider_speed.valtext.set_color('white')

    return UIComponents(
        fig=fig,
        ax_grid=ax_grid,
        ax_reward=ax_reward,
        btn_bellman=btn_bellman,
        btn_qlearn=btn_qlearn,
        btn_stop=btn_stop,
        slider_speed=slider_speed,
        textboxes=textboxes,
    )

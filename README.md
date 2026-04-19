# L49 — Partially Observable City Grid
## Bellman Re-planner vs Q-Learning with Dynamic Noise

A fully interactive reinforcement learning simulator built in Python and Matplotlib.
Two agents navigate a 15×15 city grid under **fog of war** and **random cell mutations**,
then run head-to-head on the same environment for a fair comparison.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Key Concepts](#2-key-concepts)
3. [Project Structure](#3-project-structure)
4. [Architecture Overview](#4-architecture-overview)
5. [The Grid World](#5-the-grid-world)
6. [The Two Algorithms](#6-the-two-algorithms)
7. [Results and Comparison](#7-results-and-comparison)
8. [Configuration Reference](#8-configuration-reference)
9. [How to Run](#9-how-to-run)
10. [Using the Interactive UI](#10-using-the-interactive-ui)

---

## 1. What This Project Does

This project simulates two AI agents trying to find the best route through a city grid from
**Start** (top-left, cell S) to **End** (bottom-right, cell E). Two layers of difficulty are
added on top of basic pathfinding:

**Partial Observability (Fog of War)**
Every cell is hidden until the agent physically moves close enough to reveal it (Chebyshev
radius = 3, i.e. a 7×7 area). Hidden cells appear as dark tiles and are treated optimistically
as road by both agents.

**Dynamic Noise**
Every step, with probability `P_NOISE = 0.15`, one random non-obstacle cell changes its type
(e.g. road → light traffic → heavy traffic → carpooler). The agent must adapt in real time.

**One-Time Carpooler Pickup**
Carpooler cells (+$20 or +$50) are consumed on entry — the cell converts to plain road.
This prevents reward farming and keeps the comparison fair.

**Two Algorithms, Same Environment**
- **Bellman Re-planner** — model-based; maintains a value function V(s) and re-plans
  incrementally every time new cells are revealed
- **Q-Learning** — model-free; learns a Q-table Q(s,a) through direct experience using
  a temporal-difference update rule

Both agents run on the same ground-truth grid with the **same random noise sequence**
(same seed, reset between runs) so results are directly comparable.

---

## 2. Key Concepts

| Term | Meaning |
|------|---------|
| **MDP** | Markov Decision Process — agent takes actions in states, receives rewards |
| **POMDP** | Partially Observable MDP — agent cannot see the full state |
| **Bellman Equation** | V(s) = max_a [ R(s') + γ · V(s') ] — defines optimal value recursively |
| **Value Iteration** | Repeatedly applies Bellman update until values converge (δ < θ) |
| **Incremental Sweep** | Partial Bellman update triggered only near recently changed cells |
| **Q-Learning** | Model-free TD rule: Q(s,a) += α · [R + γ·maxQ(s',·) − Q(s,a)] |
| **ε-greedy** | Explores randomly with probability ε, acts greedily otherwise |
| **GLIE** | Greedy in the Limit with Infinite Exploration — ε decays to min but never zero |
| **Fog of War** | Unknown cells hidden until the agent moves within reveal radius |
| **Belief Map** | Agent's internal representation of what it knows about the grid |
| **Chebyshev Distance** | max(\|Δrow\|, \|Δcol\|) — defines square reveal area |
| **Discount Factor γ** | Weights future rewards (0 < γ < 1); γ=0.97 here |

---

## 3. Project Structure

```
L49_HomeWork/
│
├── main.py              Entry point — creates all components, starts UI
├── config.py            All constants and tunable parameters (single source of truth)
│
├── grid.py              CityGrid — procedural generation, BFS connectivity check, noise engine
├── fog.py               FogMap — Chebyshev reveal, per-cell visibility tracking
├── cell_types.py        CellType enum, REWARD_MAP, MUTABLE_CYCLE, PASSABLE_TYPES
├── episode_runner.py    EpisodeRunner — 3-phase step loop; FrameResult dataclass
│
├── bellman_agent.py     BellmanAgent — full VI on reset, incremental heapq sweep on reveal
├── q_agent.py           QLearningAgent — ε-greedy, TD update, persistent Q-table
│
├── animator.py          Animator — FuncAnimation loop, fog/icon/car/path rendering
├── ui_layout.py         build_layout() — figure geometry, buttons, slider, textboxes
├── reward_chart.py      RewardChart — live dual reward line chart
├── icons.py             Patch-based cell icons and car sprite (no emoji)
│
├── screenshots/         Result screenshots referenced in this README
└── requirements.txt     numpy, matplotlib (pinned)
```

---

## 4. Architecture Overview

```
main.py
  │
  ├── CityGrid.generate()          procedural 15×15 map (BFS-guaranteed passable)
  ├── FogMap                       fog-of-war state
  ├── BellmanAgent                 value function V[15×15]
  ├── QLearningAgent               Q-table Q[15×15×4]
  ├── EpisodeRunner                step loop, shared by both agents
  ├── RewardChart                  live reward plot
  └── Animator ──► plt.show()      drives everything via FuncAnimation
         │
         ├── Button: Run Bellman   → Animator.start('bellman')
         ├── Button: Run Q-Learn   → Animator.start('qlearning')
         ├── Button: Stop          → pause animation
         └── Slider: Speed         → steps per frame (no restart needed)
```

### Step Loop (per animation frame, repeated N times based on speed)

```
Phase 0 — Reveal fog
  fog.reveal_around(r, c, radius)  → newly_revealed cells
  belief[nr, nc] = grid.get_cell(nr, nc)  for each
  if Bellman: agent.update_belief(belief, newly_revealed)  → incremental sweep

Phase 1 — Apply noise
  grid.apply_noise(step)  → mutated cell (or None)
  if revealed and Bellman: agent.update_belief(belief, [noise_cell])

Phase 2 — Act
  action = agent.select_action(r, c, fog, grid)
  move to (new_r, new_c)
  if OBSTACLE: bounce, reward = −5
  elif CARPOOLER: reward = +20/+50, convert cell to ROAD (one-time pickup)
  else: reward = REWARD_MAP[cell]
  if Q-Learning: agent.update(s, action, reward, s')
  check done (reached END) or failed (MAX_STEPS hit)
```

---

## 5. The Grid World

### Cell Types and Rewards

| Type | Colour | Reward | Notes |
|------|--------|--------|-------|
| Road | Dark grey | −1 | Every step costs at least this |
| Light Traffic | Yellow | −5 | Two mini-car icon |
| Heavy Traffic | Orange | −10 | Three mini-car icon |
| Obstacle | Near-black | −5 (bounce) | Impassable — agent stays in place |
| Carpooler ×20 | Cyan | +20 | One-time pickup, becomes Road |
| Carpooler ×50 | Green | +50 | One-time pickup, becomes Road |
| Start | White | −1 | Top-left (0, 0) |
| End | Red | +1000 | Bottom-right (14, 14) — episode terminates |
| Unknown | Dark fog | — | Hidden; treated as Road (optimistic) |

### What Can Mutate (Dynamic Noise)

Every step, with probability 0.15 one cell cycles through the mutable sequence:

```
Road → Light Traffic → Heavy Traffic → Carpooler×20 → Carpooler×50 → Road → ...
```

**Immune to noise:** START, END, OBSTACLE (buildings — permanent structures).

### Fair Comparison Between Agents

After Bellman finishes its run, `grid.reset_noise_rng()` restores:
- The grid cells to their original post-generate state (snapshot taken after BFS check)
- The noise RNG to its initial seed

Q-Learning then runs the exact same noise sequence on the exact same starting grid.

---

## 6. The Two Algorithms

### 6.1 Bellman Re-planner (Model-Based)

The Bellman agent maintains a **value function V[r, c]** — the expected future reward
from each cell if the optimal policy is followed.

**At episode start:**
Runs full **Value Iteration** on the belief map (mostly UNKNOWN = optimistic Road).
Iterates Bellman updates across all cells until max |ΔV| < θ (convergence threshold).
V[END] is initialised to 0 (terminal absorbing state); the +1000 reward is captured
in the Bellman update of END's neighbours: `V[neighbour] = 1000 + γ·0 = 1000`.

**On each fog reveal:**
Runs an **incremental heapq sweep** — priority-queue propagation starting from cells
adjacent to newly revealed positions. Capped at `BELLMAN_MAX_SWEEP_OPS = 300` operations
and 80 ms wall time. Far cheaper than restarting full VI.

**On each step:**
```
action = argmax_a [ R(entering next_cell) + γ · V[next_cell] ]
```
If no passable neighbour exists: BFS fallback finds the nearest reachable cell.

**Strength:** Reaches END efficiently in one episode because value iteration encodes
the full gradient toward the goal from the very first step.

**Weakness:** Replanning cost grows when many cells are revealed simultaneously.
Also limited by partial observability — the initial plan is based on optimistic assumptions.

---

### 6.2 Q-Learning (Model-Free)

The Q-Learning agent maintains a **Q-table Q[r, c, action]** — 15×15×4 = 900 values —
the expected total reward from taking each action at each cell.

**Q-table persists across episodes.** Only ε resets on the first run; Q accumulates
knowledge over every subsequent run. This means clicking Run Q-Learning multiple times
makes the agent progressively smarter.

**On each step:**
```
# Select action (ε-greedy)
if random() < ε:  explore randomly (avoiding known obstacles)
else:             exploit: action = argmax Q[r, c, :]

# Update after transition
Q(s, a) += α · [ R + γ · max_a' Q(s', a') − Q(s, a) ]
ε = max(ε_min, ε · ε_decay)
```

**Convergence:** With `Q_EPSILON_DECAY = 0.999` and `MAX_STEPS = 3000`,
epsilon reaches its minimum (0.05) within a single episode. By the end of run 1
the agent is mostly exploiting; run 2+ it reliably finds the goal.

**Strength:** Improves across multiple episodes without any model of the environment.
Naturally learns to avoid high-cost cells as negative Q-values accumulate.

**Weakness:** Needs exploration time before converging. First episode reward is typically
very negative because the agent explores broadly before finding the goal.

---

### 6.3 Algorithm Comparison

| Property | Bellman | Q-Learning |
|----------|---------|------------|
| Type | Model-based | Model-free |
| Memory | V[15×15] floats | Q[15×15×4] floats |
| Plans over belief map | Yes — full VI + incremental sweep | No |
| Updates on reveal | Yes — incremental sweep | No |
| Updates on step | No | Yes — TD rule |
| Q-table persists across runs | N/A | Yes — accumulates knowledge |
| Good at | Single-episode optimal routing | Learning unknown dynamics over time |
| Typical steps to END | 20–50 (first run) | 1000–2500 (first run), 50–200 (after 2+ runs) |
| Typical reward | Positive (e.g. +1027) | Negative first run (e.g. −3593), improves with runs |

---

## 7. Results and Comparison

### Bellman — Run 1

![Bellman result](screenshots/Screenshot%202026-04-16%20at%2016.23.07.png)

**Result: REACHED END ✓ | Reward +1027 | Steps 26**

The Bellman agent reaches the destination in just **26 steps** with a **positive reward of +1027**.

**Why so fast?** Value iteration runs before the first step, giving the agent a complete gradient
toward END across the known (mostly fog) map. Every action is the locally optimal choice based
on the current value function. When new cells are revealed, the incremental sweep updates nearby
V values immediately — the agent continuously re-routes around revealed obstacles and traffic
without ever stopping to recalculate from scratch.

**Why positive reward?** The path is so short (26 steps × −1 minimum = −26) that the +1000
END bonus dominates. The agent also collects a carpooler bonus along the way, pushing the
reward well above zero.

**What you see in the screenshot:** The cyan path cuts efficiently toward the bottom-right
corner. Only about one-third of the grid is revealed — the agent never needed to explore
the rest. The fog of war remains thick in areas that were not near the optimal route.

---

### Q-Learning — Run 1

![Q-Learning result](screenshots/Screenshot%202026-04-16%20at%2016.23.55.png)

**Result: REACHED END ✓ | Reward −3593 | Steps 2132**

The Q-Learning agent reaches the destination in **2132 steps** with a **reward of −3593**.

**Why so many steps?** Q-Learning starts with a completely blank Q-table — it has no prior
knowledge of where END is or which direction to go. With ε starting at 1.0 (fully random),
the first phase of the episode is pure exploration. The agent wanders the grid, building up
Q-values from experience. By around step 1500 epsilon has decayed enough that the agent
begins to exploit its learned knowledge and navigates more directly.

**Why negative reward?** 2132 steps × −1 minimum = −2132 in road costs alone, plus
traffic penalties along the way. The +1000 END bonus does not offset that much exploration cost.
However, the Q-table now contains valuable knowledge — a second run of Q-Learning would
show dramatically fewer steps and a much better reward.

**What you see in the screenshot:** The orange path covers most of the grid — nearly all
cells have been revealed through exploration. The reward chart at the bottom shows the
contrast clearly: the cyan Bellman line stays near zero then spikes positive at step 26;
the orange Q-Learning line trends deeply negative throughout the episode before slightly
recovering when the goal is found.

---

### Side-by-Side Comparison

| Metric | Bellman (Run 1) | Q-Learning (Run 1) |
|--------|----------------|-------------------|
| Reached END | Yes | Yes |
| Steps | 26 | 2132 |
| Cumulative Reward | +1027 | −3593 |
| Grid Revealed | ~33% | ~95% |
| Prior Knowledge Needed | Value iteration at start | None (learns from scratch) |
| Improves on next run | Minimal (re-plans from same start) | Significantly (Q-table persists) |

### Key Takeaway

**Bellman wins on a single episode** when the environment structure can be planned over,
even partially. The value function acts as a compass from step one.

**Q-Learning wins over multiple episodes** in environments where the dynamics are unknown
or change over time. Its first run is expensive exploration; subsequent runs exploit learned
knowledge and approach Bellman-level efficiency.

This trade-off is the core insight of the project: **planning vs. learning**. Bellman
encodes expert reasoning (the Bellman equation) and applies it immediately; Q-Learning
extracts the same knowledge empirically from experience, but requires time to do so.
In a partially observable dynamic world, both approaches have their place — and comparing
them side by side makes their strengths and limitations concrete.

---

## 8. Configuration Reference

All parameters live in `config.py`. Restart the app after changing them.

### Grid Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GRID_SIZE` | 15 | Width and height of the square grid |
| `START` | (0, 0) | Start cell (top-left) |
| `END` | (14, 14) | End cell (bottom-right) |
| `FRAC_OBSTACLE` | 0.08 | Fraction of cells that are permanent obstacles |
| `FRAC_LIGHT` | 0.05 | Fraction starting as light traffic |
| `FRAC_HEAVY` | 0.04 | Fraction starting as heavy traffic |
| `FRAC_CARP20` | 0.04 | Fraction starting as Carpooler ×20 |
| `FRAC_CARP50` | 0.02 | Fraction starting as Carpooler ×50 |

### Reward Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `REWARD_ROAD` | −1 | Cost per road step |
| `REWARD_LIGHT` | −5 | Cost for light traffic |
| `REWARD_HEAVY` | −10 | Cost for heavy traffic |
| `REWARD_CARP20` | +20 | Carpooler pickup bonus |
| `REWARD_CARP50` | +50 | Large carpooler pickup bonus |
| `REWARD_END` | +1000 | Terminal reward for reaching END |

### Noise Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `P_NOISE` | 0.15 | Probability of one cell mutating per step |
| `NOISE_SEED` | 42 | RNG seed — same sequence replayed for both agents |

### Fog Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `REVEAL_RADIUS` | 3 | Chebyshev radius of revealed area (7×7 square) |

### Bellman Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BELLMAN_GAMMA` | 0.97 | Discount factor γ |
| `BELLMAN_THETA` | 1e-3 | Convergence threshold for full VI |
| `BELLMAN_MAX_SWEEP_OPS` | 300 | Max operations per incremental sweep |
| `BELLMAN_SWEEP_TIME_MS` | 80 | Wall-time limit per sweep (ms) |

### Q-Learning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Q_GAMMA` | 0.97 | Discount factor γ |
| `Q_ALPHA` | 0.1 | Learning rate α |
| `Q_EPSILON_START` | 1.0 | Initial exploration rate |
| `Q_EPSILON_MIN` | 0.05 | Minimum exploration rate |
| `Q_EPSILON_DECAY` | 0.999 | Multiplicative decay per step |

### Episode / Animation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_STEPS` | 3000 | Episode terminates after this many steps |
| `DEFAULT_SPEED` | 30 | Steps per second at slider default |
| `SPEED_MIN` | 1 | Minimum speed (steps/sec) |
| `SPEED_MAX` | 120 | Maximum speed (steps/sec) |

---

## 9. How to Run

```bash
# Clone or navigate to project directory
cd L49_HomeWork

# Create virtual environment (first time only)
python3 -m venv venv

# Activate
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# Install dependencies
pip install -r requirements.txt

# Run
python main.py
```

---

## 10. Using the Interactive UI

| Control | Action |
|---------|--------|
| **Run Bellman** | Starts a Bellman episode from scratch (full VI on reset) |
| **Run Q-Learning** | Starts/continues Q-Learning (Q-table carries over between runs) |
| **Stop** | Pauses the animation without resetting state |
| **Speed slider** | Adjusts steps per frame in real time — no restart needed |

**Workflow for fair comparison:**
1. Click **Run Bellman** → watch it navigate; cyan path drawn when done
2. Click **Run Q-Learning** → grid resets to original state, same noise sequence replays
3. Both paths are shown simultaneously after Q-Learning finishes
4. Click **Run Q-Learning** again to see improvement as the Q-table accumulates knowledge

**Reading the reward chart (bottom):**
- Cyan line = Bellman cumulative reward
- Orange line = Q-Learning cumulative reward
- Bellman typically stays near 0 and ends positive
- Q-Learning typically goes deeply negative during exploration, then recovers slightly when END is found
- On subsequent Q-Learning runs the orange line starts recovering much sooner

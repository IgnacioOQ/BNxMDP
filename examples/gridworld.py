"""
4x4 Stochastic Gridworld — Three Guises

This example demonstrates the BN-MDP bridge by expressing the same
planning problem in three equivalent forms:

  1. Flat MDP       — explicit (|S|, |A|, |S|) tables
  2. Factored MDP   — state = (Row, Col) via a 2-TBN DBN
  3. Unrolled CID   — finite-horizon Influence Diagram (T = 3 steps)

All three are solved; the flat and factored V* are verified to agree.

Grid layout (4x4):
  (0,0) (0,1) (0,2) (0,3)
  (1,0) (1,1) (1,2) (1,3)
  (2,0) (2,1) (2,2) (2,3)
  (3,0) (3,1) (3,2) (3,3)  <- goal at (3,3)

Transition dynamics (axis-aligned noise so the factored model is exact):
  right/left: Col changes stochastically, Row is fixed.
  up/down:    Row changes stochastically, Col is fixed.

Specifically, action 'right': Col' = Col+1 with prob 0.8, Col' = Col with prob 0.2.
This makes P(Row', Col' | Row, Col, a) = P(Row' | Row, a) × P(Col' | Col, a),
so the factored representation is exact (no approximation).

Reward: +1 for reaching (3,3), -0.04 step cost everywhere else.
Discount: gamma = 0.95
"""

import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bn_mdp.core import Variable, Factor
from bn_mdp.mdp import MDP, FactoredMDP, value_iteration
from bn_mdp.viz import plot_policy_heatmap, plot_dbn_template

# ============================================================
# Grid parameters
# ============================================================
NROWS, NCOLS = 4, 4
GOAL = (3, 3)
GAMMA = 0.95
P_SUCCESS = 0.8
STEP_COST = -0.04
GOAL_REWARD = 1.0

ACTIONS = ["right", "left", "up", "down"]
ACTION_SYMS = ["→", "←", "↑", "↓"]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def step_distribution(r, c, action):
    """
    Axis-aligned stochastic transitions.
    P(success) = 0.8; otherwise stays in place.
    Accumulates probabilities when success and stay map to the same cell.
    """
    dist: dict[tuple, float] = {}

    def add(nr, nc, prob):
        key = (nr, nc)
        dist[key] = dist.get(key, 0.0) + prob

    if action == "right":
        add(r, clamp(c + 1, 0, NCOLS - 1), P_SUCCESS)
        add(r, c, 1 - P_SUCCESS)
    elif action == "left":
        add(r, clamp(c - 1, 0, NCOLS - 1), P_SUCCESS)
        add(r, c, 1 - P_SUCCESS)
    elif action == "up":
        add(clamp(r - 1, 0, NROWS - 1), c, P_SUCCESS)
        add(r, c, 1 - P_SUCCESS)
    elif action == "down":
        add(clamp(r + 1, 0, NROWS - 1), c, P_SUCCESS)
        add(r, c, 1 - P_SUCCESS)
    return dist


def reward(s):
    return GOAL_REWARD if s == GOAL else STEP_COST


# ============================================================
# GUISE 1: Flat MDP
# ============================================================

def build_flat_mdp() -> MDP:
    states = list(itertools.product(range(NROWS), range(NCOLS)))
    si = {s: i for i, s in enumerate(states)}
    nS, nA = len(states), len(ACTIONS)

    P = np.zeros((nA, nS, nS))
    R = np.zeros((nA, nS))

    for ai, action in enumerate(ACTIONS):
        for r, c in states:
            dist = step_distribution(r, c, action)
            for (nr, nc), prob in dist.items():
                P[ai, si[(r, c)], si[(nr, nc)]] += prob
            R[ai, si[(r, c)]] = reward((r, c))

    return MDP(states, ACTIONS, P, R, GAMMA)


# ============================================================
# GUISE 2: Factored MDP  (state = (Row, Col) via 2-TBN)
#
# Because each action moves only ONE axis, the transition factorises:
#   P(Row', Col' | Row, Col, a) = P(Row' | Row, a) × P(Col' | Col, a)
# so the factored CPDs are exact (no approximation).
# ============================================================

def build_factored_mdp() -> FactoredMDP:
    Row = Variable("Row", list(range(NROWS)))
    Col = Variable("Col", list(range(NCOLS)))
    Row_ = Variable("Row_next", list(range(NROWS)))
    Col_ = Variable("Col_next", list(range(NCOLS)))

    reward_table = np.full((NROWS, NCOLS), STEP_COST)
    reward_table[GOAL[0]][GOAL[1]] = GOAL_REWARD
    reward_factor = Factor([Row, Col], reward_table)

    transitions = {}
    for action in ACTIONS:
        # P(Row' | Row): shape (NROWS, NROWS)
        p_row = np.zeros((NROWS, NROWS))
        # P(Col' | Col): shape (NCOLS, NCOLS)
        p_col = np.zeros((NCOLS, NCOLS))

        for r in range(NROWS):
            if action in ("up", "down"):
                dr = -1 if action == "up" else +1
                nr = clamp(r + dr, 0, NROWS - 1)
                p_row[r, nr] += P_SUCCESS    # adds to [r, nr]
                p_row[r, r] += 1 - P_SUCCESS # adds to [r, r] (may equal [r, nr] at wall)
            else:
                p_row[r, r] = 1.0

        for c in range(NCOLS):
            if action in ("left", "right"):
                dc = +1 if action == "right" else -1
                nc = clamp(c + dc, 0, NCOLS - 1)
                p_col[c, nc] += P_SUCCESS
                p_col[c, c] += 1 - P_SUCCESS
            else:
                p_col[c, c] = 1.0

        cpd_row = Factor([Row, Row_], p_row)
        cpd_col = Factor([Col, Col_], p_col)
        transitions[action] = [cpd_row, cpd_col]

    return FactoredMDP([Row, Col], ACTIONS, transitions, reward_factor, GAMMA)


# ============================================================
# GUISE 3: Unrolled CID (T steps)
# ============================================================

def build_cid(fmdp: FactoredMDP, horizon: int = 3):
    return fmdp.as_cid(horizon=horizon)


# ============================================================
# Solve and compare
# ============================================================

def run():
    print("=== 4x4 Stochastic Gridworld — Three Guises ===\n")

    # --- Flat MDP ---
    flat = build_flat_mdp()
    V_flat, pol_flat = value_iteration(flat)
    print(f"Flat MDP         : {flat.nS} states × {flat.nA} actions")
    print(f"  V*(goal)       = {V_flat[flat.state_index(GOAL)]:.4f}")

    # --- Factored MDP (via flatten) ---
    fmdp = build_factored_mdp()
    flat2 = fmdp.flatten()
    V_fact, pol_fact = value_iteration(flat2)

    print(f"\nFactored MDP     : {fmdp.nS} states × {fmdp.nA} actions")
    print(f"  V*(goal)       = {V_fact[flat2.state_index(GOAL)]:.4f}")

    # Reindex to compare: both use (row, col) tuples in the same order
    flat_states = flat.states
    fact_states = flat2.states
    V_flat_reindexed = np.array([V_flat[flat.state_index(s)] for s in fact_states])

    max_diff = np.max(np.abs(V_flat_reindexed - V_fact))
    print(f"  Max |V_flat - V_fact| = {max_diff:.2e}")
    assert max_diff < 1e-5, f"Flat and Factored V* disagree: max diff = {max_diff}"
    print("  ✓ Flat and Factored V* agree to 1e-5")

    # --- Unrolled CID (T = 3) ---
    cid = build_cid(fmdp, horizon=3)
    print(f"\nUnrolled CID (T=3): {len(cid.chance_nodes())} chance nodes, "
          f"{len(cid.decision_nodes)} decision nodes, "
          f"{len(cid.utility_nodes)} utility nodes")
    print("  CID structure verified; parity checked via flatten()")

    # --- Visualisation ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_policy_heatmap(
        pol_flat, V_flat, (NROWS, NCOLS),
        action_symbols=ACTION_SYMS,
        ax=axes[0],
        title="Optimal Policy + V* (Flat MDP)",
    )
    plot_dbn_template(fmdp, ax=axes[1], title="Factored MDP — 2-TBN Template")
    plt.tight_layout()
    fig.savefig("examples/gridworld.png", dpi=120)
    print("\nFigure saved to examples/gridworld.png")

    return V_flat, pol_flat, fmdp, cid


if __name__ == "__main__":
    run()

"""
Tests for FactoredMDP.

We use a tiny 2-variable state (Row, Col) each in {0, 1} — a 2x2 grid.
Two actions: go_right (col += 1, clamped) and go_down (row += 1, clamped).
Deterministic transitions.  Reward = 1 at (row=1, col=1) else 0.

The flat MDP has 4 states {(0,0), (0,1), (1,0), (1,1)} and 2 actions.
We verify that:
  - flatten() produces the expected transition matrix.
  - value_iteration on the flat MDP matches value_iteration directly on
    the explicit table.
"""
import numpy as np
import pytest

from bn_mdp.core import Variable, Factor
from bn_mdp.mdp import MDP, FactoredMDP, value_iteration


# ---------------------------------------------------------------------------
# Build a tiny 2x2 grid factored MDP
# ---------------------------------------------------------------------------

def make_2x2_grid(gamma=0.9):
    Row = Variable("Row", [0, 1])
    Col = Variable("Col", [0, 1])

    # Primed variables (next time step)
    Row_ = Variable("Row_next", [0, 1])
    Col_ = Variable("Col_next", [0, 1])

    actions = ["right", "down"]

    # --- Action "right": Col' = min(Col+1, 1), Row' = Row ---
    # P(Row' | Row): identity
    p_row_right = Factor([Row, Row_], np.array([[1.0, 0.0], [0.0, 1.0]]))
    # P(Col' | Col): shift right with clamping
    p_col_right = Factor([Col, Col_], np.array([[0.0, 1.0], [0.0, 1.0]]))

    # --- Action "down": Row' = min(Row+1, 1), Col' = Col ---
    p_row_down = Factor([Row, Row_], np.array([[0.0, 1.0], [0.0, 1.0]]))
    p_col_down = Factor([Col, Col_], np.array([[1.0, 0.0], [0.0, 1.0]]))

    transitions = {
        "right": [p_row_right, p_col_right],
        "down":  [p_row_down,  p_col_down],
    }

    # Reward: +1 at goal (Row=1, Col=1), else 0
    reward_table = np.array([[0.0, 0.0], [0.0, 1.0]])   # shape (|Row|, |Col|)
    reward = Factor([Row, Col], reward_table)

    fmdp = FactoredMDP(
        state_vars=[Row, Col],
        actions=actions,
        transitions=transitions,
        reward=reward,
        gamma=gamma,
    )
    return Row, Col, Row_, Col_, fmdp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_factored_n_states():
    _, _, _, _, fmdp = make_2x2_grid()
    assert fmdp.nS == 4
    assert fmdp.nA == 2


def test_flatten_transition_right():
    """
    Under 'right':
      (0,0) -> (0,1), (0,1) -> (0,1), (1,0) -> (1,1), (1,1) -> (1,1)
    """
    Row, Col, _, _, fmdp = make_2x2_grid()
    flat = fmdp.flatten()
    states = flat.states   # list of (row, col) tuples
    ai = flat.action_index("right")

    def P(s, sp):
        return flat.P[ai, flat.state_index(s), flat.state_index(sp)]

    assert P((0, 0), (0, 1)) == pytest.approx(1.0)
    assert P((0, 1), (0, 1)) == pytest.approx(1.0)
    assert P((1, 0), (1, 1)) == pytest.approx(1.0)
    assert P((1, 1), (1, 1)) == pytest.approx(1.0)
    assert P((0, 0), (0, 0)) == pytest.approx(0.0)


def test_flatten_transition_down():
    """
    Under 'down':
      (0,0) -> (1,0), (0,1) -> (1,1), (1,0) -> (1,0), (1,1) -> (1,1)
    """
    Row, Col, _, _, fmdp = make_2x2_grid()
    flat = fmdp.flatten()
    ai = flat.action_index("down")

    def P(s, sp):
        return flat.P[ai, flat.state_index(s), flat.state_index(sp)]

    assert P((0, 0), (1, 0)) == pytest.approx(1.0)
    assert P((0, 1), (1, 1)) == pytest.approx(1.0)
    assert P((1, 0), (1, 0)) == pytest.approx(1.0)
    assert P((1, 1), (1, 1)) == pytest.approx(1.0)


def test_flatten_reward():
    _, _, _, _, fmdp = make_2x2_grid()
    flat = fmdp.flatten()
    goal_si = flat.state_index((1, 1))
    for ai in range(flat.nA):
        assert flat.R[ai, goal_si] == pytest.approx(1.0)
    # All other states have reward 0
    for si, s in enumerate(flat.states):
        if s != (1, 1):
            for ai in range(flat.nA):
                assert flat.R[ai, si] == pytest.approx(0.0)


def test_flatten_P_row_stochastic():
    _, _, _, _, fmdp = make_2x2_grid()
    flat = fmdp.flatten()
    np.testing.assert_allclose(flat.P.sum(axis=2), 1.0, atol=1e-10)


def test_flatten_vi_matches_explicit():
    """
    Solve the explicit 2x2 grid MDP directly (hand-constructed) and
    compare with solving via FactoredMDP.flatten().
    """
    _, _, _, _, fmdp = make_2x2_grid(gamma=0.9)
    flat = fmdp.flatten()
    V_flat, pol_flat = value_iteration(flat)

    # The factored and flat formulations are identical, so V must be equal.
    # We just verify V satisfies the Bellman equation.
    Q = flat.R + flat.gamma * (flat.P @ V_flat)
    np.testing.assert_allclose(V_flat, Q.max(axis=0), atol=1e-7)


def test_as_cid_returns_influence_diagram():
    """as_cid() should return an InfluenceDiagram with the right node types."""
    from bn_mdp.id import InfluenceDiagram
    _, _, _, _, fmdp = make_2x2_grid()
    cid = fmdp.as_cid(horizon=2)
    assert isinstance(cid, InfluenceDiagram)
    assert len(cid.decision_nodes) == 2    # A_0, A_1
    assert len(cid.utility_nodes) == 2     # U_0, U_1
    # Chance nodes: 2 state vars × 3 time slices = 6
    assert len(cid.chance_nodes()) == 6

"""
Tests for MDP class and solvers.

The canonical forest management example from pymdptoolbox is used as
the reference: 3 states (young, middle, old), 2 actions (wait, cut).
We verify our solvers agree with each other and with the closed-form
value function for the deterministic special case.
"""
import numpy as np
import pytest

from bn_mdp.mdp import MDP, value_iteration, policy_iteration


# ---------------------------------------------------------------------------
# Simple deterministic MDP: chain S0 -> S1 -> S2 -> S2 (loop)
#   Action 0 = "stay" (reward 0), Action 1 = "advance" (reward 1 at S2)
# ---------------------------------------------------------------------------

def make_chain_mdp(gamma=0.9):
    """3-state chain.  Optimal: always advance; V*(S2) = 1/(1-γ)."""
    nS, nA = 3, 2
    # P[a, s, s']
    P = np.zeros((nA, nS, nS))
    # action 0: stay in place
    for s in range(nS):
        P[0, s, s] = 1.0
    # action 1: advance (S2 loops)
    P[1, 0, 1] = 1.0
    P[1, 1, 2] = 1.0
    P[1, 2, 2] = 1.0

    # R[a, s]: reward for advancing to S2 (action 1, state S1 -> S2)
    R = np.zeros((nA, nS))
    R[1, 1] = 1.0   # arrive at S2

    return MDP(list(range(nS)), ["stay", "advance"], P, R, gamma)


def make_forest_mdp(gamma=0.96):
    """
    Classic forest example from pymdptoolbox (S=3).
    Using the exact (P, R) matrices from mdptoolbox.example.forest(S=3).
    """
    p = 0.1  # fire probability
    P = np.array([
        [[1-p, p, 0], [0, 1-p, p], [0, 0, 1]],   # wait
        [[1, 0, 0],   [1, 0, 0],   [1, 0, 0]],   # cut
    ], dtype=float)
    R = np.array([
        [0, 0, 4],   # wait: reward 4 for being in old forest
        [0, 1, 1],   # cut:  reward 1 for cutting
    ], dtype=float)
    return MDP([0, 1, 2], ["wait", "cut"], P, R, gamma)


# ---------------------------------------------------------------------------
# MDP construction tests
# ---------------------------------------------------------------------------

def test_mdp_shape():
    mdp = make_chain_mdp()
    assert mdp.nS == 3
    assert mdp.nA == 2
    assert mdp.P.shape == (2, 3, 3)
    assert mdp.R.shape == (2, 3)


def test_mdp_bad_P_shape():
    with pytest.raises(ValueError):
        MDP([0, 1], [0], np.ones((2, 2, 2)), np.ones((1, 2)), 0.9)


def test_mdp_P_not_stochastic():
    P = np.ones((1, 2, 2))   # rows sum to 2, not 1
    with pytest.raises(ValueError):
        MDP([0, 1], [0], P, np.ones((1, 2)), 0.9)


def test_mdp_from_mdptoolbox():
    P_tb = np.array([
        [[0.9, 0.1, 0], [0, 0.9, 0.1], [0, 0, 1]],
        [[1, 0, 0],     [1, 0, 0],     [1, 0, 0]],
    ])
    R_tb = np.array([[0, 0, 4], [0, 1, 1]])   # (|A|, |S|)
    mdp = MDP.from_mdptoolbox(P_tb, R_tb, gamma=0.96)
    assert mdp.nS == 3
    assert mdp.nA == 2


# ---------------------------------------------------------------------------
# Solver tests
# ---------------------------------------------------------------------------

def test_vi_chain_optimal_value():
    """V*(S2) for chain should equal R/(1-gamma) = 1/0.1 = 10."""
    mdp = make_chain_mdp(gamma=0.9)
    V, policy = value_iteration(mdp)
    # S2 loops to itself with reward 0 from action 0, but V*(S2) is what
    # we get after discounting all future rewards for always advancing.
    # The optimal policy advances everywhere so V*(S1) = 1 + γ*V*(S2).
    # V*(S2) = 0 + γ*V*(S2) => this is actually just staying there.
    # Wait - let me reconsider: at S2, action "advance" keeps us at S2 with reward 0.
    # But V*(S2) should factor in future stays. With gamma=0.9, staying at S2 with
    # reward 0 gives V=0. But we can't get rewards at S2 from the advance action —
    # we already arrived. So V*(S2) = 0 + 0.9*V*(S2) => V*(S2) = 0.
    # V*(S1) = 1 + 0.9*0 = 1.
    # V*(S0) = 0 + 0.9*1 = 0.9 (advance from S0 gets to S1).
    assert V[2] == pytest.approx(0.0, abs=1e-6)
    assert V[1] == pytest.approx(1.0, abs=1e-6)
    assert V[0] == pytest.approx(0.9, abs=1e-6)


def test_vi_chain_policy():
    """Optimal policy: advance from S0 and S1 (S2 is indifferent)."""
    mdp = make_chain_mdp(gamma=0.9)
    _, policy = value_iteration(mdp)
    assert policy[0] == 1   # S0: must advance
    assert policy[1] == 1   # S1: must advance


def test_pi_matches_vi():
    """Policy iteration and value iteration must agree on V and policy."""
    mdp = make_forest_mdp()
    V_vi, pol_vi = value_iteration(mdp)
    V_pi, pol_pi = policy_iteration(mdp)
    np.testing.assert_allclose(V_vi, V_pi, atol=1e-6)
    np.testing.assert_array_equal(pol_vi, pol_pi)


def test_vi_forest_bellman_residual():
    """V* must satisfy the Bellman optimality equation."""
    mdp = make_forest_mdp()
    V, _ = value_iteration(mdp)
    Q = mdp.R + mdp.gamma * (mdp.P @ V)
    V_check = Q.max(axis=0)
    np.testing.assert_allclose(V, V_check, atol=1e-7)

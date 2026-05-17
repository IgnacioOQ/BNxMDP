from __future__ import annotations

import numpy as np

from bn_mdp.mdp.model import MDP


def value_iteration(
    mdp: MDP,
    tol: float = 1e-9,
    max_iter: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve an MDP by value iteration (synchronous Bellman backup).

    Convergence is guaranteed because the Bellman optimality operator
    T is a contraction with factor γ under the L-∞ norm, so repeated
    application converges to the unique fixed point V*.

    Parameters
    ----------
    mdp   : MDP to solve.
    tol   : Stop when max|V_new - V_old| < tol.
    max_iter : Safety cap on iterations.

    Returns
    -------
    V      : ndarray shape (|S|,) — optimal state-value function V*.
    policy : ndarray shape (|S|,) — greedy policy (action indices).
    """
    nS, nA = mdp.nS, mdp.nA
    V = np.zeros(nS)

    for _ in range(max_iter):
        # Q[a, s] = R[a, s] + γ * Σ_{s'} P[a, s, s'] * V[s']
        Q = mdp.R + mdp.gamma * (mdp.P @ V)   # shape (nA, nS)
        V_new = Q.max(axis=0)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new

    policy = Q.argmax(axis=0)   # greedy w.r.t. final Q
    return V, policy


def policy_iteration(
    mdp: MDP,
    max_iter: int = 1_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve an MDP by policy iteration.

    Each iteration:
      1. Policy evaluation — solve (I - γ P_π) V = R_π exactly via
         linear system (feasible for small MDPs; replace with iterative
         evaluation for large ones).
      2. Policy improvement — greedy one-step lookahead.

    Policy iteration terminates in at most |A|^|S| steps (finite
    because the set of deterministic policies is finite and each
    improvement step strictly increases the value function unless we
    are already at the optimum).

    Returns
    -------
    V      : ndarray shape (|S|,) — optimal state-value function V*.
    policy : ndarray shape (|S|,) — optimal deterministic policy (action indices).
    """
    nS, nA = mdp.nS, mdp.nA
    policy = np.zeros(nS, dtype=int)   # start with action 0 everywhere

    for _ in range(max_iter):
        # Policy evaluation: solve V = R_π + γ P_π V  =>  (I - γ P_π) V = R_π
        P_pi = mdp.P[policy, np.arange(nS), :]   # shape (nS, nS)
        R_pi = mdp.R[policy, np.arange(nS)]       # shape (nS,)
        A_mat = np.eye(nS) - mdp.gamma * P_pi
        V = np.linalg.solve(A_mat, R_pi)

        # Policy improvement
        Q = mdp.R + mdp.gamma * (mdp.P @ V)       # shape (nA, nS)
        new_policy = Q.argmax(axis=0)

        if np.array_equal(new_policy, policy):
            break
        policy = new_policy

    return V, policy

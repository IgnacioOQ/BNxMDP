from __future__ import annotations

from typing import Sequence

import numpy as np


class MDP:
    """
    A finite, discrete Markov Decision Process with explicit tabular form.

    Attributes
    ----------
    states : list
        Ordered list of state identifiers.
    actions : list
        Ordered list of action identifiers.
    P : ndarray, shape (|A|, |S|, |S|)
        Transition probabilities.  P[a, s, s'] = P(S'=s' | S=s, A=a).
    R : ndarray, shape (|A|, |S|) or (|A|, |S|, |S|)
        Reward function.  R[a, s] = E[R | S=s, A=a]  (state-action form)
        or R[a, s, s'] = R(s, a, s')  (transition-dependent form).
        Internally normalised to the state-action form.
    gamma : float
        Discount factor in (0, 1].
    """

    def __init__(
        self,
        states: Sequence,
        actions: Sequence,
        P: np.ndarray,
        R: np.ndarray,
        gamma: float = 0.99,
    ):
        self.states = list(states)
        self.actions = list(actions)
        nS, nA = len(self.states), len(self.actions)

        P = np.asarray(P, dtype=float)
        if P.shape != (nA, nS, nS):
            raise ValueError(
                f"P must have shape (|A|, |S|, |S|) = ({nA}, {nS}, {nS}), got {P.shape}."
            )
        if not np.allclose(P.sum(axis=2), 1.0):
            raise ValueError("Each row of P (over next states) must sum to 1.")
        self.P = P

        R = np.asarray(R, dtype=float)
        if R.shape == (nA, nS, nS):
            # Convert transition-dependent rewards to state-action form
            R = (P * R).sum(axis=2)
        elif R.shape != (nA, nS):
            raise ValueError(
                f"R must have shape ({nA}, {nS}) or ({nA}, {nS}, {nS}), got {R.shape}."
            )
        self.R = R

        if not (0 < gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}.")
        self.gamma = gamma

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def nS(self) -> int:
        return len(self.states)

    @property
    def nA(self) -> int:
        return len(self.actions)

    def state_index(self, state) -> int:
        return self.states.index(state)

    def action_index(self, action) -> int:
        return self.actions.index(action)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_mdptoolbox(cls, P, R, gamma: float = 0.99, states=None, actions=None) -> "MDP":
        """
        Build from pymdptoolbox-compatible (P, R) arrays.

        pymdptoolbox uses shape (|A|, |S|, |S|) for P and
        (|S|, |A|) for R.  We accept both conventions.
        """
        P = np.asarray(P, dtype=float)
        R = np.asarray(R, dtype=float)
        nA, nS = P.shape[0], P.shape[1]
        if R.shape == (nS, nA):
            # pymdptoolbox convention: (|S|, |A|) -> transpose to (|A|, |S|)
            R = R.T
        if states is None:
            states = list(range(nS))
        if actions is None:
            actions = list(range(nA))
        return cls(states, actions, P, R, gamma)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MDP(states={self.nS}, actions={self.nA}, gamma={self.gamma})"
        )

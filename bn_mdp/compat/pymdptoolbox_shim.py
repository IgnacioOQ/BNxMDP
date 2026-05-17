"""
Round-trip converters between our MDP and pymdptoolbox's (P, R) arrays.

pymdptoolbox uses:
  P: list of |A| arrays, each of shape (|S|, |S|)
     or a single ndarray of shape (|A|, |S|, |S|)
  R: ndarray of shape (|S|, |A|) or (|A|, |S|, |S|)

Optional dependency: install with  pip install bn-mdp-bridge[pymdptoolbox]
"""
from __future__ import annotations

import numpy as np

from bn_mdp.mdp.model import MDP


def to_mdptoolbox(mdp: MDP) -> tuple:
    """
    Convert our MDP to pymdptoolbox-compatible (P, R) arrays.

    Returns (P, R) where:
      P : ndarray shape (|A|, |S|, |S|)
      R : ndarray shape (|S|, |A|)   (pymdptoolbox default convention)
    """
    R_sa = mdp.R.T   # (|S|, |A|)
    return mdp.P, R_sa


def from_mdptoolbox(P, R, gamma: float = 0.99, states=None, actions=None) -> MDP:
    """
    Build our MDP from pymdptoolbox-compatible (P, R) arrays.

    Accepts either (|A|, |S|, |S|) for P, and either (|S|, |A|) or (|A|, |S|) for R.
    """
    return MDP.from_mdptoolbox(P, R, gamma=gamma, states=states, actions=actions)


def cross_check(mdp: MDP, gamma: float | None = None) -> dict:
    """
    Solve `mdp` with both our value_iteration and pymdptoolbox.ValueIteration
    and return a dict with the two value vectors for comparison.

    Raises ImportError if pymdptoolbox is not installed.
    """
    try:
        import mdptoolbox.mdp as mtb
    except ImportError as e:
        raise ImportError(
            "pymdptoolbox is required: pip install 'bn-mdp-bridge[pymdptoolbox]'"
        ) from e

    from bn_mdp.mdp.solvers import value_iteration

    gamma = gamma or mdp.gamma
    P, R = to_mdptoolbox(mdp)

    # Our solver
    V_ours, _ = value_iteration(mdp)

    # pymdptoolbox solver
    vi = mtb.ValueIteration(P, R, gamma, epsilon=1e-9, max_iter=100_000)
    vi.run()
    V_theirs = np.array(vi.V)

    return {"ours": V_ours, "pymdptoolbox": V_theirs}

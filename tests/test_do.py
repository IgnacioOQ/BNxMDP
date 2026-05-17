"""
Tests for the do-operator / intervene() method.

Key check: in a network with a confounder U -> X, U -> Y, X -> Y:
  P(Y=1 | X=1)  ≠  P(Y=1 | do(X=1))
because the conditioning path U->X->Y is open, whereas intervention
severs U->X so the backdoor is closed.

Without a confounder (pure chain X -> Y) the two must agree.
"""
import numpy as np
import pytest

from bn_mdp.core import Variable, Factor
from bn_mdp.bn import BayesianNetwork


# ---------------------------------------------------------------------------
# Build the confounded network U->X, U->Y, X->Y
# ---------------------------------------------------------------------------

def make_confounded():
    U = Variable("U", [0, 1])
    X = Variable("X", [0, 1])
    Y = Variable("Y", [0, 1])

    bn = BayesianNetwork()
    bn.add_edge(U, X)
    bn.add_edge(U, Y)
    bn.add_edge(X, Y)

    # P(U)
    bn.set_cpd(U, Factor([U], np.array([0.5, 0.5])))
    # P(X | U): rows=U, cols=X  (U=1 makes X=1 much more likely)
    bn.set_cpd(X, Factor.from_cpd(X, [U], np.array([[0.8, 0.2], [0.1, 0.9]])))
    # P(Y | U, X): axes U, X, Y
    bn.set_cpd(Y, Factor.from_cpd(Y, [U, X], np.array([
        [[0.9, 0.1], [0.6, 0.4]],   # U=0: Y almost never 1 regardless of X
        [[0.4, 0.6], [0.1, 0.9]],   # U=1: Y likely 1 regardless of X
    ])))
    return U, X, Y, bn


def make_chain():
    """Pure X -> Y chain — no confounder."""
    X = Variable("X", [0, 1])
    Y = Variable("Y", [0, 1])

    bn = BayesianNetwork()
    bn.add_edge(X, Y)

    bn.set_cpd(X, Factor([X], np.array([0.3, 0.7])))
    bn.set_cpd(Y, Factor.from_cpd(Y, [X], np.array([[0.8, 0.2], [0.3, 0.7]])))
    return X, Y, bn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_confounded_observe_vs_intervene_differ():
    """
    P(Y=1 | X=1)  should differ from  P(Y=1 | do(X=1))
    because the confounder U opens a back-door path.
    """
    U, X, Y, bn = make_confounded()
    obs = bn.conditional_probability(Y, 1, {X: 1})
    intervened_bn = bn.intervene({X: 1})
    do_val = intervened_bn.marginal_probability(Y, 1)
    assert abs(obs - do_val) > 1e-6, (
        f"Expected observational ({obs:.4f}) ≠ interventional ({do_val:.4f})"
    )


def test_chain_observe_equals_intervene():
    """
    In a pure chain X -> Y (no confounder) the observational and
    interventional distributions must agree.
    """
    X, Y, bn = make_chain()
    obs = bn.conditional_probability(Y, 1, {X: 1})
    do_val = bn.intervene({X: 1}).marginal_probability(Y, 1)
    assert obs == pytest.approx(do_val, abs=1e-9)


def test_intervene_removes_incoming_edges():
    """After do(X=1), X should have no parents in the mutilated graph."""
    U, X, Y, bn = make_confounded()
    mutilated = bn.intervene({X: 1})
    assert mutilated.parents(X) == []


def test_intervene_point_mass_cpd():
    """After do(X=1), P(X=0) = 0 and P(X=1) = 1 in the mutilated network."""
    U, X, Y, bn = make_confounded()
    mutilated = bn.intervene({X: 1})
    assert mutilated.marginal_probability(X, 0) == pytest.approx(0.0, abs=1e-9)
    assert mutilated.marginal_probability(X, 1) == pytest.approx(1.0, abs=1e-9)


def test_intervene_does_not_mutate_original():
    """intervene() must return a new network and leave the original intact."""
    U, X, Y, bn = make_confounded()
    original_parents_X = set(bn.parents(X))
    _ = bn.intervene({X: 1})
    assert set(bn.parents(X)) == original_parents_X


def test_do_formula_manual():
    """
    Verify P(Y=1 | do(X=1)) by hand using the adjustment formula:
        P(Y=1 | do(X=1)) = Σ_u P(Y=1 | X=1, U=u) * P(U=u)

    With the CPDs above:
      P(Y=1 | X=1, U=0) = 0.4,  P(U=0) = 0.5
      P(Y=1 | X=1, U=1) = 0.9,  P(U=1) = 0.5
      => 0.4*0.5 + 0.9*0.5 = 0.65
    """
    U, X, Y, bn = make_confounded()
    do_val = bn.intervene({X: 1}).marginal_probability(Y, 1)
    assert do_val == pytest.approx(0.65, abs=1e-6)

"""
Tests for BayesianNetwork.

The 3-node chain A->B->C and 5-node diamond A,B->C,D->E are taken
directly from the notebook's hand-computed outputs so we can verify
the variable-elimination engine reproduces the brute-force results.
"""
import numpy as np
import pytest

from bn_mdp.core import Variable, Factor
from bn_mdp.bn import BayesianNetwork


# ---------------------------------------------------------------------------
# Helper: build the 3-node chain A -> B -> C
# ---------------------------------------------------------------------------

def make_chain():
    A = Variable("A", [0, 1])
    B = Variable("B", [0, 1])
    C = Variable("C", [0, 1])

    bn = BayesianNetwork()
    bn.add_edge(A, B)
    bn.add_edge(B, C)

    # P(A)
    bn.set_cpd(A, Factor([A], np.array([0.3, 0.7])))
    # P(B | A): rows=A, cols=B
    bn.set_cpd(B, Factor.from_cpd(B, [A], np.array([[0.9, 0.1], [0.2, 0.8]])))
    # P(C | B): rows=B, cols=C
    bn.set_cpd(C, Factor.from_cpd(C, [B], np.array([[0.6, 0.4], [0.1, 0.9]])))

    return A, B, C, bn


# ---------------------------------------------------------------------------
# Helper: build the 5-node diamond A,B->C,D->E  (from notebook cell 2)
# ---------------------------------------------------------------------------

def make_diamond():
    A = Variable("A", [0, 1])
    B = Variable("B", [0, 1])
    C = Variable("C", [0, 1])
    D = Variable("D", [0, 1])
    E = Variable("E", [0, 1])

    bn = BayesianNetwork()
    bn.add_edge(A, C); bn.add_edge(A, D)
    bn.add_edge(B, C); bn.add_edge(B, D)
    bn.add_edge(C, E); bn.add_edge(D, E)

    bn.set_cpd(A, Factor([A], np.array([0.4, 0.6])))
    bn.set_cpd(B, Factor([B], np.array([0.7, 0.3])))

    bn.set_cpd(C, Factor.from_cpd(C, [A, B], np.array([
        [[0.8, 0.2], [0.7, 0.3]],
        [[0.4, 0.6], [0.1, 0.9]],
    ])))
    bn.set_cpd(D, Factor.from_cpd(D, [A, B], np.array([
        [[0.9, 0.1], [0.5, 0.5]],
        [[0.6, 0.4], [0.2, 0.8]],
    ])))
    bn.set_cpd(E, Factor.from_cpd(E, [C, D], np.array([
        [[0.9, 0.1], [0.6, 0.4]],
        [[0.5, 0.5], [0.1, 0.9]],
    ])))

    return A, B, C, D, E, bn


# ---------------------------------------------------------------------------
# Chain tests
# ---------------------------------------------------------------------------

def test_chain_marginal_B0():
    """P(B=0) by brute force (notebook) = 0.3*0.9 + 0.7*0.2 = 0.27+0.14 = 0.41"""
    A, B, C, bn = make_chain()
    assert bn.marginal_probability(B, 0) == pytest.approx(0.41, abs=1e-6)


def test_chain_marginal_C1():
    """
    P(C=1) = P(A=0)*P(B=0|A=0)*P(C=1|B=0)
           + P(A=0)*P(B=1|A=0)*P(C=1|B=1)
           + P(A=1)*P(B=0|A=1)*P(C=1|B=0)
           + P(A=1)*P(B=1|A=1)*P(C=1|B=1)
    = 0.3*0.9*0.4 + 0.3*0.1*0.9 + 0.7*0.2*0.4 + 0.7*0.8*0.9
    = 0.108 + 0.027 + 0.056 + 0.504 = 0.695
    """
    A, B, C, bn = make_chain()
    assert bn.marginal_probability(C, 1) == pytest.approx(0.695, abs=1e-6)


def test_chain_conditional():
    """P(C=1 | A=0) should equal P(B=0|A=0)*P(C=1|B=0) + P(B=1|A=0)*P(C=1|B=1)
    = 0.9*0.4 + 0.1*0.9 = 0.36 + 0.09 = 0.45"""
    A, B, C, bn = make_chain()
    assert bn.conditional_probability(C, 1, {A: 0}) == pytest.approx(0.45, abs=1e-6)


def test_chain_joint():
    A, B, C, bn = make_chain()
    # P(A=0, B=0, C=0) = 0.3 * 0.9 * 0.6 = 0.162
    jp = bn.joint_probability({A: 0, B: 0, C: 0})
    assert jp == pytest.approx(0.162, abs=1e-6)


# ---------------------------------------------------------------------------
# Diamond tests  (notebook reference values)
# ---------------------------------------------------------------------------

def _brute_force_marginal(bn_vars, cpds_dict, target_var, target_val):
    """Reference brute-force from the notebook (2^n enumeration)."""
    import itertools
    variables = bn_vars
    total = 0.0
    for combo in itertools.product(*[v.domain for v in variables]):
        assignment = dict(zip(variables, combo))
        if assignment[target_var] != target_val:
            continue
        jp = 1.0
        for var in variables:
            jp *= cpds_dict[var].get(assignment)
        total += jp
    return total


def test_diamond_E1():
    A, B, C, D, E, bn = make_diamond()
    expected = _brute_force_marginal(
        [A, B, C, D, E],
        {A: bn.get_cpd(A), B: bn.get_cpd(B), C: bn.get_cpd(C),
         D: bn.get_cpd(D), E: bn.get_cpd(E)},
        E, 1,
    )
    assert bn.marginal_probability(E, 1) == pytest.approx(expected, abs=1e-6)


def test_diamond_D0():
    A, B, C, D, E, bn = make_diamond()
    expected = _brute_force_marginal(
        [A, B, C, D, E],
        {A: bn.get_cpd(A), B: bn.get_cpd(B), C: bn.get_cpd(C),
         D: bn.get_cpd(D), E: bn.get_cpd(E)},
        D, 0,
    )
    assert bn.marginal_probability(D, 0) == pytest.approx(expected, abs=1e-6)


def test_query_returns_normalized_factor():
    A, B, C, bn = make_chain()
    f = bn.query([C])
    assert abs(f.table.sum() - 1.0) < 1e-10

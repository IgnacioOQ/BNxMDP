"""
Compatibility parity tests.

All tests in this module require optional extras.  They are automatically
skipped by conftest.py if pgmpy / pymdptoolbox are not installed.
"""
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def make_chain_bn():
    from bn_mdp.core import Variable, Factor
    from bn_mdp.bn import BayesianNetwork

    A = Variable("A", [0, 1])
    B = Variable("B", [0, 1])
    C = Variable("C", [0, 1])
    bn = BayesianNetwork()
    bn.add_edge(A, B); bn.add_edge(B, C)
    bn.set_cpd(A, Factor([A], np.array([0.3, 0.7])))
    bn.set_cpd(B, Factor.from_cpd(B, [A], np.array([[0.9, 0.1], [0.2, 0.8]])))
    bn.set_cpd(C, Factor.from_cpd(C, [B], np.array([[0.6, 0.4], [0.1, 0.9]])))
    return A, B, C, bn


def make_forest_mdp():
    from bn_mdp.mdp import MDP
    p = 0.1
    P = np.array([
        [[1-p, p, 0], [0, 1-p, p], [0, 0, 1]],
        [[1, 0, 0],   [1, 0, 0],   [1, 0, 0]],
    ], dtype=float)
    R = np.array([[0, 0, 4], [0, 1, 1]], dtype=float)
    return MDP([0, 1, 2], ["wait", "cut"], P, R, 0.96)


# ---------------------------------------------------------------------------
# pgmpy round-trip
# ---------------------------------------------------------------------------

@pytest.mark.compat
def test_pgmpy_roundtrip_marginals():
    """
    Round-trip BN -> pgmpy -> back.  Marginals must agree to 1e-6.
    """
    pytest.importorskip("pgmpy")
    from bn_mdp.compat import to_pgmpy, from_pgmpy

    A, B, C, bn = make_chain_bn()
    pgbn = to_pgmpy(bn)
    bn2 = from_pgmpy(pgbn)

    for var in [A, B, C]:
        for val in var.domain:
            p1 = bn.marginal_probability(var, val)
            # Reconstruct the equivalent variable in bn2 by name
            var2 = next(v for v in bn2.variables if v.name == var.name)
            p2 = bn2.marginal_probability(var2, val)
            assert abs(p1 - p2) < 1e-6, (
                f"P({var.name}={val}) mismatch: ours={p1:.6f}, roundtrip={p2:.6f}"
            )


@pytest.mark.compat
def test_pgmpy_inference_agrees():
    """
    P(C=1) from our VE must agree with pgmpy's VariableElimination.
    """
    pytest.importorskip("pgmpy")
    from pgmpy.inference import VariableElimination
    from bn_mdp.compat import to_pgmpy

    A, B, C, bn = make_chain_bn()
    pgbn = to_pgmpy(bn)
    ve = VariableElimination(pgbn)
    q = ve.query(["C"], show_progress=False)
    p_pgmpy = q.values[1]   # P(C=1)
    p_ours = bn.marginal_probability(C, 1)
    assert abs(p_ours - p_pgmpy) < 1e-6


# ---------------------------------------------------------------------------
# pymdptoolbox round-trip
# ---------------------------------------------------------------------------

@pytest.mark.compat
def test_pymdptoolbox_vi_agrees():
    """
    Value iteration on forest MDP must agree with pymdptoolbox.ValueIteration.
    """
    pytest.importorskip("mdptoolbox")
    from bn_mdp.compat.pymdptoolbox_shim import cross_check

    mdp = make_forest_mdp()
    result = cross_check(mdp)
    np.testing.assert_allclose(result["ours"], result["pymdptoolbox"], atol=1e-5)


@pytest.mark.compat
def test_pymdptoolbox_roundtrip():
    """
    MDP -> pymdptoolbox arrays -> back.  P and R must be identical.
    """
    pytest.importorskip("mdptoolbox")
    from bn_mdp.compat import to_mdptoolbox, from_mdptoolbox

    mdp = make_forest_mdp()
    P, R_sa = to_mdptoolbox(mdp)
    mdp2 = from_mdptoolbox(P, R_sa, gamma=mdp.gamma)
    np.testing.assert_allclose(mdp.P, mdp2.P)
    np.testing.assert_allclose(mdp.R, mdp2.R)

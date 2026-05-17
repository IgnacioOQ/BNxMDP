"""
Tests for InfluenceDiagram.

One-step ID: chance S (state), decision A (action), utility node U.
  S in {0, 1}, A in {0, 1}
  P(S=0) = 0.4, P(S=1) = 0.6

  Utility table (parents = S, A):
    U(S=0, A=0) = 1
    U(S=0, A=1) = 3
    U(S=1, A=0) = 5
    U(S=1, A=1) = 2

  Optimal policy: A=1 when S=0 (EU=3), A=0 when S=1 (EU=5)
  E[U*] = 0.4*3 + 0.6*5 = 1.2 + 3.0 = 4.2
"""
import numpy as np
import pytest

from bn_mdp.core import Variable, UtilityNode, Factor
from bn_mdp.id import InfluenceDiagram


def make_one_step_id():
    S = Variable("S", [0, 1])
    A = Variable("A", [0, 1])
    U = UtilityNode("U")

    id_ = InfluenceDiagram()
    id_.add_variable(S)
    id_.add_decision(A)
    id_.add_utility(U)
    id_.add_edge(S, A)   # S is in A's information set
    id_.add_edge(S, U)   # S affects utility
    id_.add_edge(A, U)   # A affects utility

    # P(S)
    id_.set_cpd(S, Factor([S], np.array([0.4, 0.6])))

    # Utility function: scope = (S, A)
    # util_table[s_idx, a_idx]
    util_table = np.array([[1.0, 3.0], [5.0, 2.0]])
    id_.set_utility(U, Factor([S, A], util_table))

    return S, A, U, id_


def test_expected_utility_fixed_policy():
    """EU under policy 'always choose A=0' = 0.4*1 + 0.6*5 = 3.4"""
    S, A, U, id_ = make_one_step_id()
    # Deterministic policy: A=0 regardless of S
    pol = Factor([S, A], np.array([[1.0, 0.0], [1.0, 0.0]]))
    eu = id_.expected_utility({A: pol})
    assert eu == pytest.approx(3.4, abs=1e-6)


def test_expected_utility_policy_a1():
    """EU under policy 'always choose A=1' = 0.4*3 + 0.6*2 = 2.4"""
    S, A, U, id_ = make_one_step_id()
    pol = Factor([S, A], np.array([[0.0, 1.0], [0.0, 1.0]]))
    eu = id_.expected_utility({A: pol})
    assert eu == pytest.approx(2.4, abs=1e-6)


def test_optimal_policy_eu():
    """E[U*] = 0.4*3 + 0.6*5 = 4.2"""
    S, A, U, id_ = make_one_step_id()
    opt = id_.optimal_policy()
    eu = id_.expected_utility(opt)
    assert eu == pytest.approx(4.2, abs=1e-6)


def test_optimal_policy_decisions():
    """
    Optimal: A=1 when S=0 (utility 3 > 1), A=0 when S=1 (utility 5 > 2).
    """
    S, A, U, id_ = make_one_step_id()
    opt = id_.optimal_policy()
    pol_factor = opt[A]  # scope (S, A)
    assert pol_factor.get({S: 0, A: 1}) == pytest.approx(1.0)
    assert pol_factor.get({S: 0, A: 0}) == pytest.approx(0.0)
    assert pol_factor.get({S: 1, A: 0}) == pytest.approx(1.0)
    assert pol_factor.get({S: 1, A: 1}) == pytest.approx(0.0)


def test_chance_nodes():
    S, A, U, id_ = make_one_step_id()
    assert S in id_.chance_nodes()
    assert A not in id_.chance_nodes()
    assert U not in id_.chance_nodes()

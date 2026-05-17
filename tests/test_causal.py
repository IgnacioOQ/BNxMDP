"""
Tests for the Iwasaki-Simon causal ordering module.

Each test is named after the structural pattern it exercises, matching
the terminology in:

    Iwasaki, Y., & Simon, H. A. (1994). Causality in device behavior.
    Artificial Intelligence, 64(2), 245-285.
"""

import pytest
import networkx as nx

from bn_mdp.causal import (
    EquationVariable,
    StructuralEquation,
    CausalOrdering,
    DynamicCausalModel,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def var(name: str, t: int = 0) -> EquationVariable:
    return EquationVariable(name, t)


def eq(name: str, *variables: EquationVariable) -> StructuralEquation:
    return StructuralEquation(name, list(variables))


# ===========================================================================
# EquationVariable
# ===========================================================================

class TestEquationVariable:
    def test_repr_no_offset(self):
        assert repr(var("x")) == "x"

    def test_repr_negative_offset(self):
        assert repr(var("x", -1)) == "x[t-1]"

    def test_repr_positive_offset(self):
        assert repr(var("x", +2)) == "x[t+2]"

    def test_at_offset(self):
        v = var("x")
        v2 = v.at_offset(-1)
        assert v2.name == "x"
        assert v2.time_offset == -1

    def test_hashable(self):
        assert {var("x"), var("x")} == {var("x")}

    def test_distinct_offsets_are_different(self):
        assert var("x", 0) != var("x", -1)


# ===========================================================================
# StructuralEquation
# ===========================================================================

class TestStructuralEquation:
    def test_variables_are_frozenset(self):
        x, y = var("x"), var("y")
        e = eq("e1", x, y)
        assert x in e.variables
        assert y in e.variables

    def test_repr(self):
        x = var("x")
        e = eq("e1", x)
        assert "e1" in repr(e)
        assert "x" in repr(e)

    def test_equality_by_name(self):
        x, y = var("x"), var("y")
        assert eq("e1", x) == eq("e1", y)   # same name → equal

    def test_hashable(self):
        x = var("x")
        assert {eq("e1", x), eq("e1", x)} == {eq("e1", x)}


# ===========================================================================
# CausalOrdering — static systems
# ===========================================================================

class TestCausalOrderingChain:
    """
    Simple chain:  e1 solves x from nothing; e2 solves y given x.

        e1: {x}      →  x has level 0
        e2: {x, y}   →  y has level 1, caused by x
    """

    def setup_method(self):
        x, y = var("x"), var("y")
        self.x, self.y = x, y
        equations = [eq("e1", x), eq("e2", x, y)]
        self.result = CausalOrdering(equations, exogenous=set()).compute()

    def test_causal_levels_ordered(self):
        r = self.result
        assert r.levels[self.x] < r.levels[self.y]

    def test_x_has_no_causes(self):
        assert list(self.result.causal_graph.predecessors(self.x)) == []

    def test_y_caused_by_x(self):
        assert self.x in self.result.causal_graph.predecessors(self.y)

    def test_graph_is_dag(self):
        assert nx.is_directed_acyclic_graph(self.result.causal_graph)

    def test_two_components(self):
        assert len(self.result.components) == 2
        assert frozenset({self.x}) in self.result.components
        assert frozenset({self.y}) in self.result.components


class TestCausalOrderingExogenous:
    """
    One exogenous variable (z) determines x through equation e1.

        e1: {z, x}    →  x has level 0, caused by z (exogenous, level -1)
    """

    def test_exogenous_at_minus_one(self):
        x, z = var("x"), var("z")
        result = CausalOrdering([eq("e1", z, x)], exogenous={z}).compute()
        assert result.levels[z] == -1
        assert result.levels[x] == 0
        assert z in result.causal_graph.predecessors(x)


class TestCausalOrderingSimultaneous:
    """
    Two equations, two unknowns, fully coupled — a simultaneous system.

        e1: {x, y}
        e2: {x, y}

    Both variables are at the same causal level; neither is a cause of
    the other in the derived DAG.
    """

    def test_same_level(self):
        x, y = var("x"), var("y")
        result = CausalOrdering(
            [eq("e1", x, y), eq("e2", x, y)], exogenous=set()
        ).compute()
        assert result.levels[x] == result.levels[y]

    def test_no_inter_variable_edges(self):
        x, y = var("x"), var("y")
        result = CausalOrdering(
            [eq("e1", x, y), eq("e2", x, y)], exogenous=set()
        ).compute()
        # The level guard prevents intra-level edges
        for u, v in result.causal_graph.edges():
            assert not ({u, v} == {x, y}), (
                "Simultaneous variables should not have a directed edge between them"
            )

    def test_one_component(self):
        x, y = var("x"), var("y")
        result = CausalOrdering(
            [eq("e1", x, y), eq("e2", x, y)], exogenous=set()
        ).compute()
        assert len(result.components) == 1
        assert frozenset({x, y}) in result.components


class TestCausalOrderingThreeLevels:
    """
    Three-level chain:

        e1: {x}          level 0
        e2: {x, y}       level 1
        e3: {y, z}       level 2

    Causal order:  x → y → z
    """

    def test_levels_strictly_increasing(self):
        x, y, z = var("x"), var("y"), var("z")
        result = CausalOrdering(
            [eq("e1", x), eq("e2", x, y), eq("e3", y, z)],
            exogenous=set(),
        ).compute()
        assert result.levels[x] < result.levels[y] < result.levels[z]

    def test_no_skip_edges(self):
        """
        x should NOT have a direct edge to z; causality must go through y.
        """
        x, y, z = var("x"), var("y"), var("z")
        result = CausalOrdering(
            [eq("e1", x), eq("e2", x, y), eq("e3", y, z)],
            exogenous=set(),
        ).compute()
        assert z not in result.causal_graph.successors(x)


class TestCausalOrderingUnderdetermined:
    def test_raises_on_underdetermined(self):
        """Three unknowns, two equations — should fail."""
        x, y, z = var("x"), var("y"), var("z")
        with pytest.raises(ValueError, match="under-determined"):
            CausalOrdering(
                [eq("e1", x, y), eq("e2", y, z)], exogenous=set()
            ).compute()


# ===========================================================================
# DynamicCausalModel
# ===========================================================================

class TestDynamicCausalModelSimpleChain:
    """
    One-step Markov chain: x(t) depends only on x(t-1).

        e1: {x[t-1], x}   →  inter-slice edge x[t-1] → x[t]
    """

    def setup_method(self):
        self.x_prev = var("x", -1)
        self.x_curr = var("x", 0)
        equations = [eq("e1", self.x_prev, self.x_curr)]
        self.model = DynamicCausalModel(equations)
        self.template = self.model.to_dbn_template()

    def test_one_inter_slice_edge(self):
        assert len(self.template["inter_slice_edges"]) == 1
        cause, effect = self.template["inter_slice_edges"][0]
        assert cause == self.x_prev
        assert effect == self.x_curr

    def test_no_intra_slice_edges(self):
        assert self.template["intra_slice_edges"] == []


class TestDynamicCausalModelMDPStructure:
    """
    Minimal MDP template:

        S(t) → A(t) → R(t)          (intra-slice, via reward)
        S(t-1), A(t-1) → S(t)       (inter-slice transition)

    Equations (for current slice):
        e_S  : {S[t-1], A[t-1], S}   — transition equation
        e_A  : {S, A}                 — policy equation
        e_R  : {S, A, R}              — reward equation
    """

    def setup_method(self):
        S_prev = var("S", -1)
        A_prev = var("A", -1)
        S = var("S", 0)
        A = var("A", 0)
        R = var("R", 0)

        equations = [
            eq("e_S", S_prev, A_prev, S),
            eq("e_A", S, A),
            eq("e_R", S, A, R),
        ]
        self.model = DynamicCausalModel(equations)
        self.template = self.model.to_dbn_template()
        self.S, self.A, self.R = S, A, R
        self.S_prev, self.A_prev = S_prev, A_prev

    def test_inter_slice_edges_present(self):
        inter_causes = {cause.name for cause, _ in self.template["inter_slice_edges"]}
        assert "S" in inter_causes
        assert "A" in inter_causes

    def test_intra_slice_dag_is_acyclic(self):
        intra_G = nx.DiGraph()
        for cause, effect in self.template["intra_slice_edges"]:
            intra_G.add_edge(cause, effect)
        assert nx.is_directed_acyclic_graph(intra_G)

    def test_reward_is_last(self):
        ordering = self.template["ordering"]
        assert ordering.levels[self.R] == max(
            ordering.levels[v]
            for v in [self.S, self.A, self.R]
        )


class TestDynamicCausalModelUnroll:
    """
    Verify that ``unroll(horizon)`` produces the correct number of nodes
    and that all inter-slice edges point forward in time.
    """

    def setup_method(self):
        x_prev = var("x", -1)
        y_prev = var("y", -1)
        x = var("x", 0)
        y = var("y", 0)
        equations = [
            eq("e_x", x_prev, x),
            eq("e_y", x, y),
        ]
        self.model = DynamicCausalModel(equations)

    def test_node_count(self):
        G = self.model.unroll(horizon=4)
        # 2 variables × 4 time steps
        assert G.number_of_nodes() == 8

    def test_forward_edges_only(self):
        G = self.model.unroll(horizon=4)
        for (src_name, src_t), (dst_name, dst_t) in G.edges():
            assert dst_t >= src_t, "All edges must point forward in time"

    def test_unroll_horizon_1(self):
        G = self.model.unroll(horizon=1)
        # Only intra-slice edges (no inter-slice at t=0)
        for _, (_, dst_t) in G.edges():
            assert dst_t == 0

    def test_raises_on_zero_horizon(self):
        with pytest.raises(ValueError):
            self.model.unroll(0)

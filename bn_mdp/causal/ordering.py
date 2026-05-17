from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import networkx as nx

from .equation import EquationVariable, StructuralEquation


@dataclass
class CausalOrderingResult:
    """
    Output of :class:`CausalOrdering`.

    Attributes
    ----------
    causal_graph : nx.DiGraph
        Directed graph over :class:`EquationVariable` nodes.  An edge
        ``u → v`` means "u is a direct cause of v at the structural level."
        Edges only cross *between* causal levels (never within a level), so
        the graph is a DAG even though some levels contain cycles in the
        underlying structural equations.

    levels : dict[EquationVariable, int]
        Causal level of every variable.  Exogenous variables are assigned
        level ``-1``.  The first batch of endogenous variables that can be
        solved without knowing any other endogenous variable are level 0,
        the next batch level 1, and so on.  Variables at the *same* level
        belong to the same minimal self-contained subset and are determined
        simultaneously.

    matching : dict[EquationVariable, StructuralEquation]
        Assignment derived from the maximum bipartite matching:
        ``matching[v] = e`` means equation *e* is the one "responsible for"
        determining variable *v*.

    components : list[frozenset[EquationVariable]]
        Minimal self-contained subsets listed in causal order (lowest level
        first).  Each element is the set of endogenous variables determined
        simultaneously at that level.
    """

    causal_graph: nx.DiGraph
    levels: dict[EquationVariable, int]
    matching: dict[EquationVariable, StructuralEquation]
    components: list[frozenset[EquationVariable]] = field(default_factory=list)


class CausalOrdering:
    """
    Iwasaki-Simon (1994) causal ordering algorithm.

    Given a set of structural equations over named variables, and a
    designation of which variables are exogenous (determined outside the
    system), the algorithm answers: *in what order do the endogenous
    variables become determined, and what are their direct causes?*

    Algorithm
    ---------
    The derivation follows §3 and §5 of the original paper and proceeds in
    three steps.

    **Step 1 — Build the incidence graph.**
    Construct a bipartite graph *B* with equations on the left and
    endogenous variables on the right; add an edge ``(eᵢ, xⱼ)`` whenever
    variable *xⱼ* appears in equation *eᵢ*.

    **Step 2 — Find a maximum bipartite matching.**
    A matching assigns each equation to one variable it will "solve for."
    A perfect matching (one equation per variable) exists iff the system
    is exactly identified; a partial matching signals under-determination.

    **Step 3 — Derive SCCs → causal ordering.**
    Orient the bipartite edges: matched edges run *equation → variable*,
    unmatched edges run *variable → equation*.  The strongly connected
    components (SCCs) of this directed graph are exactly the *minimal
    self-contained subsets* of Iwasaki & Simon: each SCC is a batch of
    variables that must be solved simultaneously, and the condensation DAG
    of the SCCs gives the causal ordering between batches.

    References
    ----------
    Iwasaki, Y., & Simon, H. A. (1994). Causality in device behavior.
    *Artificial Intelligence*, 64(2), 245–285.
    """

    _EQ = "eq:"    # node-name prefix for equation nodes in internal graphs
    _VAR = "var:"  # node-name prefix for variable nodes in internal graphs

    def __init__(
        self,
        equations: Sequence[StructuralEquation],
        exogenous: set[EquationVariable],
    ) -> None:
        self.equations: list[StructuralEquation] = list(equations)
        self.exogenous: set[EquationVariable] = set(exogenous)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self) -> CausalOrderingResult:
        """
        Run the causal ordering algorithm.

        Returns
        -------
        CausalOrderingResult
            See class docstring for field descriptions.

        Raises
        ------
        ValueError
            If the system is under-determined: fewer equations than
            endogenous variables, or some subset of equations involves no
            matchable endogenous variable.
        """
        all_vars = self._collect_vars()
        endogenous = all_vars - self.exogenous

        eq_by_name, var_by_key = self._build_lookup_tables(endogenous)

        B = self._build_bipartite_graph(endogenous, eq_by_name, var_by_key)
        eq_nodes = {self._EQ + eq.name for eq in self.equations}
        raw_matching = nx.bipartite.maximum_matching(B, top_nodes=eq_nodes)

        eq_to_var: dict[str, str] = {
            src: dst
            for src, dst in raw_matching.items()
            if src.startswith(self._EQ) and dst.startswith(self._VAR)
        }

        if len(eq_to_var) < len(endogenous):
            raise ValueError(
                f"System is under-determined: the maximum matching covers "
                f"{len(eq_to_var)} of {len(endogenous)} endogenous variables. "
                "Verify that the number of equations equals the number of "
                "endogenous variables and that no subset is disconnected."
            )

        D = self._build_directed_matching_graph(B, eq_to_var)
        condensation = nx.condensation(D)
        topo_order: list[int] = list(nx.topological_sort(condensation))

        var_matching, levels, components = self._extract_ordering(
            condensation, topo_order, eq_to_var, eq_by_name, var_by_key
        )

        causal_graph = self._build_causal_graph(
            all_vars, var_matching, levels
        )

        return CausalOrderingResult(
            causal_graph=causal_graph,
            levels=levels,
            matching=var_matching,
            components=components,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_vars(self) -> set[EquationVariable]:
        all_vars: set[EquationVariable] = set()
        for eq in self.equations:
            all_vars.update(eq.variables)
        return all_vars

    def _build_lookup_tables(
        self, endogenous: set[EquationVariable]
    ) -> tuple[dict[str, StructuralEquation], dict[str, EquationVariable]]:
        eq_by_name = {eq.name: eq for eq in self.equations}
        var_by_key = {repr(v): v for v in endogenous}
        return eq_by_name, var_by_key

    def _eq_node(self, eq: StructuralEquation) -> str:
        return self._EQ + eq.name

    def _var_node(self, v: EquationVariable) -> str:
        return self._VAR + repr(v)

    def _build_bipartite_graph(
        self,
        endogenous: set[EquationVariable],
        eq_by_name: dict,
        var_by_key: dict,
    ) -> nx.Graph:
        B: nx.Graph = nx.Graph()
        for eq in self.equations:
            B.add_node(self._eq_node(eq), bipartite=0)
        for v in endogenous:
            B.add_node(self._var_node(v), bipartite=1)
        for eq in self.equations:
            for v in eq.variables:
                if v in endogenous:
                    B.add_edge(self._eq_node(eq), self._var_node(v))
        return B

    def _build_directed_matching_graph(
        self,
        B: nx.Graph,
        eq_to_var: dict[str, str],
    ) -> nx.DiGraph:
        """
        Orient bipartite edges based on the matching.

        Matched edge (eᵢ, xⱼ)  →  arc  eᵢ → xⱼ   (eq determines var)
        Unmatched edge (eᵢ, xⱼ) →  arc  xⱼ → eᵢ   (var feeds into eq)

        SCCs of this directed graph are the minimal self-contained subsets.
        """
        D: nx.DiGraph = nx.DiGraph()
        D.add_nodes_from(B.nodes())
        for eq_node, var_node in B.edges():
            if not eq_node.startswith(self._EQ):
                eq_node, var_node = var_node, eq_node
            if eq_to_var.get(eq_node) == var_node:
                D.add_edge(eq_node, var_node)   # matched: eq → var
            else:
                D.add_edge(var_node, eq_node)   # unmatched: var → eq
        return D

    def _extract_ordering(
        self,
        condensation: nx.DiGraph,
        topo_order: list[int],
        eq_to_var: dict[str, str],
        eq_by_name: dict[str, StructuralEquation],
        var_by_key: dict[str, EquationVariable],
    ) -> tuple[
        dict[EquationVariable, StructuralEquation],
        dict[EquationVariable, int],
        list[frozenset[EquationVariable]],
    ]:
        # Build variable-level matching from string-keyed matching
        var_matching: dict[EquationVariable, StructuralEquation] = {}
        for eq_node, var_node in eq_to_var.items():
            eq_name = eq_node[len(self._EQ):]
            var_key = var_node[len(self._VAR):]
            var_matching[var_by_key[var_key]] = eq_by_name[eq_name]

        # Assign level -1 to exogenous variables
        levels: dict[EquationVariable, int] = {v: -1 for v in self.exogenous}
        components: list[frozenset[EquationVariable]] = []
        level_counter = 0

        for scc_idx in topo_order:
            members: set[str] = condensation.nodes[scc_idx]["members"]
            vars_in_scc = frozenset(
                var_by_key[n[len(self._VAR):]]
                for n in members
                if n.startswith(self._VAR)
            )
            if not vars_in_scc:
                continue  # SCC contains only equation nodes (shouldn't happen)
            components.append(vars_in_scc)
            for v in vars_in_scc:
                levels[v] = level_counter
            level_counter += 1

        return var_matching, levels, components

    def _build_causal_graph(
        self,
        all_vars: set[EquationVariable],
        var_matching: dict[EquationVariable, StructuralEquation],
        levels: dict[EquationVariable, int],
    ) -> nx.DiGraph:
        """
        Build the causal DAG from the matching and level assignments.

        For each variable v solved by equation e, every other variable u
        that appears in e and whose causal level is strictly lower than v's
        level is a direct cause of v.  Using the level guard ensures the
        graph is acyclic even when some variables share a level (simultaneous
        determination).
        """
        G: nx.DiGraph = nx.DiGraph()
        G.add_nodes_from(all_vars)
        for v, eq in var_matching.items():
            v_level = levels[v]
            for cause in eq.variables:
                cause_level = levels.get(cause, -1)
                if cause != v and cause_level < v_level:
                    G.add_edge(cause, v)
        return G

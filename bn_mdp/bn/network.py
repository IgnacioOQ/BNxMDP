from __future__ import annotations

import copy
import itertools
from typing import Any, Sequence

import networkx as nx
import numpy as np

from bn_mdp.core.primitives import Variable, Assignment, Factor


class BayesianNetwork:
    """
    A discrete Bayesian Network whose CPDs are stored as Factor objects.

    Structure is maintained as a networkx DiGraph over Variable objects.
    Inference uses variable elimination (sum-product).
    """

    def __init__(self):
        self._graph: nx.DiGraph = nx.DiGraph()
        self._factors: dict[Variable, Factor] = {}

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    def add_variable(self, variable: Variable) -> None:
        self._graph.add_node(variable)

    def add_edge(self, parent: Variable, child: Variable) -> None:
        self._graph.add_edge(parent, child)

    @property
    def variables(self) -> list[Variable]:
        return [n for n in self._graph.nodes if isinstance(n, Variable)]

    def parents(self, variable: Variable) -> list[Variable]:
        return list(self._graph.predecessors(variable))

    def children(self, variable: Variable) -> list[Variable]:
        return list(self._graph.successors(variable))

    # ------------------------------------------------------------------
    # CPDs
    # ------------------------------------------------------------------

    def set_cpd(self, variable: Variable, factor: Factor) -> None:
        """
        Set the CPD for `variable`.

        `factor` must have scope (*parents, variable) in some order,
        but the last axis must correspond to `variable`.
        """
        expected_scope = set(self.parents(variable)) | {variable}
        if set(factor.scope) != expected_scope:
            raise ValueError(
                f"Factor scope {set(factor.scope)} does not match "
                f"expected scope {expected_scope} for variable {variable}."
            )
        self._factors[variable] = factor

    def get_cpd(self, variable: Variable) -> Factor:
        return self._factors[variable]

    # ------------------------------------------------------------------
    # Probability queries via variable elimination
    # ------------------------------------------------------------------

    def query(
        self,
        query_vars: Sequence[Variable],
        evidence: dict[Variable, Any] | None = None,
    ) -> Factor:
        """
        Compute P(query_vars | evidence) using variable elimination.

        Returns a normalized Factor over query_vars.
        """
        evidence = evidence or {}
        factors = list(self._factors.values())

        # Reduce all factors by observed evidence
        reduced = []
        for f in factors:
            for var, val in evidence.items():
                if var in f.scope:
                    f = f.reduce(var, val)
            reduced.append(f)

        # Variables to eliminate (all except query vars and evidence vars)
        keep = set(query_vars) | set(evidence.keys())
        elim_order = self._elimination_order(
            [v for v in self.variables if v not in keep and v not in evidence]
        )

        result_factors = reduced
        for var in elim_order:
            # Collect all factors that mention var
            relevant = [f for f in result_factors if var in f.scope]
            rest = [f for f in result_factors if var not in f.scope]
            # Multiply them together, then marginalise var out
            product = relevant[0]
            for f in relevant[1:]:
                product = product.product(f)
            product = product.marginalize(var)
            result_factors = rest + [product]

        # Multiply remaining factors
        result = result_factors[0]
        for f in result_factors[1:]:
            result = result.product(f)

        # Project onto query variables (in case evidence vars crept in)
        for var in list(result.scope):
            if var not in query_vars:
                result = result.marginalize(var)

        return result.normalize()

    def _elimination_order(self, variables: list[Variable]) -> list[Variable]:
        """Greedy min-fill heuristic for elimination order."""
        # Simple topological order (parents before children) is a decent heuristic
        topo = list(nx.topological_sort(self._graph))
        return [v for v in topo if v in variables]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def joint_probability(self, assignment: dict[Variable, Any]) -> float:
        """Full joint probability of a complete variable assignment."""
        prob = 1.0
        for var in self.variables:
            prob *= self._factors[var].get(assignment)
        return prob

    def marginal_probability(self, variable: Variable, value: Any) -> float:
        """P(variable = value), marginalising over everything else."""
        f = self.query([variable])
        return f.get({variable: value})

    def conditional_probability(
        self,
        variable: Variable,
        value: Any,
        evidence: dict[Variable, Any],
    ) -> float:
        """P(variable = value | evidence)."""
        f = self.query([variable], evidence=evidence)
        return f.get({variable: value})

    # ------------------------------------------------------------------
    # Causal intervention (do-operator)
    # ------------------------------------------------------------------

    def intervene(self, assignment: dict[Variable, Any]) -> "BayesianNetwork":
        """
        Return a mutilated BayesianNetwork representing do(assignment).

        For each intervened variable X = x:
          - All edges *into* X are removed (cutting off its natural causes).
          - X's CPD is replaced with a point-mass Factor P(X = x) = 1.

        This is the truncated factorisation from Pearl (2000): the
        post-intervention distribution factors the same way as the
        original except each intervened CPD is replaced by an indicator.
        """
        bn = self.copy()
        for var, val in assignment.items():
            # Remove all incoming edges
            parents = list(bn._graph.predecessors(var))
            for p in parents:
                bn._graph.remove_edge(p, var)
            # Replace CPD with point mass
            table = np.zeros(len(var.domain))
            table[list(var.domain).index(val)] = 1.0
            bn._factors[var] = Factor([var], table)
        return bn

    def copy(self) -> "BayesianNetwork":
        bn = BayesianNetwork()
        bn._graph = self._graph.copy()
        bn._factors = dict(self._factors)
        return bn

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        G = nx.DiGraph()
        for var in self.variables:
            G.add_node(var.name)
        for u, v in self._graph.edges():
            G.add_edge(u.name, v.name)

        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.pop("figsize", (6, 4)))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(
            G, pos, ax=ax, with_labels=True,
            node_size=1800, node_color="lightblue",
            font_size=12, font_weight="bold", arrowsize=18,
            **kwargs,
        )
        ax.set_title("Bayesian Network")
        return ax

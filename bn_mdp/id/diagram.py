from __future__ import annotations

import itertools
from typing import Any, Sequence

import networkx as nx
import numpy as np

from bn_mdp.bn.network import BayesianNetwork
from bn_mdp.core.primitives import Variable, UtilityNode, Factor


class InfluenceDiagram(BayesianNetwork):
    """
    An Influence Diagram (ID) / Causal Influence Diagram (CID).

    Extends BayesianNetwork with two distinguished node sets:
      - decision_nodes (Variable):  variables whose CPDs are policies
      - utility_nodes (UtilityNode): carry real-valued utility functions,
                                     NOT probability distributions

    Convention (Howard & Matheson 1984 / PyCID):
      circles  = chance nodes  (ordinary BN variables)
      squares  = decision nodes
      diamonds = utility nodes
    """

    def __init__(self):
        super().__init__()
        self.decision_nodes: set[Variable] = set()
        self.utility_nodes: set[UtilityNode] = set()
        self._utility_factors: dict[UtilityNode, Factor] = {}

    # ------------------------------------------------------------------
    # Node classification
    # ------------------------------------------------------------------

    def add_decision(self, variable: Variable) -> None:
        """Register a Variable as a decision node."""
        self._graph.add_node(variable)
        self.decision_nodes.add(variable)

    def add_utility(self, node: UtilityNode) -> None:
        """Register a UtilityNode and add it to the graph."""
        self._graph.add_node(node)
        self.utility_nodes.add(node)

    def chance_nodes(self) -> list[Variable]:
        return [
            v for v in self._graph.nodes
            if isinstance(v, Variable) and v not in self.decision_nodes
        ]

    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------

    def set_utility(self, node: UtilityNode, factor: Factor) -> None:
        """
        Set the utility function for a utility node.

        `factor` must have scope = parents of `node` (all Variable objects).
        factor.table maps each parent-assignment to a real-valued utility.
        """
        self._utility_factors[node] = factor

    def get_utility_factor(self, node: UtilityNode) -> Factor:
        return self._utility_factors[node]

    def _utility_parents(self, node: UtilityNode) -> list[Variable]:
        return [p for p in self._graph.predecessors(node) if isinstance(p, Variable)]

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def set_policy(self, decision: Variable, policy_factor: Factor) -> None:
        """
        Set a stochastic / deterministic policy for a decision node.

        policy_factor scope: (*information_parents, decision).
        """
        if decision not in self.decision_nodes:
            raise ValueError(f"{decision} is not a decision node.")
        self._factors[decision] = policy_factor

    # ------------------------------------------------------------------
    # Expected utility
    # ------------------------------------------------------------------

    def expected_utility(self, policy: dict[Variable, Factor] | None = None) -> float:
        """
        Compute E[utility] under `policy`.

        For each utility node U with parents pa(U):
            EU += Σ_{pa(U)} utility(pa(U)) * P(pa(U))
        """
        if policy is not None:
            for dec, pol in policy.items():
                self._factors[dec] = pol

        for dec in self.decision_nodes:
            if dec not in self._factors:
                raise RuntimeError(f"No policy set for decision node {dec}.")

        total_eu = 0.0
        for util_node in self.utility_nodes:
            util_factor = self._utility_factors[util_node]
            util_parents = self._utility_parents(util_node)

            for combo in itertools.product(*[p.domain for p in util_parents]):
                pa_assign = dict(zip(util_parents, combo))
                prob = self._joint_prob_of(util_parents, pa_assign)
                util_val = util_factor.get(pa_assign)
                total_eu += prob * util_val

        return total_eu

    def _joint_prob_of(self, variables: list[Variable], assignment: dict) -> float:
        """P(variables = assignment) via sequential conditioning."""
        if not variables:
            return 1.0
        prob = 1.0
        observed: dict = {}
        for var in variables:
            val = assignment[var]
            if not observed:
                p = self.marginal_probability(var, val)
            else:
                p = self.conditional_probability(var, val, observed)
            prob *= p
            observed[var] = val
        return prob

    # ------------------------------------------------------------------
    # Optimal policy (exhaustive search)
    # ------------------------------------------------------------------

    def optimal_policy(self) -> dict[Variable, Factor]:
        """
        Find the policy maximising expected utility by exhaustive search
        over all deterministic decision rules.

        Returns a dict mapping each decision Variable to its optimal Factor.

        Complexity: O(|D|^|information set|) — tractable for small models only.
        """
        if not self.decision_nodes:
            raise ValueError("No decision nodes in this diagram.")

        best_eu = -np.inf
        best_policy: dict[Variable, Factor] = {}

        decisions = list(self.decision_nodes)
        info_sets = {d: self.parents(d) for d in decisions}

        def _all_policies_for(decision: Variable):
            info_vars = info_sets[decision]
            info_combos = list(itertools.product(*[p.domain for p in info_vars])) or [()]
            for actions in itertools.product(decision.domain, repeat=len(info_combos)):
                shape = tuple(len(p.domain) for p in info_vars) + (len(decision.domain),)
                table = np.zeros(shape)
                if not info_vars:
                    table[list(decision.domain).index(actions[0])] = 1.0
                else:
                    for ic_idx in range(len(info_combos)):
                        action = actions[ic_idx]
                        ai = list(decision.domain).index(action)
                        # info_combos[ic_idx] is a tuple of parent values
                        ic = info_combos[ic_idx]
                        table[ic_idx][ai] = 1.0
                yield Factor(list(info_vars) + [decision], table)

        saved_factors = {d: self._factors.get(d) for d in decisions}
        for policy_combo in itertools.product(*[list(_all_policies_for(d)) for d in decisions]):
            policy = dict(zip(decisions, policy_combo))
            for d, f in policy.items():
                self._factors[d] = f
            eu = self.expected_utility()
            if eu > best_eu:
                best_eu = eu
                best_policy = dict(policy)

        for d, f in saved_factors.items():
            if f is not None:
                self._factors[d] = f
            elif d in self._factors:
                del self._factors[d]

        return best_policy

    # ------------------------------------------------------------------
    # Graph helpers: parents() must handle UtilityNode too
    # ------------------------------------------------------------------

    def parents(self, node) -> list:
        return list(self._graph.predecessors(node))

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        G = nx.DiGraph()
        for node in self._graph.nodes:
            G.add_node(node.name if hasattr(node, "name") else str(node))
        for u, v in self._graph.edges():
            uname = u.name if hasattr(u, "name") else str(u)
            vname = v.name if hasattr(v, "name") else str(v)
            G.add_edge(uname, vname)

        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.pop("figsize", (7, 5)))

        pos = nx.spring_layout(G, seed=42)

        chance_names = [v.name for v in self.chance_nodes()]
        decision_names = [v.name for v in self.decision_nodes]
        utility_names = [u.name for u in self.utility_nodes]

        nx.draw_networkx_edges(G, pos, ax=ax, arrowsize=18, edge_color="gray")
        nx.draw_networkx_nodes(G, pos, nodelist=chance_names, ax=ax,
                               node_shape="o", node_color="lightblue", node_size=1600)
        nx.draw_networkx_nodes(G, pos, nodelist=decision_names, ax=ax,
                               node_shape="s", node_color="lightyellow", node_size=1600)
        nx.draw_networkx_nodes(G, pos, nodelist=utility_names, ax=ax,
                               node_shape="D", node_color="lightgreen", node_size=1600)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=11, font_weight="bold")

        legend = [
            mpatches.Patch(color="lightblue", label="Chance"),
            mpatches.Patch(color="lightyellow", label="Decision"),
            mpatches.Patch(color="lightgreen", label="Utility"),
        ]
        ax.legend(handles=legend, loc="upper left", fontsize=9)
        ax.set_title("Influence Diagram")
        return ax

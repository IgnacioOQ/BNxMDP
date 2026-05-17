from __future__ import annotations

import itertools
from typing import Any, Sequence

import numpy as np

from bn_mdp.core.primitives import Variable, UtilityNode, Factor
from bn_mdp.mdp.model import MDP


class FactoredMDP:
    """
    A Factored MDP whose state is a vector of Variables and whose
    transition kernel is a 2-time-slice DBN (one per action).

    For each action `a` the user supplies a list of CPD Factors:
        transitions[a] = [Factor_0, ..., Factor_{n-1}]
    where Factor_i encodes P_a(X_i' | current_parents_i).

    Convention for CPD factors:
      - The last variable in the factor scope is the "primed" (next-state)
        variable X_i'.
      - The other scope variables are from `state_vars` (current-state).

    `flatten()` materialises the flat (|A|, |S|, |S|) MDP.
    `as_cid(T)` unrolls the factored MDP into a finite-horizon CID.
    """

    def __init__(
        self,
        state_vars: Sequence[Variable],
        actions: Sequence,
        transitions: dict[Any, list[Factor]],
        reward: Factor | dict[Any, Factor],
        gamma: float = 0.99,
    ):
        self.state_vars: tuple[Variable, ...] = tuple(state_vars)
        self.actions: list = list(actions)
        self.transitions = transitions
        self.gamma = gamma

        if isinstance(reward, Factor):
            self.reward: dict[Any, Factor] = {a: reward for a in self.actions}
        else:
            self.reward = dict(reward)

        # Infer primed Variable objects from the last element of each transition factor scope.
        first_action = self.actions[0]
        self.next_vars: tuple[Variable, ...] = tuple(
            cpd.scope[-1] for cpd in self.transitions[first_action]
        )

        self._state_enum: list[tuple] = list(
            itertools.product(*[v.domain for v in self.state_vars])
        )
        self._state_index: dict[tuple, int] = {s: i for i, s in enumerate(self._state_enum)}

    @property
    def nS(self) -> int:
        return len(self._state_enum)

    @property
    def nA(self) -> int:
        return len(self.actions)

    # ------------------------------------------------------------------
    # Flatten: materialise (|A|, |S|, |S|) MDP tables
    # ------------------------------------------------------------------

    def flatten(self) -> MDP:
        """
        Materialise the flat MDP by enumerating all joint state assignments.

        P_a(s' | s) = prod_i cpd_a_i(X_i' = s'_i | current_parents(s))
        R[a, s] = reward_factor_a.get(state_assignment(s))
        """
        nS, nA = self.nS, self.nA
        P = np.zeros((nA, nS, nS))
        R = np.zeros((nA, nS))

        for ai, action in enumerate(self.actions):
            cpds = self.transitions[action]
            rew_factor = self.reward[action]

            for si, s_tuple in enumerate(self._state_enum):
                R[ai, si] = rew_factor.get(dict(zip(self.state_vars, s_tuple)))

                for sp_idx, sp_tuple in enumerate(self._state_enum):
                    full_assign = {
                        **dict(zip(self.state_vars, s_tuple)),
                        **dict(zip(self.next_vars, sp_tuple)),
                    }
                    prob = 1.0
                    for cpd in cpds:
                        prob *= cpd.get(full_assign)
                    P[ai, si, sp_idx] = prob

        return MDP(
            states=list(self._state_enum),
            actions=self.actions,
            P=P,
            R=R,
            gamma=self.gamma,
        )

    # ------------------------------------------------------------------
    # Unroll: build a finite-horizon Influence Diagram (CID)
    # ------------------------------------------------------------------

    def as_cid(self, horizon: int) -> "InfluenceDiagram":
        """
        Unroll the FactoredMDP into a finite-horizon Influence Diagram.

        Structure (per step t = 0, ..., T-1):
          X_i^t (chance) -> A_t (decision) -> X_i^{t+1} (chance)
          X^t -> U_t (utility)

        The combined transition CPD P(X_i^{t+1} | X^t_parents, A_t)
        is built by conditioning on each action value.

        Returns an InfluenceDiagram with T decision nodes and T utility nodes.
        """
        from bn_mdp.id.diagram import InfluenceDiagram

        id_ = InfluenceDiagram()

        # Create time-sliced state variables
        slices: list[list[Variable]] = [
            [Variable(f"{v.name}_{t}", v.domain) for v in self.state_vars]
            for t in range(horizon + 1)
        ]
        action_vars = [Variable(f"A_{t}", self.actions) for t in range(horizon)]

        # Register all nodes
        for t_slice in slices:
            for var in t_slice:
                id_.add_variable(var)
        for a_var in action_vars:
            id_.add_decision(a_var)

        # Initial state: uniform (caller can replace these CPDs)
        for var in slices[0]:
            id_.set_cpd(var, Factor([var], np.ones(len(var.domain)) / len(var.domain)))

        for t in range(horizon):
            a_var = action_vars[t]

            # Uniform policy placeholder
            id_.set_cpd(a_var, Factor([a_var], np.ones(len(self.actions)) / len(self.actions)))

            # Information edges: X^t -> A_t
            for sv in slices[t]:
                id_.add_edge(sv, a_var)

            # Transition CPDs: P(X_i^{t+1} | X^t_parents_i, A_t)
            for i, next_var in enumerate(slices[t + 1]):
                # Find which original state variables appear in any action's CPD for var i
                cur_parent_originals: list[Variable] = []
                for action in self.actions:
                    cpd_i = self.transitions[action][i]
                    for sv_orig in cpd_i.scope[:-1]:
                        if sv_orig in self.state_vars and sv_orig not in cur_parent_originals:
                            cur_parent_originals.append(sv_orig)

                orig_to_t = dict(zip(self.state_vars, slices[t]))
                cpd_parent_vars = [orig_to_t[v] for v in cur_parent_originals]
                factor_parents = cpd_parent_vars + [a_var]

                # Add edges before set_cpd (required by BN scope validation)
                for pv in factor_parents:
                    id_.add_edge(pv, next_var)

                shape = tuple(len(p.domain) for p in factor_parents) + (len(next_var.domain),)
                table = np.zeros(shape)

                for combo in itertools.product(*[p.domain for p in factor_parents]):
                    assign = dict(zip(factor_parents, combo))
                    action_val = assign[a_var]
                    cpd_i = self.transitions[action_val][i]
                    for next_val in next_var.domain:
                        orig_assign = {
                            orig: assign[orig_to_t[orig]]
                            for orig in cur_parent_originals
                        }
                        orig_assign[self.next_vars[i]] = next_val
                        idx = tuple(
                            list(p.domain).index(assign[p]) for p in factor_parents
                        ) + (list(next_var.domain).index(next_val),)
                        table[idx] = cpd_i.get(orig_assign)

                id_.set_cpd(next_var, Factor(factor_parents + [next_var], table))

            # Utility node: avg_R(X^t) * gamma^t
            u_node = UtilityNode(f"U_{t}")
            id_.add_utility(u_node)
            cur_slice = slices[t]
            shape_r = tuple(len(v.domain) for v in cur_slice)
            util_table = np.zeros(shape_r)
            for combo in itertools.product(*[v.domain for v in cur_slice]):
                orig_assign = dict(zip(self.state_vars, combo))
                avg_r = np.mean([self.reward[a].get(orig_assign) for a in self.actions])
                idx = tuple(list(v.domain).index(c) for v, c in zip(cur_slice, combo))
                util_table[idx] = (self.gamma ** t) * avg_r
            id_.set_utility(u_node, Factor(cur_slice, util_table))
            for sv in cur_slice:
                id_.add_edge(sv, u_node)

        return id_

    def __repr__(self) -> str:
        return (
            f"FactoredMDP(vars={[v.name for v in self.state_vars]}, "
            f"actions={self.actions}, nS={self.nS}, gamma={self.gamma})"
        )

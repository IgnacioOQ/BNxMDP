from __future__ import annotations

from typing import Sequence

import networkx as nx

from .equation import EquationVariable, StructuralEquation
from .ordering import CausalOrdering, CausalOrderingResult


class DynamicCausalModel:
    """
    Causal model for a system of *time-indexed* structural equations.

    This class applies the Iwasaki-Simon (1994) causal ordering algorithm
    to a 2-time-slice equation template, producing:

    * the **contemporaneous** causal ordering among variables within the
      current time slice (``time_offset=0``), and
    * the **lagged** causal structure from the previous slice
      (``time_offset=-1``) to the current slice.

    The combination is a directed graph over ``(name, time_offset)`` pairs
    that serves directly as a **2-TBN template** for a Dynamic Bayesian
    Network or a Factored MDP.

    How exogeneity works
    --------------------
    Variables with ``time_offset < 0`` are *automatically* treated as
    exogenous.  At each time step their values are already known (they were
    determined in the previous step), so they play the role of "inputs" to
    the current-step equations, exactly as Iwasaki & Simon describe for
    differential/difference equation systems.

    References
    ----------
    Iwasaki, Y., & Simon, H. A. (1994). Causality in device behavior.
    *Artificial Intelligence*, 64(2), 245–285.  §5 (dynamic systems).
    """

    def __init__(
        self,
        equations: Sequence[StructuralEquation],
    ) -> None:
        """
        Parameters
        ----------
        equations
            Structural equations over time-indexed
            :class:`~bn_mdp.causal.EquationVariable` objects.  Variables
            with ``time_offset=0`` are the current-slice endogenous
            variables; variables with ``time_offset=-1`` (or lower) are
            treated as exogenous.
        """
        self.equations: list[StructuralEquation] = list(equations)

        all_vars: set[EquationVariable] = set()
        for eq in self.equations:
            all_vars.update(eq.variables)

        # Past-slice variables are exogenous for current-slice ordering
        self._exogenous: set[EquationVariable] = {
            v for v in all_vars if v.time_offset < 0
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_ordering(self) -> CausalOrderingResult:
        """
        Derive the causal ordering for the current time slice.

        Returns
        -------
        CausalOrderingResult
            Contains the causal DAG, per-variable levels, matching, and
            the list of minimal self-contained subsets in causal order.
        """
        return CausalOrdering(self.equations, self._exogenous).compute()

    def to_dbn_template(self) -> dict:
        """
        Produce a 2-TBN template dict from the derived causal structure.

        The returned dict has three keys:

        ``"intra_slice_edges"``
            List of ``(cause, effect)`` pairs where both endpoints have
            ``time_offset=0``.  These are *contemporaneous* causal links
            within the current time slice.

        ``"inter_slice_edges"``
            List of ``(cause, effect)`` pairs where the cause has
            ``time_offset=-1`` and the effect has ``time_offset=0``.
            These are the *temporal* causal links that carry information
            forward in time.

        ``"ordering"``
            The full :class:`~bn_mdp.causal.ordering.CausalOrderingResult`
            for further inspection.

        The intra- and inter-slice edge lists together define the DBN
        template graph.  To build the full unrolled graph for horizon *T*,
        replicate the template *T* times, shifting time offsets by 1 at
        each step.
        """
        result = self.compute_ordering()

        intra: list[tuple[EquationVariable, EquationVariable]] = []
        inter: list[tuple[EquationVariable, EquationVariable]] = []

        for cause, effect in result.causal_graph.edges():
            if cause.time_offset < 0 and effect.time_offset == 0:
                inter.append((cause, effect))
            elif cause.time_offset == 0 and effect.time_offset == 0:
                intra.append((cause, effect))

        return {
            "intra_slice_edges": intra,
            "inter_slice_edges": inter,
            "ordering": result,
        }

    def unroll(self, horizon: int) -> nx.DiGraph:
        """
        Unroll the 2-TBN template into a full DAG over *horizon* time steps.

        Each variable ``v`` (``time_offset=0``) at step *t* becomes node
        ``(v.name, t)``.  The returned graph has nodes labelled as
        ``(name, time)`` tuples and directed edges reflecting both the
        contemporaneous and lagged causal structure.

        Parameters
        ----------
        horizon : int
            Number of time steps to unroll (must be ≥ 1).

        Returns
        -------
        nx.DiGraph
            Unrolled causal DAG with ``(name, time)`` tuple nodes.
        """
        if horizon < 1:
            raise ValueError("horizon must be ≥ 1")

        template = self.to_dbn_template()
        G: nx.DiGraph = nx.DiGraph()

        # Collect all current-slice variable names
        current_names = {
            v.name
            for eq in self.equations
            for v in eq.variables
            if v.time_offset == 0
        }

        for t in range(horizon):
            for name in current_names:
                G.add_node((name, t))

        for t in range(horizon):
            for cause, effect in template["intra_slice_edges"]:
                G.add_edge((cause.name, t), (effect.name, t))
            if t > 0:
                for cause, effect in template["inter_slice_edges"]:
                    G.add_edge((cause.name, t - 1), (effect.name, t))

        return G

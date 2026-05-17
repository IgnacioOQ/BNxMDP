"""
Round-trip converters between our BayesianNetwork and pgmpy's BayesianNetwork.

pgmpy uses string node names and TabularCPD objects (numpy arrays with
explicit variable-order metadata).  We translate both ways.

Optional dependency: install with  pip install bn-mdp-bridge[pgmpy]
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bn_mdp.core.primitives import Variable, Factor
from bn_mdp.bn.network import BayesianNetwork

if TYPE_CHECKING:
    pass


def to_pgmpy(bn: BayesianNetwork):
    """
    Convert our BayesianNetwork to a pgmpy BayesianNetwork.

    Returns a ``pgmpy.models.BayesianNetwork`` with TabularCPD objects.
    """
    try:
        from pgmpy.models import BayesianNetwork as PgmpyBN
        from pgmpy.factors.discrete import TabularCPD
    except ImportError as e:
        raise ImportError("pgmpy is required: pip install 'bn-mdp-bridge[pgmpy]'") from e

    edges = [(u.name, v.name) for u, v in bn._graph.edges()]
    pgbn = PgmpyBN(edges)

    # Add isolated nodes (no edges)
    for var in bn.variables:
        if var.name not in pgbn.nodes():
            pgbn.add_node(var.name)

    for var in bn.variables:
        factor = bn.get_cpd(var)
        parents = bn.parents(var)
        nvar = len(var.domain)
        n_parent_states = [len(p.domain) for p in parents]

        # pgmpy wants the CPD as shape (len(var.domain), prod(parent_domains))
        # with the variable axis first.
        # Our factor has scope (*parents, var) so var axis is last.
        # Reshape: move last axis to first, then flatten parent axes.
        table = factor.table   # shape (*parent_shapes, nvar)
        if parents:
            # Move variable axis (last) to front, flatten the rest
            t = np.moveaxis(table, -1, 0)               # (nvar, *parent_shapes)
            t = t.reshape(nvar, -1)                      # (nvar, prod(parent_shapes))
        else:
            t = table.reshape(nvar, 1)

        cpd = TabularCPD(
            variable=var.name,
            variable_card=nvar,
            values=t,
            evidence=[p.name for p in parents] if parents else None,
            evidence_card=n_parent_states if parents else None,
            state_names={p.name: list(p.domain) for p in parents + [var]},
        )
        pgbn.add_cpds(cpd)

    if not pgbn.check_model():
        raise RuntimeError("pgmpy model check failed after conversion.")
    return pgbn


def from_pgmpy(pgbn) -> BayesianNetwork:
    """
    Convert a pgmpy BayesianNetwork to our BayesianNetwork.

    The variable domain is taken from the state_names metadata in each CPD.
    If state_names are not available, integer indices 0..cardinality-1 are used.
    """
    try:
        from pgmpy.models import BayesianNetwork as PgmpyBN
    except ImportError as e:
        raise ImportError("pgmpy is required: pip install 'bn-mdp-bridge[pgmpy]'") from e

    bn = BayesianNetwork()
    name_to_var: dict[str, Variable] = {}

    for cpd in pgbn.cpds:
        vname = cpd.variable
        if hasattr(cpd, "state_names") and cpd.state_names:
            domain = list(cpd.state_names[vname])
        else:
            domain = list(range(cpd.variable_card))
        name_to_var[vname] = Variable(vname, domain)

    for u_name, v_name in pgbn.edges():
        u_var = name_to_var[u_name]
        v_var = name_to_var[v_name]
        bn.add_edge(u_var, v_var)

    for var in name_to_var.values():
        if var not in bn._graph.nodes:
            bn.add_variable(var)

    for cpd in pgbn.cpds:
        var = name_to_var[cpd.variable]
        parents = [name_to_var[p] for p in cpd.variables[1:]]

        # cpd.values shape: (nvar, prod(parent_cards)) → reshape to (*parent_shapes, nvar)
        nvar = len(var.domain)
        parent_shapes = tuple(len(p.domain) for p in parents)
        values = np.asarray(cpd.values, dtype=float)

        if parents:
            t = values.reshape(nvar, *parent_shapes)
            t = np.moveaxis(t, 0, -1)   # (*parent_shapes, nvar)
        else:
            t = values.flatten()

        bn.set_cpd(var, Factor(parents + [var], t))

    return bn

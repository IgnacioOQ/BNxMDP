"""
Round-trip converters between our InfluenceDiagram and pycid's CID.

pycid is built on pgmpy; its CID class uses string node names and
TabularCPD objects.  We reuse the pgmpy shim for chance/decision nodes
and add utility handling on top.

Optional dependency: install with  pip install bn-mdp-bridge[pycid]
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bn_mdp.core.primitives import Variable, UtilityNode, Factor
from bn_mdp.id.diagram import InfluenceDiagram

if TYPE_CHECKING:
    pass


def to_pycid(id_: InfluenceDiagram):
    """
    Convert our InfluenceDiagram to a pycid CID.

    Returns a ``pycid.core.cid.CID`` object with:
      - chance and decision nodes populated from the BN structure
      - utility nodes registered via add_cpds with a constant "utility" CPD

    Note: pycid represents utility nodes as regular nodes with a special
    utility CPD; we use their NullCPD mechanism.
    """
    try:
        from pycid.core.cid import CID
        from pgmpy.factors.discrete import TabularCPD
    except ImportError as e:
        raise ImportError("pycid is required: pip install 'bn-mdp-bridge[pycid]'") from e

    from bn_mdp.compat.pgmpy_shim import to_pgmpy

    # Build the edge list for a CID (all edges, including utility edges)
    edges = []
    for u, v in id_._graph.edges():
        uname = u.name if hasattr(u, "name") else str(u)
        vname = v.name if hasattr(v, "name") else str(v)
        edges.append((uname, vname))

    decision_names = {d.name for d in id_.decision_nodes}
    utility_names = {u.name for u in id_.utility_nodes}

    cid = CID(edges, decisions=list(decision_names), utilities=list(utility_names))

    # Add CPDs for chance and decision nodes (reuse pgmpy shim logic)
    pgbn = to_pgmpy(id_)
    for cpd in pgbn.cpds:
        if cpd.variable not in decision_names and cpd.variable not in utility_names:
            cid.add_cpds(cpd)

    # Add utility CPDs as deterministic functions
    for util_node in id_.utility_nodes:
        util_factor = id_.get_utility_factor(util_node)
        parents = id_._utility_parents(util_node)
        nvar = 1  # utility nodes have a single "value" per parent assignment
        parent_shapes = [len(p.domain) for p in parents]
        n_combos = 1
        for s in parent_shapes:
            n_combos *= s

        # pycid uses TabularCPD with the utility values as a (1, n_combos) table
        values = np.zeros((1, n_combos))
        for idx, combo in enumerate(
            __import__("itertools").product(*[p.domain for p in parents])
        ):
            assign = dict(zip(parents, combo))
            values[0, idx] = util_factor.get(assign)

        cpd = TabularCPD(
            variable=util_node.name,
            variable_card=1,
            values=values,
            evidence=[p.name for p in parents] if parents else None,
            evidence_card=parent_shapes if parents else None,
        )
        cid.add_cpds(cpd)

    return cid


def from_pycid(cid) -> InfluenceDiagram:
    """
    Convert a pycid CID to our InfluenceDiagram.

    Reconstructs Variable objects from the CPD state_names metadata.
    Utility node values are read from the TabularCPD table.
    """
    try:
        from pycid.core.cid import CID
    except ImportError as e:
        raise ImportError("pycid is required: pip install 'bn-mdp-bridge[pycid]'") from e

    from bn_mdp.compat.pgmpy_shim import from_pgmpy

    decision_names = set(cid.decisions)
    utility_names = set(cid.utilities)

    id_ = InfluenceDiagram()
    name_to_node: dict[str, Variable | UtilityNode] = {}

    # Reconstruct node objects
    for cpd in cid.cpds:
        vname = cpd.variable
        if vname in utility_names:
            node = UtilityNode(vname)
            id_.add_utility(node)
        elif vname in decision_names:
            if hasattr(cpd, "state_names") and cpd.state_names:
                domain = list(cpd.state_names[vname])
            else:
                domain = list(range(cpd.variable_card))
            node = Variable(vname, domain)
            id_.add_decision(node)
        else:
            if hasattr(cpd, "state_names") and cpd.state_names:
                domain = list(cpd.state_names[vname])
            else:
                domain = list(range(cpd.variable_card))
            node = Variable(vname, domain)
            id_.add_variable(node)
        name_to_node[vname] = node

    # Reconstruct edges
    for u_name, v_name in cid.edges():
        u_node = name_to_node.get(u_name)
        v_node = name_to_node.get(v_name)
        if u_node is not None and v_node is not None:
            id_.add_edge(u_node, v_node)

    # Reconstruct CPDs
    for cpd in cid.cpds:
        vname = cpd.variable
        node = name_to_node[vname]
        parents = [name_to_node[p] for p in cpd.variables[1:] if p in name_to_node]

        values = np.asarray(cpd.values, dtype=float)

        if isinstance(node, UtilityNode):
            # Utility factor: scope = parents, table = flattened values
            parent_shapes = tuple(len(p.domain) for p in parents if isinstance(p, Variable))
            var_parents = [p for p in parents if isinstance(p, Variable)]
            util_table = values.reshape(parent_shapes) if parent_shapes else values.flatten()
            id_.set_utility(node, Factor(var_parents, util_table))
        else:
            nvar = len(node.domain)
            var_parents = [p for p in parents if isinstance(p, Variable)]
            parent_shapes = tuple(len(p.domain) for p in var_parents)
            if var_parents:
                t = values.reshape(nvar, *parent_shapes)
                t = np.moveaxis(t, 0, -1)
            else:
                t = values.flatten()
            id_.set_cpd(node, Factor(var_parents + [node], t))

    return id_

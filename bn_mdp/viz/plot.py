from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from bn_mdp.bn.network import BayesianNetwork
    from bn_mdp.id.diagram import InfluenceDiagram
    from bn_mdp.mdp.model import MDP


def plot_bn(bn: "BayesianNetwork", ax=None, title: str = "Bayesian Network", **kwargs):
    """
    Draw a BayesianNetwork using networkx + matplotlib.

    Nodes are labelled by variable name.  Spring layout is used by default.
    Returns the matplotlib Axes.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()
    for var in bn.variables:
        G.add_node(var.name)
    for u, v in bn._graph.edges():
        G.add_edge(u.name, v.name)

    if ax is None:
        _, ax = plt.subplots(figsize=kwargs.pop("figsize", (6, 4)))

    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G, pos, ax=ax, with_labels=True,
        node_size=kwargs.pop("node_size", 1800),
        node_color=kwargs.pop("node_color", "lightblue"),
        font_size=kwargs.pop("font_size", 12),
        font_weight="bold",
        arrowsize=18,
        **kwargs,
    )
    ax.set_title(title)
    return ax


def plot_id(id_: "InfluenceDiagram", ax=None, title: str = "Influence Diagram", **kwargs):
    """
    Draw an InfluenceDiagram with node-type shape conventions:
      circles  = chance nodes  (lightblue)
      squares  = decision nodes (lightyellow)
      diamonds = utility nodes  (lightgreen)
    """
    return id_.plot(ax=ax, **{**{"figsize": (7, 5)}, **kwargs})


def plot_policy_heatmap(
    policy: np.ndarray,
    value: np.ndarray,
    grid_shape: tuple[int, int],
    action_symbols: Sequence[str] | None = None,
    ax=None,
    **kwargs,
):
    """
    Draw a policy + value-function heatmap for a gridworld MDP.

    Parameters
    ----------
    policy       : ndarray shape (nS,) — action indices, one per state.
    value        : ndarray shape (nS,) — V* values.
    grid_shape   : (nrows, ncols) — the 2D layout of the states.
    action_symbols : list of strings for each action (defaults to 0,1,...).
                    Typically: ['→', '←', '↑', '↓'] or similar.
    ax           : matplotlib Axes (created if None).

    Returns the matplotlib Axes.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    nrows, ncols = grid_shape
    nS = nrows * ncols
    assert len(policy) == nS and len(value) == nS, (
        f"policy/value length {len(policy)} != grid size {nS}"
    )

    if action_symbols is None:
        action_symbols = [str(a) for a in range(int(policy.max()) + 1)]

    V_grid = value.reshape(grid_shape)
    P_grid = policy.reshape(grid_shape)

    if ax is None:
        _, ax = plt.subplots(figsize=kwargs.pop("figsize", (ncols * 1.2 + 1, nrows * 1.2 + 0.5)))

    im = ax.imshow(V_grid, cmap="YlGn", origin="upper")
    plt.colorbar(im, ax=ax, label="V*")

    for r in range(nrows):
        for c in range(ncols):
            sym = action_symbols[P_grid[r, c]]
            ax.text(c, r, sym, ha="center", va="center", fontsize=14, fontweight="bold")

    ax.set_xticks(range(ncols))
    ax.set_yticks(range(nrows))
    ax.set_title(kwargs.pop("title", "Policy + Value Heatmap"))
    return ax


def plot_dbn_template(
    fmdp,
    ax=None,
    title: str = "2-TBN Template",
    **kwargs,
):
    """
    Draw the 2-time-slice DBN template for a FactoredMDP.

    Shows current-slice variables (X^t) on the left, next-slice (X^{t+1})
    on the right, and the action A_t in the middle, with edges reflecting
    the factored transition structure (inferred from the first action's CPD scopes).
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    state_vars = fmdp.state_vars
    next_vars = fmdp.next_vars
    actions = fmdp.actions

    G = nx.DiGraph()
    # Left column: current-slice  (suffix _t)
    left = {v: f"{v.name}_t" for v in state_vars}
    right = {v: f"{v.name}_t+1" for v in next_vars}
    action_node = "A_t"

    for name in left.values():
        G.add_node(name)
    for name in right.values():
        G.add_node(name)
    G.add_node(action_node)

    # Information edges: X^t -> A_t
    for lname in left.values():
        G.add_edge(lname, action_node)

    # Transition edges based on CPD scopes (first action)
    first_action = actions[0]
    for i, cpd_i in enumerate(fmdp.transitions[first_action]):
        rname = right[next_vars[i]]
        for sv_orig in cpd_i.scope[:-1]:
            if sv_orig in state_vars:
                G.add_edge(left[sv_orig], rname)
        G.add_edge(action_node, rname)

    if ax is None:
        _, ax = plt.subplots(figsize=kwargs.pop("figsize", (7, 4)))

    nv = len(state_vars)
    pos = {}
    for i, lname in enumerate(left.values()):
        pos[lname] = (0, -i)
    pos[action_node] = (1, -(nv - 1) / 2)
    for i, rname in enumerate(right.values()):
        pos[rname] = (2, -i)

    left_names = list(left.values())
    right_names = list(right.values())

    nx.draw_networkx_edges(G, pos, ax=ax, arrowsize=16, edge_color="gray")
    nx.draw_networkx_nodes(G, pos, nodelist=left_names, ax=ax,
                           node_color="lightblue", node_size=1400)
    nx.draw_networkx_nodes(G, pos, nodelist=[action_node], ax=ax,
                           node_shape="s", node_color="lightyellow", node_size=1400)
    nx.draw_networkx_nodes(G, pos, nodelist=right_names, ax=ax,
                           node_color="lightblue", node_size=1400)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")
    ax.set_title(title)
    ax.axis("off")
    return ax

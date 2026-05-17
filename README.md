# BN × MDP Bridge

A from-scratch Python library that builds the conceptual and computational bridge between **Bayesian Networks** (BNs) and **Markov Decision Processes** (MDPs), connecting them through Dynamic Bayesian Networks, Influence Diagrams, Pearl's do-operator, Dynamic Causal Models, and Causal Influence Diagrams.

For the mathematical theory behind this bridge, see [docs/EXPLANATION.md](docs/EXPLANATION.md).

---

## Repository Layout

```
BNxMDP/
├── bn_mdp/               # main package
│   ├── core/             # Variable, Assignment, Factor primitives
│   ├── bn/               # BayesianNetwork + do-operator (intervene)
│   ├── causal/           # Dynamic Causal Models (Iwasaki-Simon algorithm)
│   ├── id/               # InfluenceDiagram / CID
│   ├── mdp/              # MDP, FactoredMDP, value/policy iteration
│   ├── viz/              # plotting helpers
│   └── compat/           # pgmpy / pycid / pymdptoolbox shims
├── examples/
│   ├── gridworld.py      # 4×4 gridworld in three equivalent forms
│   └── dynamic_causal_model.py  # Iwasaki-Simon DCM walkthrough
├── notebooks/
│   ├── Bayes_Net_from_Scratch.ipynb
│   └── Bayesian_Network_vs_HMM_Updated.ipynb
├── tests/                # pytest test suite
├── docs/
│   └── EXPLANATION.md    # mathematical bridge: BNs → DBNs → IDs → MDPs → CIDs
├── pyproject.toml
└── Makefile
```

---

## Installation

```bash
# Core (numpy, networkx, matplotlib only)
pip install -e .

# With development tools (pytest)
pip install -e ".[dev]"

# With optional compatibility libraries
pip install -e ".[compat]"   # pgmpy + pycid + pymdptoolbox
```

---

## Quick Start

### Bayesian Network with variable elimination and do-operator

```python
from bn_mdp import Variable, Factor, BayesianNetwork
import numpy as np

Rain  = Variable("Rain",  ("yes", "no"))
Sprinkler = Variable("Sprinkler", ("on", "off"))
WetGrass  = Variable("WetGrass",  ("wet", "dry"))

bn = BayesianNetwork()
bn.add_edge(Rain, Sprinkler)
bn.add_edge(Rain, WetGrass)
bn.add_edge(Sprinkler, WetGrass)

bn.set_cpd(Rain, Factor([Rain], [0.2, 0.8]))
bn.set_cpd(Sprinkler, Factor.from_cpd(Sprinkler, [Rain], np.array([[0.01, 0.99], [0.4, 0.6]])))
bn.set_cpd(WetGrass,  Factor.from_cpd(WetGrass,  [Rain, Sprinkler],
    np.array([[[0.99, 0.01],[0.8, 0.2]],[[0.9, 0.1],[0.0, 1.0]]])))

# Posterior query
f = bn.query([WetGrass], evidence={Rain: "yes"})

# Causal intervention: do(Sprinkler = "on")
bn_do = bn.intervene({Sprinkler: "on"})
```

### Dynamic Causal Model (Iwasaki-Simon algorithm)

```python
from bn_mdp import EquationVariable, StructuralEquation, DynamicCausalModel

S_prev = EquationVariable("S", time_offset=-1)
A_prev = EquationVariable("A", time_offset=-1)
S = EquationVariable("S", time_offset=0)
A = EquationVariable("A", time_offset=0)
R = EquationVariable("R", time_offset=0)

equations = [
    StructuralEquation("transition", [S_prev, A_prev, S]),
    StructuralEquation("policy",     [S, A]),
    StructuralEquation("reward",     [S, A, R]),
]

model = DynamicCausalModel(equations)
result = model.compute_ordering()   # derives S(t) → A(t) → R(t)
template = model.to_dbn_template()  # intra- and inter-slice edges
G = model.unroll(horizon=5)         # full DAG over 5 time steps
```

### Flat MDP with value iteration

```python
from bn_mdp import MDP
from bn_mdp.mdp import value_iteration
import numpy as np

states  = [0, 1, 2]
actions = ["left", "right"]
P = np.array([...])   # shape (|A|, |S|, |S|)
R = np.array([...])   # shape (|A|, |S|)

mdp = MDP(states, actions, P, R, gamma=0.99)
V, policy = value_iteration(mdp)
```

### Factored MDP flattened and unrolled as a CID

```python
from bn_mdp import Variable, Factor, FactoredMDP

Row = Variable("Row", [0, 1, 2, 3])
Col = Variable("Col", [0, 1, 2, 3])
# ... define CPD factors per action ...

fmdp = FactoredMDP([Row, Col], actions, transitions, reward_factor, gamma=0.95)
flat_mdp = fmdp.flatten()          # materialise (|A|, |S|, |S|) tables
cid      = fmdp.as_cid(horizon=3)  # finite-horizon Influence Diagram
```

---

## Module Reference

### `bn_mdp.core` — Primitives

| Class | Description |
|---|---|
| `Variable(name, domain)` | Frozen dataclass for a discrete random variable with a finite ordered domain |
| `UtilityNode(name)` | Marker for utility nodes in an Influence Diagram; carries no probability distribution |
| `Assignment` | Dict subclass with domain-checked `__setitem__`; use `Assignment.from_dict(...)` |
| `Factor(scope, table)` | Tabular factor over a tuple of `Variable`s backed by a numpy array; supports `product`, `marginalize`, `reduce`, `normalize` |

### `bn_mdp.bn` — Bayesian Network

`BayesianNetwork` stores structure as a `networkx.DiGraph` over `Variable` nodes and CPDs as `Factor` objects.

Key methods:
- `add_variable`, `add_edge`, `set_cpd`, `get_cpd`
- `query(query_vars, evidence)` — variable elimination (sum-product)
- `joint_probability(assignment)`, `marginal_probability(variable, value)`, `conditional_probability(...)`
- `intervene(assignment)` — Pearl's do-operator via mutilated graph (returns a new `BayesianNetwork`)
- `plot(ax)` — draws the DAG with matplotlib

### `bn_mdp.causal` — Dynamic Causal Models

Implements the **Iwasaki & Simon (1994)** causal ordering algorithm for structural equation systems, including time-indexed (dynamic) variants.

| Class | Description |
|---|---|
| `EquationVariable(name, time_offset)` | Variable in a structural equation; `time_offset=0` is current slice, `-1` is previous |
| `StructuralEquation(name, variables, func?)` | One equation represented by its incidence structure (which variables appear in it); optional `func` for numerical simulation |
| `CausalOrdering(equations, exogenous)` | Runs the algorithm: bipartite matching → SCC condensation → causal DAG |
| `CausalOrderingResult` | Output dataclass: `causal_graph`, `levels`, `matching`, `components` |
| `DynamicCausalModel(equations)` | Wraps `CausalOrdering` for time-indexed systems; produces 2-TBN templates and unrolled DAGs |

`DynamicCausalModel` key methods:
- `compute_ordering()` → `CausalOrderingResult` for the current time slice
- `to_dbn_template()` → `{"intra_slice_edges": [...], "inter_slice_edges": [...], "ordering": ...}`
- `unroll(horizon)` → `nx.DiGraph` with `(name, t)` tuple nodes

The algorithm works by:
1. Building a bipartite graph: equations on one side, endogenous variables on the other
2. Finding a maximum matching (each equation "solves for" one variable)
3. Orienting edges: matched → equation→variable; unmatched → variable→equation
4. Finding SCCs of the directed graph — each SCC is a **minimal self-contained subset**
5. Topological sort of the condensation DAG gives the causal order

### `bn_mdp.id` — Influence Diagram

`InfluenceDiagram` extends `BayesianNetwork` with decision and utility nodes.

Key additions:
- `add_decision(variable)`, `add_utility(node)`, `set_utility(node, factor)`
- `chance_nodes()`, `decision_nodes`, `utility_nodes`
- `set_policy(decision, policy_factor)`, `expected_utility(policy?)`
- `optimal_policy()` — exhaustive search over deterministic decision rules (tractable for small models)
- `plot(ax)` — draws circles/squares/diamonds for chance/decision/utility nodes

### `bn_mdp.mdp` — MDP and Solvers

| Class/Function | Description |
|---|---|
| `MDP(states, actions, P, R, gamma)` | Explicit tabular MDP; `P` shape `(|A|,|S|,|S|)`, `R` shape `(|A|,|S|)` |
| `MDP.from_mdptoolbox(P, R, ...)` | Constructor from pymdptoolbox-convention arrays |
| `FactoredMDP(state_vars, actions, transitions, reward, gamma)` | State = tuple of `Variable`s; transitions = per-action list of CPD `Factor`s |
| `FactoredMDP.flatten()` | Materialise the flat `(|A|,|S|,|S|)` MDP by enumerating joint states |
| `FactoredMDP.as_cid(horizon)` | Unroll into a finite-horizon `InfluenceDiagram` |
| `value_iteration(mdp, tol, max_iter)` | Synchronous Bellman backup; returns `(V*, policy)` |
| `policy_iteration(mdp, max_iter)` | Exact policy evaluation via linear solve + greedy improvement; returns `(V*, policy)` |

### `bn_mdp.viz` — Visualisation

| Function | Description |
|---|---|
| `plot_bn(bn, ax, title)` | Spring-layout DAG for a `BayesianNetwork` |
| `plot_id(id_, ax, title)` | Influence Diagram with Howard-Matheson shape conventions |
| `plot_policy_heatmap(policy, value, grid_shape, ...)` | Value-function heatmap with action-symbol overlays for gridworld MDPs |
| `plot_dbn_template(fmdp, ax, title)` | 2-TBN template for a `FactoredMDP`: left = X^t, middle = A_t, right = X^{t+1} |

### `bn_mdp.compat` — Compatibility Shims

Round-trip converters to established libraries (all optional dependencies):

| Module | Functions | Library |
|---|---|---|
| `compat.pgmpy_shim` | `to_pgmpy(bn)`, `from_pgmpy(pgbn)` | [pgmpy](https://pgmpy.org/) |
| `compat.pycid_shim` | `to_pycid(id_)`, `from_pycid(cid)` | [pycid](https://github.com/causalincentives/pycid) |
| `compat.pymdptoolbox_shim` | `to_mdptoolbox(mdp)`, `from_mdptoolbox(P, R, ...)`, `cross_check(mdp)` | [pymdptoolbox](https://pymdptoolbox.readthedocs.io/) |

---

## Examples

### `examples/gridworld.py`

Demonstrates the full BN-MDP bridge on a 4×4 stochastic gridworld in **three equivalent forms**:

1. **Flat MDP** — explicit `(|S|, |A|, |S|)` transition tables solved with value iteration
2. **Factored MDP** — state = `(Row, Col)` with a 2-TBN DBN per action; `flatten()` is verified to agree with the flat V* to 1e-5
3. **Unrolled CID** — `fmdp.as_cid(horizon=3)` produces an Influence Diagram with 3 decision nodes and 3 utility nodes

Run with:
```bash
python examples/gridworld.py
```
Saves `examples/gridworld.png` (policy heatmap + 2-TBN template).

### `examples/dynamic_causal_model.py`

Walkthrough of `DynamicCausalModel` on a minimal MDP-like structural equation system (state transition, policy, reward). Prints:
- Causal levels per variable
- Minimal self-contained subsets
- Equation-to-variable matching
- Causal edges
- 2-TBN template (intra- and inter-slice edges)
- Unrolled DAG for horizon = 3

Run with:
```bash
python examples/dynamic_causal_model.py
```

---

## Running Tests

```bash
# From-scratch tests only (no optional libraries required)
make test

# All tests including compatibility cross-checks
make test-compat

# With coverage report
make test-cov
```

The test suite covers:
- `test_core.py` — `Variable`, `Factor` algebra, `Assignment` domain checking
- `test_bn.py` — structure, CPDs, variable elimination, do-operator
- `test_do.py` — interventional vs observational queries in confounder examples
- `test_causal.py` — `EquationVariable`, `StructuralEquation`, `CausalOrdering` (chain, exogenous, simultaneous, three-level, under-determined), `DynamicCausalModel` (simple chain, MDP structure, unrolling)
- `test_id.py` — `InfluenceDiagram`, `expected_utility`, `optimal_policy`
- `test_mdp.py` — `MDP` construction, value/policy iteration convergence
- `test_factored.py` — `FactoredMDP.flatten()`, `as_cid()`, parity with flat MDP
- `test_compat.py` — round-trip tests for pgmpy/pycid/pymdptoolbox (skipped if libraries absent)

---

## Conceptual Map

The frameworks form a 2D lattice:

```
                  no time               with time (2-TBN / unrolled)
                  ──────────────────    ──────────────────────────────
no decisions   |  Bayesian Network      Dynamic Bayesian Network
with decisions |  Influence Diagram     MDP / POMDP  (unrolled ID)
+ causal sem.  |  Causal BN             Causal Influence Diagram (CID)
```

- Left → right: **add time** (DBN / 2-TBN template)
- Top → bottom: **add an agent** (decision + utility nodes)
- The bottom-right cell is where MDPs, do-operators, and graphical models meet

An MDP action is `do()` applied to a decision node with no chance-node parents. The do-operator collapses to ordinary conditioning in the well-specified case, which is why MDP textbooks write `P(s'|s,a)` without mentioning causality. See [docs/EXPLANATION.md](docs/EXPLANATION.md) for the full derivation.

---

## Key References

- Iwasaki, Y., & Simon, H. A. (1994). Causality in device behavior. *Artificial Intelligence*, 64(2), 245–285.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
- Howard, R. A., & Matheson, J. E. (1984). Influence diagrams. *Decision Analysis*, 2(3), 127–143.
- Boutilier, C., Dean, T., & Hanks, S. (1999). Decision-theoretic planning. *JAIR*, 11, 1–94.
- Everitt, T. et al. (2021). Agent incentives: A causal perspective. *AAAI 2021*.
- Koller, D., & Friedman, N. (2009). *Probabilistic Graphical Models*. MIT Press.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

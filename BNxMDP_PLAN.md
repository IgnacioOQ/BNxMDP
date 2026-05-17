---
status: todo
type: plan
id: bn_x_mdp.plan
description: Phased plan to build a from-scratch Python codebase that demonstrates and tests the BN-MDP bridge, with compatibility shims to pgmpy, PyCID, and pymdptoolbox for cross-validation.
label: [planning]
injection: informational
volatility: evolving
scope: project-specific
last_checked: '2026-05-17'
---
# BN × MDP Bridge Codebase Plan

This plan turns the conceptual bridge in `BNxMDP_EXPLANATION.md` into a concrete, testable Python repository. The guiding principle is **from-scratch first, compatibility second**: every concept (BN, DBN, ID, MDP, CID, do-operator, value iteration) gets a clean, didactic Python implementation that grows out of the existing `BayesianNetwork` class in `Bayes_Net_from_Scratch.ipynb`. Each module then gets a compatibility shim so that the same model can be round-tripped to and from an established library (`pgmpy` for BNs/DBNs, `pycid` for CIDs, `pymdptoolbox` for solver cross-checks). The cross-checks are how we know our from-scratch math is correct.

Target layout for the repo:

```text
bn-mdp-bridge/
├── bn_mdp/
│   ├── core/         # variables, assignments, factors, DAG primitives
│   ├── bn/           # BayesianNetwork + do-operator
│   ├── id/           # InfluenceDiagram + CID semantics
│   ├── mdp/          # MDP, factored MDP via 2-TBN, solvers
│   ├── viz/          # plotting (BN, ID, policies, value functions)
│   └── compat/       # pgmpy / pycid / pymdptoolbox shims
├── examples/         # gridworld, sprinkler, one-step MDP-as-CID, ...
├── tests/            # parity tests vs reference libraries
└── notebooks/        # narrative walkthroughs paralleling the EXPLANATION doc
```

The tasks below are ordered so each depends only on what comes before it. Tasks marked `stretch` are nice-to-haves once the spine is in place.

## Bootstrap repository and dependencies

```yaml
status: todo
type: task
id: bn_x_mdp.plan.bootstrap
estimate: 0.5d
```

Create the package skeleton, `pyproject.toml`, and a minimal test harness with `pytest`. Pin core deps (`numpy`, `networkx`, `matplotlib`) and add optional-extras groups for `pgmpy`, `pycid`, and `pymdptoolbox` (so the from-scratch code runs without them, and the compatibility tests run only when they are installed). Add a CI-style `make test` target that runs the from-scratch tests and skips the compatibility tests if extras are absent.

## Core primitives: variables, assignments, and factors

```yaml
status: todo
type: task
id: bn_x_mdp.plan.core_primitives
blocked_by: [bn_x_mdp.plan.bootstrap]
estimate: 1d
```

Introduce small, well-tested primitives that the rest of the codebase reuses: a `Variable(name, domain)` dataclass, an `Assignment` mapping (built on Python dicts but with explicit domain checking), and a `Factor` class for tabular CPDs (a numpy array plus an ordered tuple of `Variable` scopes). These are the things the original notebook handles ad-hoc with raw Python dicts; promoting them to typed objects pays off the moment we want to do factor products, marginalization, and intervention symbolically. Test with hand-computed factor products and marginals.

## Extend the BayesianNetwork class

```yaml
status: todo
type: task
id: bn_x_mdp.plan.bn_class
blocked_by: [bn_x_mdp.plan.core_primitives]
estimate: 1d
```

Refactor the notebook's `BayesianNetwork` to use the new primitives (so `set_cpd` takes a `Factor` instead of a nested dict). Keep the API surface — `add_edge`, `joint_probability`, `marginal_probability` — but make `marginal_probability` work for general queries `P(X = x | E = e)` by variable elimination over factors rather than brute-force enumeration over $2^n$ assignments. Variable elimination is still didactic and turns the implementation from $O(2^n)$ into something polynomial in well-structured networks. Cross-check against the notebook's existing 5-node example and against `pgmpy`'s `VariableElimination` on the same network.

## Implement the do-operator on the BN

```yaml
status: todo
type: task
id: bn_x_mdp.plan.do_operator
blocked_by: [bn_x_mdp.plan.bn_class]
estimate: 1d
```

Add an `intervene(self, assignment)` method that returns a *new* `BayesianNetwork` with the mutilated graph: edges into the intervened variables are removed and their CPDs are replaced by point masses. This is the truncated-factorization construction from `BNxMDP_EXPLANATION.md`. Verify on a textbook confounder example (e.g., the smoking-cancer toy network) that $P(Y \mid X = x)$ and $P(Y \mid \mathrm{do}(X = x))$ disagree in the presence of a confounder and agree in its absence. Cross-check against `pgmpy.inference.CausalInference.query`.

## Influence Diagram and Causal Influence Diagram classes

```yaml
status: todo
type: task
id: bn_x_mdp.plan.id_class
blocked_by: [bn_x_mdp.plan.do_operator]
estimate: 1.5d
```

Add an `InfluenceDiagram` class that subclasses (or composes) the `BayesianNetwork` and additionally tracks two disjoint sets of nodes: `decision_nodes` and `utility_nodes`. Utility nodes carry a deterministic function of their parents. Implement `expected_utility(policy)` by substituting each decision node's CPD with the policy's chosen distribution and reducing to a BN query. Implement `optimal_policy()` by exhaustive search over decision rules for the small didactic case, and document the path to backward induction (Shachter's variable-elimination-on-IDs algorithm) as a follow-up. Treat `InfluenceDiagram` as a CID by convention — the only difference is that we *interpret* edges causally, which matters for the do-operator.

## MDP class with explicit (S, A, P, R, γ)

```yaml
status: todo
type: task
id: bn_x_mdp.plan.mdp_class
blocked_by: [bn_x_mdp.plan.core_primitives]
estimate: 1d
```

Build an `MDP` class with `states`, `actions`, `P: ndarray[|S|, |A|, |S|]`, `R: ndarray[|S|, |A|]` (or `[|S|, |A|, |S|]`), and `gamma`. Provide constructors from explicit tables and from a `pymdptoolbox`-compatible `(P, R)` pair. This task is independent of the BN side — it is the "flat" MDP we will later show is a special case of an unrolled CID.

## Solvers: value iteration and policy iteration from scratch

```yaml
status: todo
type: task
id: bn_x_mdp.plan.solvers
blocked_by: [bn_x_mdp.plan.mdp_class]
estimate: 1d
```

Implement value iteration and policy iteration directly from the Bellman equation. Comment the code heavily — convergence criterion, contraction-factor argument, why policy iteration terminates in finitely many steps. Verify on the canonical forest example (`mdptoolbox.example.forest`) that our from-scratch solver matches `pymdptoolbox.mdp.ValueIteration` to within numerical tolerance.

## Bridge: factored MDP via 2-TBN DBN

```yaml
status: todo
type: task
id: bn_x_mdp.plan.factored_mdp
blocked_by: [bn_x_mdp.plan.id_class, bn_x_mdp.plan.mdp_class]
estimate: 1.5d
```

This is the keystone task. Build a `FactoredMDP` whose state is a *vector* of variables $X^{(1)}, \ldots, X^{(n)}$ and whose transition kernel is a 2-time-slice DBN keyed by action: for each action $a$, the user supplies one DBN $P_a(X_{t+1} \mid X_t)$. Provide two methods: `flatten()` materializes the explicit $|S| \times |A| \times |S|$ table by enumerating joint state assignments (so we can hand it to the flat solver), and `as_cid(horizon=T)` unrolls the factored MDP into a finite-horizon CID by replicating the 2-TBN motif $T$ times and wiring in decision and utility nodes. A parity test verifies that solving via `flatten()` + value iteration agrees with computing the optimal policy on the unrolled CID, on a small didactic example.

## Visualization layer

```yaml
status: todo
type: task
id: bn_x_mdp.plan.viz
blocked_by: [bn_x_mdp.plan.id_class]
estimate: 1d
```

Wrap `networkx` + `matplotlib` to render BNs, IDs/CIDs (with the conventional shape conventions: circle = chance, square = decision, diamond = utility), DBN two-slice templates side-by-side, and policy/value-function heatmaps for gridworld-like MDPs. Keep the API as `model.plot(ax=None, **kwargs)` so the same call works in scripts and notebooks. The shape conventions for IDs are documented in Howard & Matheson (1984) and are what PyCID uses by default — borrow them.

## Compatibility shims

```yaml
status: todo
type: task
id: bn_x_mdp.plan.compat
blocked_by: [bn_x_mdp.plan.do_operator, bn_x_mdp.plan.factored_mdp]
estimate: 1.5d
```

Implement three round-trip converters in `bn_mdp/compat/`: `to_pgmpy(bn)` / `from_pgmpy(...)` for BNs and DBNs (pgmpy's `DynamicBayesianNetwork` uses a 2-TBN with `(name, time_slice)` tuple nodes — match that), `to_pycid(id)` / `from_pycid(...)` for IDs and CIDs (PyCID is itself built on pgmpy, so this should reuse most of the BN converter), and `to_mdptoolbox(mdp)` / `from_mdptoolbox(P, R)` for flat MDPs. Each converter ships with a parity test: build a model on each side and assert that joint probabilities (BN), expected utilities (ID), or optimal value vectors (MDP) agree to numerical tolerance.

## Worked example: gridworld in three guises

```yaml
status: todo
type: task
id: bn_x_mdp.plan.example_gridworld
blocked_by: [bn_x_mdp.plan.compat, bn_x_mdp.plan.solvers, bn_x_mdp.plan.viz]
estimate: 1d
```

Pick a small $4 \times 4$ stochastic gridworld and express it three ways in the same notebook: as a flat MDP (with explicit tables), as a factored MDP whose state is $(\text{row}, \text{col})$ with a 2-TBN transition per action, and as an unrolled finite-horizon CID. Solve all three. Show that the optimal policies and value functions agree. Visualize the CID for $T = 3$ and the value-function heatmap for the flat MDP. This notebook is the executable counterpart of `BNxMDP_EXPLANATION.md` — the reader can run it end-to-end and see every bridge in action.

## Stretch: POMDP via partial observability and PyCID CID-of-MDP

```yaml
status: todo
type: task
id: bn_x_mdp.plan.stretch_pomdp
blocked_by: [bn_x_mdp.plan.example_gridworld]
estimate: 2d
```

Drop the $S_t \to A_t$ information edge from the gridworld CID and add an observation node $O_t$ with $S_t \to O_t$; this is the smallest move from MDP to POMDP. Implement belief-state updating as a BN filtering query and finite-horizon value iteration over beliefs (the discretized case is tractable for tiny grids). Cross-check against PyCID's built-in CID solver on the same finite-horizon model. This is where the BN-side inference machinery (filtering, smoothing) and the MDP-side planning machinery (value iteration) meet in a single algorithm — the conceptual payoff of having built both halves from scratch.

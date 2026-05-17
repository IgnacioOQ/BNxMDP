---
status: active
type: explanation
id: bn_x_mdp.explanation
description: Conceptual bridge between Bayesian Networks and Markov Decision Processes via Dynamic Bayesian Networks, Influence Diagrams, Pearl's do-operator, and Causal Influence Diagrams.
label: [source-material]
injection: background
volatility: stable
scope: general
last_checked: '2026-05-17'
---
# Bayesian Networks and Markov Decision Processes: The Conceptual Bridge

This document explains how Bayesian Networks (BNs) and Markov Decision Processes (MDPs) are related — not as two unrelated formalisms that happen to use graphs, but as two ends of a single chain of generalizations that runs through **Dynamic Bayesian Networks** (DBNs), **Influence Diagrams** (IDs), Pearl's **do-operator**, and modern **Causal Influence Diagrams** (CIDs).

Read this when you want to understand *why* an MDP can be drawn as a graphical model, what the do-operator has to do with "taking an action," and how the existing `BayesianNetwork` class in `Bayes_Net_from_Scratch.ipynb` is one rung of a ladder that, with a few well-chosen extensions, reaches MDPs and beyond. The companion document `BNxMDP_PLAN.md` turns the ideas here into a concrete codebase plan.

## Why This Bridge Matters

The two formalisms answer different-looking questions. A BN answers *"given what I know, what is the probability that…?"* It is a model of a passive joint distribution. An MDP answers *"given the dynamics of the world, what should I do to maximize long-run reward?"* It is a model of a controllable, time-extended process.

The bridge matters for three reasons. First, **modeling efficiency**: large MDPs almost always have factored state, and a DBN is the standard way to represent the transition kernel compactly. Second, **causal correctness**: an action is not the same as an observation, and treating them the same is the source of many silent bugs in agent design — the do-operator is what makes the difference precise. Third, **agent analysis**: the modern theory of *agent incentives* — what an RL agent is incentivized to learn, control, or manipulate — is built directly on causal influence diagrams, a graphical model that contains both BNs and MDPs as special cases.

## Building Block 1: The Bayesian Network as a Factored Joint

A Bayesian Network is a pair $(\mathcal{G}, \mathcal{P})$ where $\mathcal{G} = (V, E)$ is a directed acyclic graph (DAG) over random variables $X = \{X_1, \ldots, X_n\}$, and $\mathcal{P} = \{P(X_i \mid \mathrm{Pa}(X_i))\}_{i=1}^n$ is a collection of conditional probability distributions (CPDs), one per node. The defining property is that the joint distribution factorizes along the DAG:

$$P(X_1, \ldots, X_n) \;=\; \prod_{i=1}^n P\!\left(X_i \mid \mathrm{Pa}(X_i)\right).$$

This is exactly what the notebook's `joint_probability` method computes. Marginal queries — `marginal_probability` in the notebook — are obtained by summing the joint over the unobserved variables. The DAG also encodes a set of conditional independencies (read off via *d-separation*), which is what makes inference computationally tractable on structured networks.

A BN as defined here is **descriptive**: it tells you what the world looks like, not what you can do to it. It has no notion of time, no notion of action, and no notion of reward. Every step that follows lifts one of these limitations.

## Building Block 2: The MDP as a Controlled Stochastic Process

A Markov Decision Process is a tuple $\mathcal{M} = (S, A, P, R, \gamma)$ where $S$ is a set of states, $A$ a set of actions, $P(s' \mid s, a)$ a transition kernel, $R(s, a)$ a reward function (sometimes $R(s, a, s')$), and $\gamma \in [0, 1)$ a discount factor. A *policy* $\pi: S \to A$ (deterministic) or $\pi(a \mid s)$ (stochastic) selects actions given states, and the *value function* under $\pi$ is the expected discounted return,

$$V^\pi(s) \;=\; \mathbb{E}_\pi\!\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \;\middle|\; s_0 = s\right].$$

The optimal value satisfies the **Bellman optimality equation**,

$$V^*(s) \;=\; \max_{a \in A}\!\left[\, R(s, a) \;+\; \gamma \sum_{s' \in S} P(s' \mid s, a)\, V^*(s') \,\right],$$

which is the fixed point that value iteration converges to. The canonical references for these definitions are Puterman (1994) for the operations-research treatment and Sutton & Barto (2018) for the reinforcement-learning treatment.

An MDP differs from a BN on three axes at once: it has **time** (state evolves), it has **action** (an exogenous chooser), and it has **reward** (a scalar to optimize). The next three sections add these axes one at a time.

## Bridge 1 — Time: Dynamic Bayesian Networks

A **Dynamic Bayesian Network** (DBN) is a Bayesian Network whose variables carry a time index. In the standard *2-time-slice* presentation, you specify a prior network over $X_0$ and a transition network over $(X_t, X_{t+1})$, and unrolling these two slices through time yields a BN over $X_0, X_1, X_2, \ldots$ that factorizes as

$$P(X_0, X_1, \ldots, X_T) \;=\; P(X_0)\, \prod_{t=0}^{T-1} P(X_{t+1} \mid X_t),$$

with each inter-slice term itself factored over the components of $X_{t+1}$ given their (possibly cross-slice) parents — see Dean & Kanazawa (1989) for the original formulation. The Markov property — $X_{t+1} \perp X_{<t} \mid X_t$ — falls out of the DAG structure, not from an extra axiom.

This is the first half of an MDP transition kernel: a DBN gives you $P(s' \mid s)$ as a factored model. What is still missing is the action.

## Bridge 2 — Action and Reward: Influence Diagrams

An **Influence Diagram** (ID), introduced by Howard & Matheson in 1984 as a graphical representation of decision problems, extends a Bayesian Network with two new node types:

- **Chance nodes** (drawn as circles) — ordinary random variables with CPDs, exactly as in a BN.
- **Decision nodes** (drawn as squares) — variables whose values are *chosen* by an agent, not sampled. They have no CPD; their incoming edges are *information edges*, encoding what the agent observes before deciding.
- **Utility nodes** (drawn as diamonds) — deterministic functions of their parents, contributing additively to a scalar utility $U = \sum_j u_j(\mathrm{Pa}(U_j))$.

A *policy* in an ID is a collection of decision rules $\pi_k$, one per decision node, each mapping the values of that node's information parents to a value (or distribution over values) for the decision. The optimal policy maximizes expected utility:

$$\pi^* \;=\; \arg\max_\pi\, \mathbb{E}_\pi\!\left[ \sum_j U_j \right] \;=\; \arg\max_\pi \sum_{\mathbf{x}} P_\pi(\mathbf{x})\, U(\mathbf{x}),$$

where $P_\pi(\mathbf{x})$ is the joint distribution induced by replacing each decision node's CPD with the chosen $\pi_k$ and then factorizing as usual.

A **finite-horizon MDP is an Influence Diagram whose graph repeats one structural motif per time step.** Unrolling $\mathcal{M} = (S, A, P, R, \gamma)$ for $T$ steps gives an ID with chance nodes $S_0, S_1, \ldots, S_T$, decision nodes $A_0, \ldots, A_{T-1}$, and utility nodes $R_1, \ldots, R_T$ (with $\gamma^t$ folded into the utility weight). The information edges $S_t \to A_t$ encode full observability of the state at decision time; the causal edges $\{S_t, A_t\} \to S_{t+1}$ encode the transition kernel; and the utility edges $\{S_t, A_t\} \to R_{t+1}$ encode the reward function. The DeepMind Safety Research group's expository write-up of CIDs makes this picture explicit for the one-step case (Everitt et al., 2019, blog post). With a factored state and a 2-TBN transition, the result is a **factored MDP** in the sense of Boutilier, Dean & Hanks (1999) and Boutilier, Dearden & Goldszmidt (2000).

```text
        ┌─── S_t ────┐         ┌─── S_{t+1} ───┐
        │            ▼         │               ▼
        │           A_t ───────┘             A_{t+1} ─── ...
        │            │                          │
        │            ▼                          ▼
        └─────────► R_{t+1}                   R_{t+2}
```

The reverse direction is just as useful: any ID with the right structural motif *is* an MDP, and any ID without that motif is a strictly more general object — generalizing partial observability (a missing $S_t \to A_t$ edge becomes a POMDP), non-stationarity (per-step CPDs that vary with $t$), and multi-agent settings (multiple decision-makers).

## Bridge 3 — Causality: The do-Operator and What "Taking an Action" Means

The subtlety the user originally flagged — *"what is the relation between the do-operator in the do-calculus and the actions in the MDP?"* — deserves a section of its own, because the relationship is exact but counterintuitive.

Pearl's **do-operator** is defined on a causal BN by *truncated factorization* (also called the *mutilated graph* construction; see Pearl, 2009, ch. 3): to compute the post-intervention distribution after setting $X_j$ to value $x_j^*$, replace the CPD of $X_j$ with a point mass and remove its incoming edges,

$$P\!\left(X_1, \ldots, X_n \mid \mathrm{do}(X_j = x_j^*)\right) \;=\; \mathbb{1}[X_j = x_j^*] \cdot \prod_{i \ne j} P\!\left(X_i \mid \mathrm{Pa}(X_i)\right).$$

In general $P(Y \mid \mathrm{do}(X = x)) \ne P(Y \mid X = x)$: conditioning updates on observed evidence, intervention forces a value. Now consider the action node $A_t$ in the influence-diagram view of an MDP. **By construction, $A_t$ has no chance-node parents** — only the information edge $S_t \to A_t$. When the agent chooses $A_t = a$, it is performing $\mathrm{do}(A_t = a)$, but because there is no upstream chance structure on $A_t$ to mutilate, the truncated-factorization formula collapses to ordinary conditioning:

$$P\!\left(S_{t+1} \mid \mathrm{do}(A_t = a),\, S_t = s\right) \;=\; P\!\left(S_{t+1} \mid A_t = a,\, S_t = s\right) \;=\; P(s' \mid s, a).$$

This is why MDP textbooks write the transition kernel as a conditional probability without ever mentioning causality: the two coincide on the specific graph topology MDPs use. The causal reading nonetheless matters in two situations the standard MDP formalism does not handle gracefully:

1. **Off-policy data**: if the data was generated by behaviour policy $\pi_b(a \mid s, z)$ that depends on some confounder $z$, then $A_t$ now has chance-node parents in the data-generating process. Estimating $P(s' \mid s, a)$ by conditioning on observed $(s, a, s')$ pairs without adjusting for $z$ yields the wrong kernel. The do-operator makes this misalignment explicit.
2. **Policy evaluation under intervention**: the value of a target policy $\pi$ from data collected under $\pi_b$ is a do-calculus quantity — the inner workings of counterfactual policy evaluation reduce to interventional queries on the underlying CID.

The slogan is therefore: **an MDP action is do() applied to a decision node with no chance-node parents; the formalism collapses to conditioning in the well-specified case, and you need the full do-operator the moment that assumption breaks.**

## Bridge 4 — Putting It Together: Causal Influence Diagrams

A **Causal Influence Diagram** (CID), as developed by Everitt and collaborators (Everitt et al., 2019; Everitt, Carey, Langlois, Ortega & Legg, 2021), is an Influence Diagram whose chance-node CPDs and structural edges are interpreted *causally* in Pearl's sense — every edge $X \to Y$ asserts that $X$ is a direct cause of $Y$, and interventions on chance nodes follow the do-operator semantics. CIDs are the framework in which the bridge above becomes formally seamless:

- A CID with no decision and no utility nodes is a **causal BN**.
- A CID with chance, decision, and utility nodes but no temporal structure is a one-shot **decision problem**.
- A CID whose graph is the time-unrolled MDP motif of the previous section is an **MDP**, with the do-operator giving the correct semantics of *action*.
- A CID with multiple decision-makers per time step is a **Multi-Agent Influence Diagram** (MAID), extending naturally to game-theoretic settings (Hammond, Fox, Everitt, Abate & Wooldridge, 2023).

The DeepMind blog post "Progress on Causal Influence Diagrams" walks through the one-step MDP as a CID explicitly: $S_1 \to A_1 \to S_2 \to R_2$ with information edge $S_1 \to A_1$ (the post is linked in the references). The PyCID library (Fox, Everitt, Carey, Langlois, Abate & Wooldridge, 2021) implements CIDs in Python on top of `pgmpy` and `networkx`; it is the natural compatibility target for any from-scratch implementation that wants to verify itself against an established reference.

## A Compact Mental Map

The frameworks line up along two axes — *does the graph have decision/utility nodes?* and *does it carry temporal structure?*:

```text
                  no time           with time (unrolled / 2-TBN)
                  ──────────────    ─────────────────────────────
no decisions   |  Bayesian Net      Dynamic Bayesian Network
with decisions |  Influence Diag.   MDP / POMDP (as unrolled ID)
+ causal sem.  |  Causal BN         Causal Influence Diagram (CID)
```

Reading the table left-to-right is "add time"; top-to-bottom is "add an agent". The bottom-right cell — CIDs — is where the user's original idea lives: *"the environment is a BN with a do operator, and the MDP adds the rest."* That intuition is exactly correct: a CID *is* a causal BN augmented with decision and utility nodes, and the MDP is the special case where the graph is built from a single repeating motif.

## What This Bridge Does Not Give You for Free

Three things are *not* automatic from the bridge above:

1. **Continuous state and action spaces.** Everything in this document is stated for discrete variables. The bridge extends to continuous variables, but the inference algorithms change (Gaussian belief propagation, particle methods, function approximation).
2. **Partial observability.** A POMDP is an ID with hidden state — the agent observes $O_t$, not $S_t$. The belief state $b(s) = P(s \mid o_{0:t}, a_{0:t-1})$ is itself the result of a BN filtering computation, which is one of the cleanest places where BN inference and MDP planning are entangled in a single algorithm.
3. **Reinforcement learning vs. planning.** All of the above assumes the model — CPDs, transition kernel, reward — is known. Reinforcement learning is the regime where the agent must *learn* these from experience. The graphical model still applies (in fact, it tells you which independencies your learning algorithm can exploit), but the algorithms shift from value iteration to TD-learning, Q-learning, and so on.

## References

- Boutilier, C., Dean, T., & Hanks, S. (1999). [Decision-theoretic planning: Structural assumptions and computational leverage](https://jair.org/index.php/jair/article/view/10237). *Journal of Artificial Intelligence Research*, 11, 1–94.
- Boutilier, C., Dearden, R., & Goldszmidt, M. (2000). Stochastic dynamic programming with factored representations. *Artificial Intelligence*, 121(1–2), 49–107.
- Dean, T., & Kanazawa, K. (1989). A model for reasoning about persistence and causation. *Computational Intelligence*, 5(3), 142–150.
- Everitt, T., Ortega, P. A., Barnes, E., & Legg, S. (2019). [Understanding agent incentives using causal influence diagrams, Part I: Single action settings](https://arxiv.org/abs/1902.09980). arXiv:1902.09980.
- Everitt, T., Carey, R., Langlois, E., Ortega, P. A., & Legg, S. (2021). [Agent incentives: A causal perspective](https://arxiv.org/abs/2102.01685). *AAAI 2021*. arXiv:2102.01685.
- Fox, J., Everitt, T., Carey, R., Langlois, E., Abate, A., & Wooldridge, M. (2021). [PyCID: A Python library for causal influence diagrams](https://proceedings.scipy.org/articles/majora-1b6fd038-008). *Proceedings of the 20th Python in Science Conference (SciPy 2021)*.
- Howard, R. A., & Matheson, J. E. (1984). Influence diagrams. In *Readings on the Principles and Applications of Decision Analysis, Vol. II*. Strategic Decisions Group. Reprinted (2005) in *Decision Analysis*, 2(3), 127–143.
- Koller, D., & Friedman, N. (2009). *Probabilistic Graphical Models: Principles and Techniques*. MIT Press.
- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference*. Morgan Kaufmann.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
- Puterman, M. L. (1994). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Causal Incentives Working Group: [causalincentives.com](https://causalincentives.com/) — hub for ongoing CID research and PyCID.
- DeepMind Safety Research, "Progress on Causal Influence Diagrams" (2021): [alignmentforum.org post](https://www.alignmentforum.org/posts/Cd7Hw492RqooYgQAS/progress-on-causal-influence-diagrams).
- Tom Everitt's personal site (mentioned by the user in the original notebook): [tomeveritt.se](https://www.tomeveritt.se/).

"""
Dynamic causal model example — Iwasaki & Simon (1994) style.

We model a minimal MDP-like system using structural equations:

    S(t) = f( S(t-1), A(t-1) )    # state transition
    A(t) = g( S(t) )               # policy (action depends on current state)
    R(t) = h( S(t), A(t) )         # reward (contemporaneous)

The three equations involve five variables:
  - S(t-1), A(t-1)  — previous slice, treated as *exogenous* (already fixed)
  - S(t), A(t), R(t) — current slice, *endogenous* (to be solved for)

The Iwasaki-Simon algorithm derives the causal ordering automatically
from the equation incidence structure, without needing to be told which
variable each equation solves for.

Expected result
---------------
Causal order within the current slice:   S(t) → A(t) → R(t)
Inter-slice edges:  S(t-1) → S(t),  A(t-1) → S(t)

Run
---
    python examples/dynamic_causal_model.py
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from bn_mdp.causal import EquationVariable, StructuralEquation, DynamicCausalModel


# ---------------------------------------------------------------------------
# 1. Declare variables
# ---------------------------------------------------------------------------

S_prev = EquationVariable("S", time_offset=-1)   # S(t-1) — previous state
A_prev = EquationVariable("A", time_offset=-1)   # A(t-1) — previous action

S = EquationVariable("S", time_offset=0)          # S(t) — current state
A = EquationVariable("A", time_offset=0)          # A(t) — current action
R = EquationVariable("R", time_offset=0)          # R(t) — current reward


# ---------------------------------------------------------------------------
# 2. Declare structural equations (incidence only — no functional form needed)
# ---------------------------------------------------------------------------
#
# Each StructuralEquation names the variables that appear together in one
# physical/model equation.  The algorithm only uses these sets to ask:
# "which variables constrain which equations?"

equations = [
    StructuralEquation("transition", [S_prev, A_prev, S]),   # S depends on past
    StructuralEquation("policy",     [S, A]),                 # A depends on current S
    StructuralEquation("reward",     [S, A, R]),              # R depends on S and A
]


# ---------------------------------------------------------------------------
# 3. Build the dynamic causal model and derive the ordering
# ---------------------------------------------------------------------------

model = DynamicCausalModel(equations)
result = model.compute_ordering()


# ---------------------------------------------------------------------------
# 4. Inspect the results
# ---------------------------------------------------------------------------

print("=" * 60)
print("Causal levels (exogenous = -1, higher = determined later)")
print("=" * 60)
for var, level in sorted(result.levels.items(), key=lambda kv: (kv[1], kv[0].name)):
    tag = " [exogenous]" if level == -1 else ""
    print(f"  {repr(var):15s}  level {level}{tag}")

print()
print("=" * 60)
print("Minimal self-contained subsets  (causal order, low → high)")
print("=" * 60)
for i, component in enumerate(result.components):
    names = ", ".join(repr(v) for v in sorted(component, key=lambda v: v.name))
    label = "simultaneous" if len(component) > 1 else "singleton"
    print(f"  [{i}] {{{names}}}  ({label})")

print()
print("=" * 60)
print("Equation-to-variable matching  (which equation solves what)")
print("=" * 60)
for var, eq in result.matching.items():
    print(f"  {eq.name!r:15s}  →  {repr(var)}")

print()
print("=" * 60)
print("Causal edges  (direct causes derived from structural equations)")
print("=" * 60)
for cause, effect in sorted(result.causal_graph.edges(),
                             key=lambda e: (repr(e[0]), repr(e[1]))):
    print(f"  {repr(cause):15s}  →  {repr(effect)}")

print()
print("=" * 60)
print("2-TBN template")
print("=" * 60)
template = model.to_dbn_template()

print("  Intra-slice edges (contemporaneous, within t):")
if template["intra_slice_edges"]:
    for cause, effect in template["intra_slice_edges"]:
        print(f"    {repr(cause)} → {repr(effect)}")
else:
    print("    (none)")

print("  Inter-slice edges (temporal, t-1 → t):")
for cause, effect in template["inter_slice_edges"]:
    print(f"    {repr(cause)} → {repr(effect)}")

print()
print("=" * 60)
print("Unrolled DAG  (horizon = 3 steps)")
print("=" * 60)
G = model.unroll(horizon=3)
for (src_name, src_t), (dst_name, dst_t) in sorted(G.edges()):
    print(f"  ({src_name}, t={src_t}) → ({dst_name}, t={dst_t})")

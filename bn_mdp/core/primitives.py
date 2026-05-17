from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class UtilityNode:
    """
    A utility node in an Influence Diagram.

    Unlike a Variable, a utility node has no probability distribution and
    no domain — it carries a real-valued utility function of its parents.
    """

    name: str

    def __repr__(self) -> str:
        return f"UtilityNode({self.name!r})"


@dataclass(frozen=True)
class Variable:
    """A discrete random variable with a finite domain."""

    name: str
    domain: tuple

    def __post_init__(self):
        object.__setattr__(self, "domain", tuple(self.domain))
        if len(self.domain) < 2:
            raise ValueError(f"Variable '{self.name}' must have at least 2 domain values.")

    def __repr__(self) -> str:
        return f"Variable({self.name!r}, domain={self.domain})"

    def __hash__(self):
        return hash((self.name, self.domain))

    def __eq__(self, other):
        return isinstance(other, Variable) and self.name == other.name and self.domain == other.domain


class Assignment(dict):
    """
    A mapping from Variable -> value with domain checking.

    Behaves like a plain dict but validates that every assigned value
    is in the variable's declared domain.
    """

    def __setitem__(self, var: Variable, value: Any):
        if value not in var.domain:
            raise ValueError(
                f"Value {value!r} is not in the domain of {var!r}. "
                f"Domain is {var.domain}."
            )
        super().__setitem__(var, value)

    @classmethod
    def from_dict(cls, mapping: dict[Variable, Any]) -> "Assignment":
        a = cls()
        for var, val in mapping.items():
            a[var] = val
        return a

    def restrict(self, variables: Sequence[Variable]) -> "Assignment":
        """Return a new Assignment containing only the given variables."""
        return Assignment.from_dict({v: self[v] for v in variables if v in self})


class Factor:
    """
    A tabular factor over an ordered tuple of Variables.

    The underlying storage is a numpy array whose axes correspond to
    the variables in `scope` (in order) and whose size along each axis
    equals `len(var.domain)`.

    Indexing convention: `table[i0, i1, ...]` is the factor value when
    scope[0] = scope[0].domain[i0], scope[1] = scope[1].domain[i1], etc.
    """

    def __init__(self, scope: Sequence[Variable], table: np.ndarray):
        self.scope: tuple[Variable, ...] = tuple(scope)
        expected_shape = tuple(len(v.domain) for v in self.scope)
        table = np.asarray(table, dtype=float)
        if table.shape != expected_shape:
            raise ValueError(
                f"Table shape {table.shape} does not match scope shape {expected_shape}."
            )
        self.table = table

    # ------------------------------------------------------------------
    # Indexing helpers
    # ------------------------------------------------------------------

    def _indices(self, assignment: dict) -> tuple:
        """Convert a (partial) variable→value dict to axis indices."""
        return tuple(
            list(v.domain).index(assignment[v])
            for v in self.scope
        )

    def get(self, assignment: dict) -> float:
        """Look up the factor value for a complete scope assignment."""
        return float(self.table[self._indices(assignment)])

    # ------------------------------------------------------------------
    # Factor algebra
    # ------------------------------------------------------------------

    def product(self, other: "Factor") -> "Factor":
        """Return the factor product (union of scopes)."""
        combined_scope = list(self.scope)
        other_only = [v for v in other.scope if v not in self.scope]
        combined_scope.extend(other_only)

        new_shape = tuple(len(v.domain) for v in combined_scope)
        new_table = np.zeros(new_shape)

        for idx in itertools.product(*[range(len(v.domain)) for v in combined_scope]):
            assignment = {v: v.domain[i] for v, i in zip(combined_scope, idx)}
            new_table[idx] = self.get(assignment) * other.get(assignment)

        return Factor(combined_scope, new_table)

    def marginalize(self, variable: Variable) -> "Factor":
        """Sum out `variable` from this factor."""
        if variable not in self.scope:
            raise ValueError(f"{variable} not in scope {self.scope}.")
        axis = self.scope.index(variable)
        new_scope = [v for v in self.scope if v != variable]
        new_table = self.table.sum(axis=axis)
        return Factor(new_scope, new_table)

    def reduce(self, variable: Variable, value: Any) -> "Factor":
        """Fix `variable = value` and remove it from the scope."""
        if variable not in self.scope:
            raise ValueError(f"{variable} not in scope {self.scope}.")
        axis = self.scope.index(variable)
        idx = list(variable.domain).index(value)
        slices = [slice(None)] * len(self.scope)
        slices[axis] = idx
        new_scope = [v for v in self.scope if v != variable]
        new_table = self.table[tuple(slices)]
        return Factor(new_scope, new_table)

    def normalize(self) -> "Factor":
        """Return a factor scaled so that all entries sum to 1."""
        total = self.table.sum()
        if total == 0:
            raise ValueError("Cannot normalize a zero factor.")
        return Factor(self.scope, self.table / total)

    @classmethod
    def from_cpd(cls, variable: Variable, parents: Sequence[Variable], table: np.ndarray) -> "Factor":
        """
        Build a CPD factor: scope = (*parents, variable), table axes ordered
        so the last axis corresponds to `variable`.
        """
        scope = list(parents) + [variable]
        return cls(scope, table)

    def __repr__(self) -> str:
        names = ", ".join(v.name for v in self.scope)
        return f"Factor(scope=({names}), shape={self.table.shape})"

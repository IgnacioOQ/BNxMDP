from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence


@dataclass(frozen=True)
class EquationVariable:
    """
    A variable in a structural equation system.

    For static systems, set ``time_offset=0`` (the default).  For dynamic
    (time-indexed) systems the convention follows Iwasaki & Simon (1994):
    ``time_offset=0`` is the *current* time slice and ``time_offset=-1`` is
    the immediately preceding slice.  Variables from earlier slices are
    treated as exogenous when ordering variables within the current slice,
    because their values are already fixed by the previous step.

    References
    ----------
    Iwasaki, Y., & Simon, H. A. (1994). Causality in device behavior.
    Artificial Intelligence, 64(2), 245–285.  §5 (dynamic systems).
    """

    name: str
    time_offset: int = 0

    def __repr__(self) -> str:
        if self.time_offset == 0:
            return self.name
        return f"{self.name}[t{self.time_offset:+d}]"

    def at_offset(self, offset: int) -> "EquationVariable":
        """Return the same variable name at a different time offset."""
        return EquationVariable(self.name, offset)


class StructuralEquation:
    """
    One equation in a structural equation model.

    The equation is represented by its *incidence structure* — the
    ``frozenset`` of :class:`EquationVariable` objects that appear in it —
    rather than by a functional form.  The incidence structure is all the
    Iwasaki-Simon causal ordering algorithm needs: it only asks which
    variables constrain which equations.

    An optional ``func`` callable can be attached for numerical experiments.
    It should accept a ``dict[EquationVariable, float]`` of current variable
    values and return the equation's *residual* (0.0 when the equation is
    satisfied).  Attach it when you want to verify solutions or drive
    simulation; the ordering algorithm ignores it.
    """

    def __init__(
        self,
        name: str,
        variables: Sequence[EquationVariable],
        func: Optional[Callable[[dict], float]] = None,
    ) -> None:
        self.name = name
        self.variables: frozenset[EquationVariable] = frozenset(variables)
        self.func = func

    # ------------------------------------------------------------------
    # Dunder helpers so equations can live in sets / dicts
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = ", ".join(sorted(repr(v) for v in self.variables))
        return f"StructuralEquation({self.name!r}, {{{parts}}})"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StructuralEquation) and self.name == other.name

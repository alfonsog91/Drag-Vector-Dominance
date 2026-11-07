"""Model-energy accounting for convex edge-flow results."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from aerotvc.graph import FlowResult, Graph, _grounded_solve


def _coefficient(value, graph, name):
    try:
        result = np.broadcast_to(np.asarray(value, dtype=float), (graph.m,))
    except ValueError as exc:
        raise ValueError(f"{name} must be a scalar or length m") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _energy(x, a, b, c):
    return float(np.sum(a * x**2 / 2 + b * x**4 / 4 + c * x))


@dataclass(frozen=True)
class AeroResult:
    """A flow result and its model-energy reduction from a feasible start."""

    solution: FlowResult
    energy_initial: float
    energy_final: float
    reduction_pct: float
    iterations: int

    label: ClassVar[str] = "model energy reduction"

    @property
    def x(self):
        return self.solution.x

    @classmethod
    def from_solution(cls, graph, supply, solution, a=1.0, b=0.0, c=0.0):
        if not isinstance(graph, Graph):
            raise TypeError("graph must be a Graph")
        if not isinstance(solution, FlowResult):
            raise TypeError("solution must be a FlowResult")

        supply = np.asarray(supply, dtype=float)
        if supply.shape != (graph.n,) or not np.all(np.isfinite(supply)):
            raise ValueError("supply must be a finite vector of length n")
        for nodes in graph.components:
            component_supply = supply[nodes]
            if abs(np.sum(component_supply)) > (
                    1e-12 * max(1, np.sum(np.abs(component_supply)))):
                raise ValueError("supply must balance in each connected component")

        a = _coefficient(a, graph, "a")
        b = _coefficient(b, graph, "b")
        c = _coefficient(c, graph, "c")
        if np.any(a <= 0) or np.any(b < 0):
            raise ValueError("a must be positive and b nonnegative")
        if (solution.x.shape != (graph.m,)
                or not np.all(np.isfinite(solution.x))
                or not np.isfinite(solution.energy)):
            raise ValueError("solution must contain a finite flow of length m")

        initial_potential = _grounded_solve(graph, graph.L, supply)
        initial_x = np.asarray(graph.B.T @ initial_potential).reshape(-1)
        energy_initial = _energy(initial_x, a, b, c)
        energy_final = float(solution.energy)
        if energy_initial == 0:
            if energy_final != 0:
                raise ValueError("percentage reduction is undefined for zero initial energy")
            reduction_pct = 0.0
        else:
            reduction_pct = 100 * (energy_initial - energy_final) / energy_initial

        return cls(solution, energy_initial, energy_final, reduction_pct,
                   solution.iterations)

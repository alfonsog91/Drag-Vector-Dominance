"""Strictly convex signed edge flows constrained by component mass balance."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve
from scipy.sparse.csgraph import connected_components


class Graph:
    def __init__(self, n, tails, heads):
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer")
        arrays = [np.asarray(v) for v in (tails, heads)]
        if any(v.ndim != 1 for v in arrays) or arrays[0].shape != arrays[1].shape:
            raise ValueError("tails and heads must be equal-length vectors")
        for v in arrays:
            if not np.all(np.isfinite(v)) or np.any(v != np.floor(v)) or (
                    np.any(v < 0) or np.any(v >= n)):
                raise ValueError("edge endpoints must be integer node indices")
        self.n = n
        self.tails, self.heads = (v.astype(int) for v in arrays)
        self.m = len(self.tails)
        self.B = np.zeros((n, self.m))
        np.add.at(self.B, (self.tails, np.arange(self.m)), -1)
        np.add.at(self.B, (self.heads, np.arange(self.m)), 1)
        self.L = self.B @ self.B.T
        _, labels = connected_components(np.abs(self.B) @ np.abs(self.B).T,
                                         directed=False)
        self.components = [np.flatnonzero(labels == k) for k in np.unique(labels)]


def nozzle_supply(graph, source, sink, mdot):
    if not np.isfinite(mdot) or mdot < 0:
        raise ValueError("mdot must be finite and nonnegative")
    for node in (source, sink):
        if not isinstance(node, (int, np.integer)) or not 0 <= node < graph.n:
            raise ValueError("source and sink must be node indices")
    s = np.zeros(graph.n)
    s[source] -= mdot
    s[sink] += mdot
    return s


def _grounded_solve(graph, matrix, rhs):
    potential = np.zeros(graph.n)
    for nodes in graph.components:
        free = nodes[1:]
        if free.size:
            potential[free] = solve(matrix[np.ix_(free, free)], rhs[free],
                                    assume_a="pos")
    return potential


@dataclass(frozen=True)
class FlowResult:
    x: np.ndarray
    energy: float
    iterations: int


def solve_flow(graph, supply, a=1.0, b=0.0, c=0.0, tol=1e-10, max_iter=100):
    s = np.asarray(supply, dtype=float)
    if s.shape != (graph.n,) or not np.all(np.isfinite(s)):
        raise ValueError("supply must be a finite vector of length n")
    for nodes in graph.components:
        if abs(np.sum(s[nodes])) > 1e-12 * max(1, np.sum(np.abs(s[nodes]))):
            raise ValueError("supply must balance in each connected component")
    coefficients = []
    for value in (a, b, c):
        try:
            array = np.broadcast_to(np.asarray(value, dtype=float), (graph.m,))
        except ValueError as exc:
            raise ValueError("coefficients must be scalars or length m") from exc
        if not np.all(np.isfinite(array)):
            raise ValueError("coefficients must be finite")
        coefficients.append(array)
    a, b, c = coefficients
    if np.any(a <= 0) or np.any(b < 0):
        raise ValueError("a must be positive and b nonnegative")
    if not np.isfinite(tol) or tol <= 0 or max_iter < 1:
        raise ValueError("tol and max_iter must be positive")

    def energy(x, linear=c):
        return float(np.sum(a * x**2 / 2 + b * x**4 / 4 + linear * x))

    x = graph.B.T @ _grounded_solve(graph, graph.L, s)
    if not graph.m:
        return FlowResult(x, 0.0, 0)
    # Node-potential costs are constant on feasible flows; remove them from
    # both the gradient and line search to avoid large gauge-dependent terms.
    projected_c = c - graph.B.T @ _grounded_solve(graph, graph.L, graph.B @ c)
    for iteration in range(max_iter + 1):
        gradient = a * x + b * x**3 + projected_c
        inverse_hessian = 1 / (a + 3 * b * x**2)
        laplacian = (graph.B * inverse_hessian) @ graph.B.T
        potential = _grounded_solve(graph, laplacian,
                                    graph.B @ (inverse_hessian * gradient))
        residual = gradient - graph.B.T @ potential
        direction = -inverse_hessian * residual
        if np.linalg.norm(direction, ord=np.inf) <= tol * (
                1 + np.linalg.norm(x, ord=np.inf)):
            return FlowResult(x, energy(x), iteration)
        if iteration == max_iter:
            break
        slope = gradient @ direction
        current = energy(x, projected_c)
        step = 1.0
        for _ in range(60):
            trial = x + step * direction
            if energy(trial, projected_c) <= current + 1e-4 * step * slope:
                x = trial
                break
            step /= 2
        else:
            raise RuntimeError("Armijo line search failed")
    raise RuntimeError("flow Newton solver did not converge")

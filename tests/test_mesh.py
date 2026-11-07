import math
import os
import platform
import time

import numpy as np
import pytest
from scipy.sparse import csc_matrix, issparse
from scipy.sparse.linalg import factorized

import aerotvc.graph as graph_module
from aerotvc.mesh import icosphere


class _DenseAllocationGuard:
    def __getattr__(self, name):
        return getattr(np, name)

    def zeros(self, shape, *args, **kwargs):
        dimensions = (shape,) if isinstance(shape, int) else tuple(shape)
        if len(dimensions) > 1 and math.prod(dimensions) > 1_000_000:
            raise MemoryError("frequency-32 mesh must not allocate dense graph matrices")
        return np.zeros(shape, *args, **kwargs)


@pytest.fixture(scope="module")
def icosphere32():
    original_numpy = graph_module.np
    graph_module.np = _DenseAllocationGuard()
    try:
        return icosphere(32)
    finally:
        graph_module.np = original_numpy


def _grounded_solver(graph):
    reduced = graph.L[1:, 1:]
    if issparse(reduced):
        reduced = reduced.tocsc()
    else:
        reduced = csc_matrix(reduced)
    solve_reduced = factorized(reduced)

    def solve(rhs):
        potential = np.zeros(graph.n)
        potential[1:] = solve_reduced(rhs[1:])
        return potential

    return solve


def _energy(x, a, b):
    return float(np.sum(a * x**2 / 2 + b * x**4 / 4))


def _independent_projected_gradient(graph, supply, a, b):
    incidence = graph.B
    solve_laplacian = _grounded_solver(graph)
    x = np.asarray(incidence.T @ solve_laplacian(supply)).reshape(-1)

    for _ in range(200):
        gradient = a * x + b * x**3
        node_gradient = np.asarray(incidence @ gradient).reshape(-1)
        projected = gradient - np.asarray(
            incidence.T @ solve_laplacian(node_gradient)).reshape(-1)
        direction = -projected
        if np.linalg.norm(direction, ord=np.inf) <= 1e-12 * (
                1 + np.linalg.norm(x, ord=np.inf)):
            return x, _energy(x, a, b)

        current_energy = _energy(x, a, b)
        slope = float(gradient @ direction)
        step = 0.5
        for _ in range(50):
            candidate = x + step * direction
            if _energy(candidate, a, b) <= (
                    current_energy + 1e-4 * step * slope):
                x = candidate
                break
            step *= 0.5
        else:
            raise AssertionError("independent projected-gradient line search failed")

    raise AssertionError("independent projected-gradient solver did not converge")


def _relative_optimality_residual(graph, x, a, b):
    gradient = a * x + b * x**3
    projected = gradient - np.asarray(
        graph.B.T @ _grounded_solver(graph)(np.asarray(graph.B @ gradient).reshape(-1))
    ).reshape(-1)
    denominator = max(np.linalg.norm(gradient, ord=np.inf),
                      np.finfo(float).tiny)
    return np.linalg.norm(projected, ord=np.inf) / denominator


def _hardware_description():
    processor = platform.processor()
    if not processor:
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
                processor = next(
                    line.split(":", 1)[1].strip()
                    for line in cpuinfo
                    if line.lower().startswith("model name")
                )
        except (OSError, StopIteration):
            processor = "unknown CPU"
    return (f"{platform.system()} {platform.machine()}, {processor}, "
            f"{os.cpu_count()} logical CPUs")


def _timing_summary(samples):
    return (f"min/median/max="
            f"{min(samples):.6f}/{float(np.median(samples)):.6f}/"
            f"{max(samples):.6f}s")


def test_frequency_32_icosphere_topology(icosphere32):
    graph = icosphere32
    triangle_frequency = 32**2
    assert graph.n == 10 * triangle_frequency + 2 == 10_242
    assert graph.m == 30 * triangle_frequency == 30_720
    assert np.all(graph.tails < graph.heads)
    edges = np.column_stack((graph.tails, graph.heads))
    assert np.unique(edges, axis=0).shape[0] == graph.m
    degree = np.bincount(
        np.concatenate((graph.tails, graph.heads)), minlength=graph.n)
    assert np.count_nonzero(degree == 5) == 12
    assert np.count_nonzero(degree == 6) == graph.n - 12
    assert np.all(np.isin(degree, (5, 6)))


def test_frequency_32_solve_matches_independent_projected_gradient(icosphere32):
    from aerotvc.graph import solve_flow

    graph = icosphere32
    rng = np.random.default_rng(2)
    a = 1 + rng.random(graph.m)
    b = 0.2 * rng.random(graph.m)
    supply = np.zeros(graph.n)
    supply[0] = -1
    supply[-1] = 1
    solver_times = []
    oracle_times = []

    for _ in range(3):
        started = time.perf_counter()
        result = solve_flow(graph, supply, a=a, b=b, c=0)
        solver_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        oracle_x, oracle_energy = _independent_projected_gradient(
            graph, supply, a, b)
        oracle_times.append(time.perf_counter() - started)

        np.testing.assert_allclose(graph.B @ result.x, supply, rtol=0, atol=1e-9)
        assert _relative_optimality_residual(graph, result.x, a, b) <= 1e-9
        assert abs(result.energy - oracle_energy) <= 1e-6
        assert abs(_energy(oracle_x, a, b) - oracle_energy) <= 1e-12

    print(
        f"frequency-32 solver benchmark hardware={_hardware_description()}; "
        f"solve_flow seconds {_timing_summary(solver_times)}; "
        f"independent projected-gradient seconds {_timing_summary(oracle_times)}"
    )
    assert float(np.median(solver_times)) <= 3 * float(np.median(oracle_times))

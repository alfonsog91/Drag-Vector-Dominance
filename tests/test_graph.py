import numpy as np
import pytest
from scipy.linalg import null_space
from scipy.optimize import minimize

from aerotvc.graph import Graph, nozzle_supply, solve_flow


def test_incidence_and_icosahedron_spectrum(icosahedron):
    graph = icosahedron
    assert graph.B.shape == (12, 30)
    phi = np.arange(12.)**2
    np.testing.assert_allclose(graph.B.T @ phi, phi[graph.heads] - phi[graph.tails])
    x = np.arange(30.)
    direct = np.zeros(12)
    np.add.at(direct, graph.tails, -x)
    np.add.at(direct, graph.heads, x)
    np.testing.assert_allclose(graph.B @ x, direct)
    np.testing.assert_allclose(graph.L, graph.B @ graph.B.T)
    expected = [0] + [5 - np.sqrt(5)] * 3 + [6] * 5 + [5 + np.sqrt(5)] * 3
    np.testing.assert_allclose(np.linalg.eigvalsh(graph.L), expected, atol=1e-12)


def test_parallel_paths_and_energy():
    graph = Graph(4, [0, 1, 0, 2], [1, 3, 2, 3])
    supply = nozzle_supply(graph, 0, 3, 5)
    np.testing.assert_array_equal(supply, [-5, 0, 0, 5])
    result = solve_flow(graph, supply, a=[0.5, 0.5, 2, 2])
    np.testing.assert_allclose(result.x, [4, 4, 1, 1], atol=1e-10)
    assert result.energy == pytest.approx(0.5 * 0.8 * 5**2)
    np.testing.assert_allclose(graph.B @ result.x, supply, atol=1e-11)


def test_independent_nullspace_bfgs(icosahedron):
    graph = icosahedron
    supply = nozzle_supply(graph, 0, 11, 3)
    a = np.linspace(0.5, 2, 30)
    b = np.linspace(0.1, 0.4, 30)
    c = np.linspace(-0.2, 0.3, 30)
    x0 = np.linalg.lstsq(graph.B, supply, rcond=None)[0]
    Z = null_space(graph.B)
    energy = lambda x: np.sum(a * x**2 / 2 + b * x**4 / 4 + c * x)
    oracle = minimize(lambda z: energy(x0 + Z @ z), np.zeros(Z.shape[1]),
                      jac=lambda z: Z.T @ (a * (x0 + Z @ z)
                                          + b * (x0 + Z @ z)**3 + c),
                      method="BFGS", options={"gtol": 1e-9})
    assert np.linalg.norm(oracle.jac, ord=np.inf) < 1e-7
    result = solve_flow(graph, supply, a=a, b=b, c=c)
    np.testing.assert_allclose(result.x, x0 + Z @ oracle.x, atol=2e-7)
    assert result.energy == pytest.approx(oracle.fun, abs=1e-10)
    np.testing.assert_allclose(graph.B @ result.x, supply, atol=1e-10)
    np.testing.assert_allclose(Z.T @ (a * result.x + b * result.x**3 + c),
                               0, atol=1e-8)


def test_balanced_disconnected_components_and_isolated_node():
    graph = Graph(5, [0, 2], [1, 3])
    result = solve_flow(graph, [-2, 2, -3, 3, 0], a=1)
    np.testing.assert_allclose(result.x, [2, 3], atol=1e-12)
    empty = solve_flow(Graph(1, [], []), [0], a=1)
    assert empty.x.size == 0
    assert empty.energy == 0


@pytest.mark.parametrize("call", [
    lambda: solve_flow(Graph(4, [0, 2], [1, 3]), [-1, 0, 0, 1], a=1),
    lambda: solve_flow(Graph(2, [0], [1]), [-1, 2], a=1),
    lambda: solve_flow(Graph(2, [0], [1]), [-1, 1], a=0),
    lambda: solve_flow(Graph(2, [0], [1]), [-1, 1], a=-1),
    lambda: solve_flow(Graph(2, [0], [1]), [-1, 1], a=1, b=-1),
    lambda: nozzle_supply(Graph(2, [0], [1]), 0, 1, -1),
    lambda: Graph(2, [0], [2]),
    lambda: Graph(2, [0, 1], [1]),
    lambda: Graph(2, [0.5], [1]),
    lambda: solve_flow(Graph(2, [0], [1]), [-1, 1], a=float("nan")),
])
def test_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()

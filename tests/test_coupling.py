import numpy as np
import pytest

from aerotvc.graph import Graph, nozzle_supply, solve_flow
from aerotvc.nozzle import Nozzle


def test_nozzle_icosahedron_conservation_and_quadratic_scaling(icosahedron):
    mdot = Nozzle(0.01, 0.06, g=1.2, R=300).solve(7e6, 3200, 101325).mdot
    graph = icosahedron
    supply = nozzle_supply(graph, 0, 11, mdot)
    result = solve_flow(graph, supply, a=1)
    np.testing.assert_allclose(graph.B @ result.x, supply, atol=1e-9)
    outgoing = np.sum(result.x[graph.tails == 0]) - np.sum(result.x[graph.heads == 0])
    assert outgoing == pytest.approx(mdot, rel=1e-12)
    doubled = solve_flow(graph, 2 * supply, a=1)
    np.testing.assert_allclose(doubled.x, 2 * result.x, atol=1e-9)
    assert doubled.energy == pytest.approx(4 * result.energy, rel=1e-12)
    potentials = np.linalg.pinv(graph.B @ graph.B.T) @ supply
    assert result.energy == pytest.approx(0.5 * supply @ potentials, rel=1e-12)


def test_quartic_path_load_decreases_with_nozzle_mass_flow():
    mdot = Nozzle(0.01, 0.06, g=1.2, R=300).solve(7e6, 3200, 101325).mdot
    graph = Graph(4, [0, 1, 0, 2], [1, 3, 2, 3])
    shares = []
    for flow in (mdot / 10, mdot):
        result = solve_flow(graph, nozzle_supply(graph, 0, 3, flow),
                            a=1, b=[1, 1, 0, 0])
        q = result.x[0]
        assert 2 * q + 2 * q**3 == pytest.approx(2 * (flow - q), rel=1e-9)
        np.testing.assert_allclose(result.x, [q, q, flow - q, flow - q], atol=1e-9)
        shares.append(q / flow)
    assert 0 < shares[1] < shares[0] < 0.5

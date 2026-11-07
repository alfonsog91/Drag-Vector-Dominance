import numpy as np
import pytest

from aerotvc.graph import Graph, nozzle_supply, solve_flow
from aerotvc.nozzle import Nozzle, area_ratio, mach_from_area_ratio


def test_disconnected_graph_rejects_supply_unbalanced_per_component():
    graph = Graph(4, [0, 2], [1, 3])
    with pytest.raises(ValueError):
        solve_flow(graph, [-1, 0, 0, 1])


def test_connected_graph_rejects_supply_with_nonzero_sum():
    graph = Graph(2, [0], [1])
    with pytest.raises(ValueError):
        solve_flow(graph, [-1, 2])


@pytest.mark.parametrize("coefficient", [0.0, -1.0])
def test_flow_rejects_nonpositive_quadratic_cost(coefficient):
    graph = Graph(2, [0], [1])
    with pytest.raises(ValueError):
        solve_flow(graph, [-1, 1], a=coefficient)


def test_nozzle_rejects_exit_area_smaller_than_throat_area():
    with pytest.raises(ValueError):
        Nozzle(A_t=0.02, A_exit=0.01)


@pytest.mark.parametrize("gamma", [1.0, 0.9])
def test_nozzle_rejects_gamma_not_greater_than_one(gamma):
    with pytest.raises(ValueError):
        Nozzle(A_t=0.01, A_exit=0.02, g=gamma)


@pytest.mark.parametrize("name", ["p0", "T0"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nozzle_rejects_nonfinite_stagnation_inputs(name, value):
    parameters = {"p0": 7e6, "T0": 3200.0}
    parameters[name] = value
    with pytest.raises(ValueError):
        Nozzle(A_t=0.01, A_exit=0.02).solve(**parameters)


@pytest.mark.parametrize("mdot", [float("nan"), float("inf"), -1.0])
def test_nozzle_supply_rejects_nonfinite_or_negative_mass_flow(mdot):
    graph = Graph(2, [0], [1])
    with pytest.raises(ValueError):
        nozzle_supply(graph, 0, 1, mdot)


def test_zero_mass_flow_is_accepted_and_produces_zero_edge_flow():
    graph = Graph(2, [0], [1])
    supply = nozzle_supply(graph, 0, 1, 0.0)
    result = solve_flow(graph, supply)
    np.testing.assert_array_equal(result.x, [0.0])


def test_very_large_area_ratio_is_invertible():
    ratio = 1e9
    mach = mach_from_area_ratio(ratio)
    assert mach > 1
    assert area_ratio(mach) == pytest.approx(ratio, rel=1e-10)


def test_subsonic_area_ratio_inversion_is_supported():
    mach = mach_from_area_ratio(4.0, branch="subsonic")
    assert 0 < mach < 1
    assert area_ratio(mach) == pytest.approx(4.0, rel=1e-12)


def test_single_edge_graph_is_supported():
    graph = Graph(2, [0], [1])
    result = solve_flow(graph, [-2.0, 2.0])
    np.testing.assert_allclose(result.x, [2.0], rtol=0, atol=1e-12)


def test_parallel_edges_are_supported():
    graph = Graph(2, [0, 0, 0], [1, 1, 1])
    result = solve_flow(graph, [-3.0, 3.0], a=[1.0, 2.0, 4.0])
    expected = 3 * np.array([1.0, 0.5, 0.25]) / 1.75
    np.testing.assert_allclose(result.x, expected, rtol=0, atol=1e-12)

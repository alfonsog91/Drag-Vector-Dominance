from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from aerotvc.graph import Graph, solve_flow
from aerotvc.ledger import AeroResult


def _energy(x, a, b, c):
    return float(np.sum(a * x**2 / 2 + b * x**4 / 4 + c * x))


def test_aero_result_ledgers_model_energy_reduction_without_mutating_flow():
    graph = Graph(4, [0, 0, 1, 2], [1, 2, 3, 3])
    supply = np.array([-2.0, 0.0, 0.0, 2.0])
    a = np.array([0.5, 1.0, 1.5, 2.0])
    b = np.array([0.1, 0.2, 0.3, 0.4])
    c = np.array([-0.1, 0.05, 0.2, -0.15])
    flow = solve_flow(graph, supply, a=a, b=b, c=c)
    original_x = flow.x.copy()
    initial_x = np.linalg.lstsq(graph.B, supply, rcond=None)[0]

    result = AeroResult.from_solution(graph, supply, flow, a=a, b=b, c=c)
    expected_initial = _energy(initial_x, a, b, c)
    expected_final = _energy(original_x, a, b, c)
    expected_reduction = 100 * (expected_initial - expected_final) / expected_initial

    assert result.label == "model energy reduction"
    assert result.energy_initial == pytest.approx(expected_initial, abs=1e-12)
    assert result.energy_final == pytest.approx(expected_final, abs=1e-12)
    assert result.energy_final <= result.energy_initial
    assert result.reduction_pct == pytest.approx(expected_reduction, abs=1e-12)
    assert result.iterations == flow.iterations
    assert result.solution is flow
    assert result.x is flow.x
    np.testing.assert_array_equal(result.x, original_x)
    with pytest.raises(FrozenInstanceError):
        result.energy_final = 0.0

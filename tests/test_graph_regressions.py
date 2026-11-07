import numpy as np
import pytest

from aerotvc.graph import Graph, solve_flow


def test_large_constant_linear_cost_does_not_accept_starting_flow():
    graph = Graph(2, [0, 0], [1, 1])
    result = solve_flow(graph, [-1, 1], a=[0.5, 2], c=[1e12, 1e12])
    np.testing.assert_allclose(result.x, [0.8, 0.2], rtol=0, atol=1e-9)
    assert result.iterations > 0


def test_convergence_on_last_permitted_update():
    graph = Graph(2, [0, 0], [1, 1])
    result = solve_flow(graph, [-1, 1], a=[1, 3], max_iter=1)
    np.testing.assert_allclose(result.x, [0.75, 0.25], rtol=0, atol=1e-9)
    assert result.iterations == 1


def test_unconverged_last_update_still_raises():
    graph = Graph(2, [0, 0], [1, 1])
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_flow(graph, [-1, 1], a=[1, 3], b=[2, 4], max_iter=1)


@pytest.mark.parametrize("seed", [0, 7, 42])
@pytest.mark.parametrize("scale", [1, 1e6, 1e12])
@pytest.mark.parametrize("max_iter", [1, 100])
def test_node_potential_shift_preserves_solution_and_convergence(
        seed, scale, max_iter):
    graph = Graph(4, [0, 0, 2, 2], [1, 1, 3, 3])
    supply = [-1, 1, -2, 2]
    a = [0.5, 2, 1, 3]
    b = [0.25, 0.5, 0.5, 0.25]
    c = np.array([0.25, -0.5, -0.25, 0.5])
    # Dyadic costs and integer potentials retain the costs even at large scale.
    phi = np.random.default_rng(seed).integers(-100, 101, graph.n) * scale

    def outcome(cost):
        try:
            return solve_flow(graph, supply, a=a, b=b, c=cost,
                              max_iter=max_iter)
        except RuntimeError as exc:
            assert "did not converge" in str(exc)
            return None

    baseline = outcome(c)
    shifted = outcome(c + graph.B.T @ phi)
    assert (baseline is None) == (shifted is None)
    if max_iter == 1:
        assert baseline is None
    else:
        assert baseline is not None
        np.testing.assert_allclose(shifted.x, baseline.x, rtol=0, atol=1e-9)
        np.testing.assert_allclose(graph.B @ shifted.x, supply, rtol=0,
                                   atol=1e-10)

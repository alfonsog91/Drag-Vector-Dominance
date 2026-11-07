import os

import numpy as np
import pytest

from aerotvc import ad


def test_energy_gradient_hvp_against_independent_oracles():
    rng = np.random.default_rng(2026)
    x = rng.normal(size=30)
    v = rng.normal(size=30)
    a, b, c = rng.uniform(0.5, 2, (3, 30))
    energy = ad.make_energy(a, b, c)
    expected_value = np.sum(a * x**2 / 2 + b * x**4 / 4 + c * x)
    expected_grad = a * x + b * x**3 + c
    value, gradient = ad.value_and_grad(energy)(x)
    assert value == pytest.approx(expected_value, rel=1e-14)
    np.testing.assert_allclose(gradient, expected_grad, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ad.grad(energy)(x), expected_grad, rtol=0, atol=1e-12)
    eps = 1e-5
    direct = lambda y: np.sum(a * y**2 / 2 + b * y**4 / 4 + c * y)
    differences = [(direct(x + eps * e) - direct(x - eps * e)) / (2 * eps)
                   for e in np.eye(30)]
    np.testing.assert_allclose(gradient, differences, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(ad.hvp(energy, x, v),
                               (a + 3 * b * x**2) * v, atol=1e-12, rtol=1e-12)


def test_numpy_reverse_mode_general_expression():
    x = ad.Var(np.array([0.2, -0.4, 0.8]))
    y = ((2 * x + 1)**3 / 3 - x * x).sum()
    y.backward()
    np.testing.assert_allclose(x.grad, 2 * (2 * x.value + 1)**2 - 2 * x.value,
                               atol=1e-14)
    scalar = ad.Var(2.)
    (((x * scalar).sum()) + scalar**2).backward()
    assert scalar.grad == pytest.approx(np.sum(x.value) + 4)


def test_backend_selection():
    if os.environ.get("AEROTVC_DISABLE_JAX") == "1":
        assert ad.BACKEND == "numpy"
    else:
        try:
            import jax  # noqa: F401
        except ImportError:
            assert ad.BACKEND == "numpy"
        else:
            assert ad.BACKEND == "jax"


def test_jax_float64_and_general_hvp():
    if ad.BACKEND != "jax":
        pytest.skip("JAX backend unavailable or explicitly disabled")
    import jax
    import jax.numpy as jnp

    assert jax.config.x64_enabled
    x = np.array([0.2, -0.3])
    v = np.array([1., 2.])
    energy = lambda y: jnp.sum(jnp.sin(y) + y**2)
    np.testing.assert_allclose(ad.grad(energy)(x), np.cos(x) + 2 * x, atol=1e-14)
    np.testing.assert_allclose(ad.hvp(energy, x, v), (2 - np.sin(x)) * v,
                               atol=1e-14)

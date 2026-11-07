import os
import platform
import time

import numpy as np
import pytest

from aerotvc import ad


def _hand_energy(x, a, b, c):
    return float(np.sum(a * x**2 / 2 + b * x**4 / 4 + c * x))


def test_make_energy_gradient_matches_hand_formula_and_finite_differences():
    rng = np.random.default_rng(20261107)
    x = rng.normal(size=24)
    a, b, c = rng.uniform(0.5, 2.0, size=(3, x.size))
    energy = ad.make_energy(a, b, c)
    gradient = np.asarray(ad.grad(energy)(x))
    expected = a * x + b * x**3 + c
    np.testing.assert_allclose(gradient, expected, rtol=0, atol=1e-12)

    epsilon = 1e-5
    finite_differences = np.empty_like(x)
    for index in range(x.size):
        offset = np.zeros_like(x)
        offset[index] = epsilon
        finite_differences[index] = (
            _hand_energy(x + offset, a, b, c)
            - _hand_energy(x - offset, a, b, c)
        ) / (2 * epsilon)
    np.testing.assert_allclose(gradient, finite_differences,
                               rtol=2e-8, atol=2e-8)


def _hardware_description():
    processor = platform.processor() or platform.machine()
    return (f"{platform.system()} {platform.machine()}, {processor}, "
            f"{os.cpu_count()} logical CPUs")


def _timing_summary(samples):
    return (f"min/median/max="
            f"{min(samples):.1f}/{float(np.median(samples)):.1f}/"
            f"{max(samples):.1f}us")


def test_make_energy_gradient_meets_hand_formula_timing_gate():
    rng = np.random.default_rng(20261107)
    x = rng.normal(size=32_768)
    a, b, c = rng.uniform(0.5, 2.0, size=(3, x.size))
    energy = ad.make_energy(a, b, c)
    gradient_function = ad.grad(energy)
    if ad.BACKEND == "jax":
        import jax

        gradient_function = jax.jit(gradient_function)

    def hand_gradient():
        return a * x + b * x**3 + c

    warmup = gradient_function(x)
    if hasattr(warmup, "block_until_ready"):
        warmup.block_until_ready()
    np.testing.assert_allclose(
        np.asarray(warmup), hand_gradient(), rtol=0, atol=1e-11)
    hand_gradient()

    ad_times = []
    hand_times = []
    for _ in range(3):
        started = time.perf_counter()
        result = gradient_function(x)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        ad_times.append((time.perf_counter() - started) * 1e6)

        started = time.perf_counter()
        hand_gradient()
        hand_times.append((time.perf_counter() - started) * 1e6)

    print(
        f"make_energy gradient benchmark backend={ad.BACKEND}; "
        f"hardware={_hardware_description()}; "
        f"AD microseconds {_timing_summary(ad_times)}; "
        f"hand-formula microseconds {_timing_summary(hand_times)}"
    )
    assert float(np.median(ad_times)) <= 10 * float(np.median(hand_times))

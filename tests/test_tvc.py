import numpy as np
import pytest
from scipy.linalg import expm

from aerotvc.tvc import (
    PID, LQR, PitchModel, PlanarModel, rk4,
)


def integrate(f, state, dt, steps):
    state = np.asarray(state, dtype=float)
    for i in range(steps):
        state = rk4(f, i * dt, state, dt)
    return state


def test_rk4_order():
    errors = [abs(integrate(lambda t, y: y, [1], 1 / n, n)[0] - np.e)
              for n in (10, 20, 40)]
    assert 14 < errors[0] / errors[1] < 18
    assert 14 < errors[1] / errors[2] < 18


def test_pendulum_energy():
    rhs = lambda t, y: np.array([y[1], -9.81 * np.sin(y[0])])
    state = np.array([0.7, 0.0])
    initial = 9.81 * (1 - np.cos(state[0]))
    for i in range(2000):
        state = rk4(rhs, i * 0.01, state, 0.01)
        energy = state[1]**2 / 2 + 9.81 * (1 - np.cos(state[0]))
        assert abs(energy - initial) < 1e-6


def test_constant_gimbal_pitch():
    model = PitchModel(inertia=4, lever_arm=0.5, thrust=20, gimbal_limit=0.2)
    alpha = 20 * np.sin(0.1) * 0.5 / 4
    state = integrate(lambda t, y: model.rhs(t, y, 0.1), [0, 0], 0.01, 200)
    np.testing.assert_allclose(state, [alpha * 2**2 / 2, alpha * 2], atol=1e-12)
    assert model.rhs(0, [0, 0], 10)[1] == pytest.approx(
        20 * np.sin(0.2) * 0.5 / 4)


def test_actuator_lag():
    model = PitchModel(4, 0.5, 20, gimbal_limit=0.2, actuator_tau=0.1)
    state = integrate(lambda t, y: model.rhs(t, y, 1), [0, 0, 0], 0.001, 500)
    assert state[2] == pytest.approx(0.2 * (1 - np.exp(-5)), abs=1e-10)
    assert 0 <= state[2] <= 0.2


def test_lqr_riccati_and_exact_linear_trajectory():
    A = np.array([[0., 1.], [0., 0.]])
    B = np.array([[0.], [2.]])
    Q, R = np.diag([4., 1.]), np.array([[0.5]])
    control = LQR(A, B, Q, R, gimbal_limit=1)
    P = control.P
    residual = A.T @ P + P @ A - P @ B @ np.linalg.solve(R, B.T @ P) + Q
    np.testing.assert_allclose(residual, 0, atol=1e-12)
    initial = np.array([0.01, -0.02])
    result = integrate(lambda t, y: A @ y + B[:, 0] * control.command(y),
                       initial, 0.002, 500)
    np.testing.assert_allclose(result, expm((A - B @ control.K)) @ initial,
                               atol=1e-12)
    assert abs(control.command([100, 100])) <= 1


def test_pid_clamp_limit_and_disturbance_rejection():
    pid = PID(4, 2, 3, gimbal_limit=0.3, integral_limit=0.1)
    for _ in range(100):
        assert abs(pid.command(100, 0, 0.1)) <= 0.3
    assert abs(pid.integral) <= 0.1
    pid.reset()
    assert pid.integral == 0
    pid = PID(4, 2, 3, gimbal_limit=0.3, integral_limit=0.5)
    model = PitchModel(1, 1, 1, gimbal_limit=0.3)
    state = np.zeros(2)
    for i in range(6000):
        delta = pid.command(-state[0], -state[1], 0.005)
        state = rk4(lambda t, y: model.rhs(t, y, delta, disturbance=0.05),
                    i * 0.005, state, 0.005)
    np.testing.assert_allclose(state, 0, atol=1e-5)
    assert np.sin(pid.command(-state[0], -state[1], 0.005)) == pytest.approx(
        -0.05, abs=1e-5)


def test_planar_ballistic_and_thrust():
    model = PlanarModel(mass=2, inertia=4, lever_arm=0.5, thrust=0)
    initial = [0, 10, 3, 4, 0.2, 0]
    final = integrate(lambda t, y: model.rhs(t, y, 0), initial, 0.01, 100)
    np.testing.assert_allclose(final, [3, 10 + 4 - 9.80665 / 2,
                                    3, 4 - 9.80665, 0.2, 0], atol=1e-12)
    model = PlanarModel(2, 4, 0.5, 20, gimbal_limit=0.2)
    rhs = model.rhs(0, [0, 0, 0, 0, 0.4, 0], 0.1)
    np.testing.assert_allclose(rhs, [0, 0, 10 * np.cos(0.5),
                                   10 * np.sin(0.5) - 9.80665, 0,
                                   20 * np.sin(0.1) * 0.5 / 4])


@pytest.mark.parametrize("call", [
    lambda: PitchModel(0, 1, 1),
    lambda: PitchModel(1, 1, 1, actuator_tau=-1),
    lambda: PlanarModel(0, 1, 1, 1),
    lambda: PID(1, 1, 1, gimbal_limit=-1),
    lambda: PID(1, 1, 1).command(1, 1, 0),
    lambda: rk4(lambda t, y: y, 0, [1], -1),
])
def test_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()

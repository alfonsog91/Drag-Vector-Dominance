"""Fixed-step dynamics and small-angle feedback design (SI units, radians)."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_continuous_are

from .nozzle import _positive


def rk4(rhs, t, state, dt):
    _positive(dt, "time step")
    y = np.asarray(state, dtype=float)
    k1 = np.asarray(rhs(t, y))
    k2 = np.asarray(rhs(t + dt / 2, y + dt * k1 / 2))
    k3 = np.asarray(rhs(t + dt / 2, y + dt * k2 / 2))
    k4 = np.asarray(rhs(t + dt, y + dt * k3))
    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


@dataclass
class PitchModel:
    inertia: float
    lever_arm: float
    thrust: float
    gimbal_limit: float = np.pi / 6
    actuator_tau: float = 0.0

    def __post_init__(self):
        _positive(self.inertia, "inertia")
        _positive(self.gimbal_limit, "gimbal limit")
        for name in ("lever_arm", "thrust", "actuator_tau"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def rhs(self, t, state, command, disturbance=0.0):
        y = np.asarray(state, dtype=float)
        if y.shape != ((3,) if self.actuator_tau > 0 else (2,)):
            raise ValueError("pitch state is [theta, omega, optional delta]")
        target = np.clip(command, -self.gimbal_limit, self.gimbal_limit)
        delta = np.clip(y[2], -self.gimbal_limit, self.gimbal_limit) if (
            self.actuator_tau > 0) else target
        result = [y[1], (self.thrust * np.sin(delta) * self.lever_arm
                        + disturbance) / self.inertia]
        if self.actuator_tau > 0:
            result.append((target - y[2]) / self.actuator_tau)
        return np.array(result)


@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    gimbal_limit: float = np.pi / 6
    integral_limit: float = 1.0
    integral: float = 0.0

    def __post_init__(self):
        _positive(self.gimbal_limit, "gimbal limit")
        _positive(self.integral_limit, "integral limit")
        if not np.all(np.isfinite([self.kp, self.ki, self.kd])):
            raise ValueError("PID gains must be finite")

    def reset(self):
        self.integral = 0.0

    def command(self, error, error_rate, dt):
        _positive(dt, "time step")
        self.integral = float(np.clip(self.integral + error * dt,
                                     -self.integral_limit, self.integral_limit))
        return float(np.clip(self.kp * error + self.ki * self.integral
                             + self.kd * error_rate,
                             -self.gimbal_limit, self.gimbal_limit))


class LQR:
    def __init__(self, A, B, Q, R, gimbal_limit=np.pi / 6):
        _positive(gimbal_limit, "gimbal limit")
        A, B, Q, R = (np.asarray(m, dtype=float) for m in (A, B, Q, R))
        if A.shape != (2, 2) or B.shape != (2, 1):
            raise ValueError("LQR expects [theta, omega] and one gimbal input")
        self.P = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.solve(R, B.T @ self.P)
        self.gimbal_limit = gimbal_limit

    def command(self, state):
        y = np.asarray(state, dtype=float)
        if y.shape != (2,):
            raise ValueError("LQR state is [theta, omega]")
        return float(np.clip(-(self.K @ y)[0],
                             -self.gimbal_limit, self.gimbal_limit))


@dataclass
class PlanarModel:
    mass: float
    inertia: float
    lever_arm: float
    thrust: float
    gimbal_limit: float = np.pi / 6
    actuator_tau: float = 0.0
    gravity: float = 9.80665

    def __post_init__(self):
        _positive(self.mass, "mass")
        if not np.isfinite(self.gravity) or self.gravity < 0:
            raise ValueError("gravity must be finite and nonnegative")
        self._pitch = PitchModel(self.inertia, self.lever_arm, self.thrust,
                                 self.gimbal_limit, self.actuator_tau)

    def rhs(self, t, state, command, disturbance=0.0):
        y = np.asarray(state, dtype=float)
        if y.shape != ((7,) if self.actuator_tau > 0 else (6,)):
            raise ValueError("planar state is [x, z, vx, vz, theta, omega, optional delta]")
        pitch = self._pitch.rhs(t, y[4:], command, disturbance)
        delta = np.clip(y[6] if self.actuator_tau > 0 else command,
                        -self.gimbal_limit, self.gimbal_limit)
        angle = y[4] + delta
        return np.concatenate(([y[2], y[3], self.thrust * np.cos(angle) / self.mass,
                                self.thrust * np.sin(angle) / self.mass - self.gravity],
                               pitch))

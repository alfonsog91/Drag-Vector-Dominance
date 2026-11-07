# aerotvc

Python 3.11+ package for idealized nozzle, thrust-vector-control, and convex
edge-flow models. Runtime dependencies are NumPy and SciPy; JAX is optional.
These are **model-level calculations**, not evidence of real aircraft drag
reduction. See [CLAIMS.md](CLAIMS.md).

## Install and test

```sh
python -m pip install ".[test]"
AEROTVC_DISABLE_JAX=1 python run_tests.py
python -m pip install ".[test,jax]"
python run_tests.py
sha256sum -c FROZEN_HASHES.txt
```

`run_tests.py` prints PASS/SKIP/FAIL and propagates pytest's exit status.
Without pytest it reports SKIP and exits 2, never claiming success.
JAX-only tests skip when JAX is missing or explicitly disabled. Both CI jobs
use Python 3.11 and verify the frozen hashes before testing.

Tests were committed before implementation and are not modified afterward.
`FROZEN_HASHES.txt` records SHA-256 for every `aerotvc/*.py` and `tests/*.py`
after both backend suites pass. Numerical oracles are textbook values, closed
forms, finite differences, matrix exponentials, or independent null-space
BFGS—not AD-versus-AD comparisons.

## Models and API

All inputs use SI units; angles are radians. No physical calibration is implied.

### Nozzle

```python
from aerotvc import Nozzle

result = Nozzle(0.01, 0.06, g=1.2, R=300).solve(7e6, 3200, 101325)
print(result.mdot, result.thrust, result.cf, result.isp)
```

`Nozzle(A_t, A_exit, g=1.4, R=287)` assumes a **choked**, steady,
quasi-one-dimensional, isentropic perfect gas with constant heat capacity.
`solve(p0, T0, pa=0)` returns `mach`, `mdot`, `pe`, `te`, `ve`, `thrust`,
`cf`, `isp` (seconds, standard gravity 9.80665 m/s²), and `separation_risk`.
The last flag is only the heuristic `pe/pa < 0.4`, false in vacuum; no
separated-flow correction is applied. Back pressure does not select a
different flow regime: shocks, unchoking, viscosity and heat loss are omitted.

`area_ratio(mach, g)`, `pressure_ratio(mach, g)`, `temperature_ratio(mach, g)`
and `mach_from_area_ratio(ratio, g, branch="supersonic")` are in
`aerotvc.nozzle`. The inverse uses Brent's method; `"subsonic"` is also
supported. The rounded mass-flow coefficient 0.0404 assumes R=287 and g=1.4;
the vacuum limiting Cf values assume zero ambient pressure.

### Thrust-vector dynamics

`aerotvc.tvc.rk4(rhs, t, state, dt)` advances one fixed RK4 step.
`PitchModel(inertia, lever_arm, thrust, gimbal_limit, actuator_tau=0)`
uses `[theta, omega]`, appending `delta` when lag is enabled.
Its `rhs(t, state, command, disturbance=0)` implements
`I*theta'' = thrust*sin(delta)*lever_arm + disturbance` (disturbance is torque).
Positive lag follows `delta'=(clipped_command-delta)/actuator_tau`;
zero lag uses the clipped command directly.

`PID(kp, ki, kd, gimbal_limit, integral_limit)` exposes
`command(error, error_rate, dt)` and `reset()`. Supply the error derivative
explicitly (e.g. `-omega` for a fixed pitch target). The integrator and output
are clamped. `LQR(A, B, Q, R, gimbal_limit)` exposes the Riccati solution `P`,
gain `K` and `command(state)=-K*state` with saturation. It is a continuous-time
small-angle design on `[theta, omega]`; the exact linear oracle applies only
when unsaturated, not to arbitrary nonlinear trajectories.

`PlanarModel(mass, inertia, lever_arm, thrust, gimbal_limit, actuator_tau=0,
gravity=9.80665)` uses `[x, z, vx, vz, theta, omega, optional delta]`.
Pitch is measured from the positive horizontal x-axis; z is upward.
Thrust acts at angle `theta+delta`; positive gimbal gives positive pitch torque.
Mass, inertia, thrust and lever arm are constant; aerodynamic forces are absent.

### Convex edge flow and nozzle coupling

```python
from aerotvc.graph import Graph, nozzle_supply, solve_flow
from aerotvc.ledger import AeroResult

graph = Graph(4, [0, 1, 0, 2], [1, 3, 2, 3])
supply = nozzle_supply(graph, 0, 3, result.mdot)
coefficients = [0.5, 0.5, 2, 2]
flow = solve_flow(graph, supply, a=coefficients)
accounting = AeroResult.from_solution(
    graph, supply, flow, a=coefficients)
print(flow.x, accounting.label, accounting.energy_initial,
      accounting.energy_final, accounting.reduction_pct)
```

`Graph(n, tails, heads)` exposes incidence `B` and Laplacian `L=BB.T`, using
dense arrays for small graphs and CSR sparse matrices when the combined dense
matrices would exceed one million elements. An edge has -1 at its tail and +1
at its head, so `B.T@phi` is head minus tail potential. Signed flows may run
against the stored orientation.
`nozzle_supply(graph, source, sink, mdot)` gives -mdot at source and +mdot
at sink. `solve_flow(graph, supply, a=1, b=0, c=0, tol=1e-10, max_iter=100)`
minimizes `sum(a*x²/2 + b*x⁴/4 + c*x)` subject to `B@x=supply`.
Coefficients are scalars or edge vectors; `a>0`, `b>=0`. Supplies must balance
in every connected component, including isolated nodes. Newton steps use
grounded weighted Laplacians per component and Armijo backtracking.
Nonconvergence raises `RuntimeError`, not a silently accepted partial result.
The result exposes `x`, `energy`, and `iterations`.

`AeroResult.from_solution(...)` wraps a flow result with the energy of the
minimum-norm feasible start, final energy, reduction percentage, and iteration
count. Its label is exactly **model energy reduction**. These are abstract
model-cost quantities, not real drag measurements or reduction claims.

`aerotvc.mesh.icosphere(frequency=32)` builds an edge graph by subdividing and
projecting an icosahedron onto the unit sphere. With `T=frequency²`, it has
`10T+2` nodes and `30T` edges; at frequency 32 that is 10,242 nodes and 30,720
edges. Twelve nodes have degree 5 and all remaining nodes have degree 6.

These abstract costs are not a physical aerodynamic drag law. Quadratic flows
scale linearly with mass flow, while their cost scales quadratically.

### Automatic differentiation

```python
import numpy as np
from aerotvc.ad import make_energy, value_and_grad, hvp

energy = make_energy(np.ones(3), b=0.1)
x = np.array([1., 2., 3.])
value, gradient = value_and_grad(energy)(x)
curvature = hvp(energy, x, np.ones(3))
```

`grad(f)(x)`, `value_and_grad(f)(x)` and `hvp(f, x, v)` use float64 JAX
when importable, unless `AEROTVC_DISABLE_JAX=1` is set before import.
Otherwise `Var` provides NumPy reverse-mode arithmetic (+, -, *, /, constant
powers and full-array `.sum()`), including broadcasting and shared expressions.
HVPs propagate tangents through the reverse pass, not finite differences.
General fallback functions must use these supported `Var` operations, not
arbitrary NumPy ufuncs. JAX functions can use JAX's wider operation set.

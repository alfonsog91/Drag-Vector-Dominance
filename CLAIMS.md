# Scope of claims

All results are **model-level**. This project makes **no claim of real aircraft
drag reduction**, flight readiness, propulsion efficiency improvement, or
experimental validation.

- Nozzle checks cover the ideal choked isentropic perfect-gas equations and
  textbook limits. They do not validate separated flow, shocks, combustion,
  variable gas properties, or an actual engine.
- RK4 convergence, pendulum energy, analytic pitch trajectories, PID rejection
  of a constant modeled torque, and LQR Riccati/exponential comparisons are
  numerical checks. They do not establish stability under arbitrary saturation,
  lag, disturbances, or real vehicle aerodynamics.
- Edge-flow conservation and convex optimality concern an abstract graph.
  The edge energy is a user-specified mathematical cost, not measured drag.
  Passing nozzle mass flow into this graph is a mathematical coupling only.
- AD comparisons use analytic derivatives and finite differences. Correct
  derivatives do not establish physical correctness of the modeled energy.

Test outcomes and interpreter/backend details are reported in the PR and CI.
Skips and failures are reported explicitly; no unexecuted check is claimed
to pass. Frozen hashes detect changes to the recorded files, but are not an
independent security audit or a guarantee of physical validity.

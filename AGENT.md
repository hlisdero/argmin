# AGENT.md

## Purpose

This file is a handoff document for any future AI agent or contributor working on the first experimental interior point method (IPM) implementation for the `argmin` crate. It records the architecture decisions, implementation scope, constraints, expectations, and non-goals that were established during project discussion so work can continue without re-discovering context.

The immediate project goal is **not** to build a production-grade nonlinear optimization solver. The goal is to deliver a **maintainable, working, dense, nalgebra-based first IPM implementation** that can solve smooth, small demonstration problems in time for a seminar presentation. The user has already invested significant time in the idea and needs something concrete, explainable, and demoable.

## Project context

The solver is being developed for the `argmin` crate under an **experimental** feature gate. The implementation is intended to demonstrate that constrained nonlinear optimization can live inside `argmin` in a principled way, while accepting that the first version will be limited in scope and numerical robustness.

The user plans to present the work in a seminar and expects to run **small smooth 2D test problems**. This is important because it justifies several deliberate simplifications that would be unacceptable in a general-purpose solver.

The user explicitly stated that this first version only needs to be:

- maintainable,
- working on simple problems,
- suitable for slides and demonstration,
- honest about being experimental.

The user does **not** need:

- sparse support,
- production robustness,
- handling of degenerate or badly scaled problems,
- advanced globalization,
- advanced linear solvers,
- restoration phases or regularization machinery.

## High-level architecture

### Core architectural decisions

The following decisions are settled and should be treated as project constraints unless the user explicitly changes them:

1. The solver is strictly **nalgebra-based**.
2. The solver is behind the `experimental_ipm` Cargo feature.
3. Backend-specific linear algebra and KKT details belong in backend modules, especially `nalgebra_backend.rs`.
4. A new intermediate module named `iteration.rs` should be introduced to encapsulate one full IPM iteration.
5. A new dedicated solver-specific line search module named `linesearch.rs` should be introduced.
6. `interiorpointmethod.rs` should remain the public solver/orchestration entry point, not the home for all low-level logic.
7. The implementation should support **equality and inequality constraints**.
8. The implementation should be as basic as possible while still being coherent and working.

These decisions are consistent with the existing module exposure and feature-gated experimental solver organization.

### Intended module responsibilities

#### `interiorpointmethod.rs`

Responsibilities:

- public solver type,
- constructor/configuration,
- `Solver` trait implementation,
- `init`, `next_iter`, `terminate` orchestration,
- interaction with `Problem`, `State`, and executor-level logging.

This file may use concrete nalgebra types where needed. However, it should not accumulate all the detailed logic for assembling KKT systems, computing steps, or handling line search internals.

#### `iteration.rs`

Responsibilities:

- represent and execute **one full primal-dual IPM iteration**,
- evaluate all required problem quantities at the current point,
- compute residuals and diagnostics,
- compute/update the barrier parameter `mu`,
- ask the backend to assemble and solve the KKT system,
- split the Newton step into semantic blocks,
- compute the positivity-preserving maximum step length,
- call the line search,
- accept a trial iterate and write back state/logging values.

This module is the critical maintainability boundary. It should prevent `interiorpointmethod.rs` from becoming a giant, fragile control-flow file.

#### `linesearch.rs`

Responsibilities:

- solver-specific backtracking for the IPM,
- work on a **single scalar step length** `alpha`,
- respect positivity of slacks and inequality multipliers,
- accept/reject trial points based on a **barrier objective only**,
- terminate with a clean solver exit on failure.

This should be a custom implementation. The existing generic backtracking line search in argmin is **not** considered a drop-in replacement for this solver because it is built around a generic cost/gradient/search-direction pattern and does not naturally model the coupled primal-dual trial point needed for an IPM.

#### `nalgebra_backend.rs`

Responsibilities:

- dense KKT dimension validation,
- dense KKT matrix assembly,
- residual assembly,
- dense LU solve through the `argmin-math` LU solver trait,
- splitting the solved Newton step into semantic blocks.

This file already has the right scope and should remain focused on low-level dense nalgebra logic. It is not the place for line search, barrier updates, or high-level solver control flow.

## Dense backend semantics

The dense KKT backend uses the fixed semantic block ordering

```text
(x, s, lambda_ineq, lambda_eq)
```

where:

- `x` are primal variables,
- `s` are slack variables,
- `lambda_ineq` are inequality multipliers,
- `lambda_eq` are equality multipliers.

The current dense KKT matrix form is documented in the backend as

```text
[ H 0 J_ineq^T J_eq^T ]
[ 0 diag(z) diag(s) 0 ]
[ J_ineq I 0 0 ]
[ J_eq 0 0 0 ]
```

and the residual stack is documented as

```text
[ grad f(x) + J_ineq^T z + J_eq^T lambda_eq ]
[ s .* z - mu * 1 ]
[ g(x) + s ]
[ h(x) ]
```

with Newton step obtained from

```text
KKT * delta = -residuals
```

Any future code should preserve this semantic ordering consistently in comments, tests, documentation, and APIs.

## State and diagnostics

The solver state is based on `NonLinearProgramState`, which already contains or is intended to contain the major IPM-relevant quantities such as:

- parameter vector,
- slacks,
- gradient,
- Lagrangian Hessian,
- equality constraints,
- inequality constraints,
- equality Jacobian,
- inequality Jacobian,
- equality multipliers,
- inequality multipliers,
- barrier parameter `mu`,
- `alpha_primal`,
- `alpha_dual`,
- primal infeasibility,
- dual infeasibility,
- complementarity infeasibility,
- best parameter / best slacks / best cost.

The state is broad and mutable, so future contributors should be careful not to spread iteration-only invariants everywhere. The earlier design discussion identified this as a maintainability risk. `iteration.rs` should be used to concentrate transient iteration logic rather than letting the public state object become the only place where solver reasoning happens.

### Accepted convention for v1 step lengths

Although the state contains separate `alpha_primal` and `alpha_dual` fields, the user explicitly approved using the **same accepted scalar step length** for both in version 1. Therefore:

- the implementation may compute one accepted `alpha`,
- `alpha_primal` and `alpha_dual` should both be set to that same number,
- the structure should still preserve the two fields for future extensibility.

This was a deliberate simplification to reduce implementation complexity while keeping the public state compatible with a later split-step design.

### Logging requirements

The solver should return rich `KV` information from iterations because the user wants informative logs for a seminar presentation. At minimum, logging should include any values that are practical and meaningful, especially:

- `mu`,
- primal infeasibility (`inf_pr`),
- dual infeasibility (`inf_du`),
- complementarity infeasibility (`compl_inf`),
- accepted step length `alpha` (or both alpha fields),
- possibly barrier objective value,
- possibly cost if already available.

The motivation is not just debugging; the logs are part of the presentation value. The more interpretable per-iteration information is exposed, the better. This fits naturally with the existing executor/observer model in argmin.

## Problem scope

### In scope

The first implementation is expected to support:

- dense problems only,
- smooth nonlinear objective,
- equality constraints,
- inequality constraints,
- mixed equality + inequality cases,
- inequality-only and equality-only cases,
- small demonstration problems, especially 2D smooth examples.

The solver should work on simple, well-behaved problems suitable for demos.

### Explicitly out of scope

These are intentionally excluded for v1:

- sparse matrices,
- sparse linear algebra backends,
- iterative linear solvers,
- regularization strategies,
- inertia correction,
- restoration phases,
- filter methods,
- merit functions,
- second-order correction,
- handling degenerate problems,
- handling badly scaled problems,
- production-grade robustness,
- Cholesky-based KKT solve work unless the user explicitly changes direction.

The user stated that dense demos are the entire target and that complexity growth from advanced safeguards is unacceptable for the current milestone.

## Line search design

### Why the generic backtracking line search is not a drop-in

The existing `backtracking.rs` implementation in argmin is designed around a standard line-search pattern with an initial parameter vector, initial cost, initial gradient, a search direction, and an Armijo-like condition. That model aligns naturally with unconstrained or simply parameterized descent methods, but not with a primal-dual interior point step where the trial iterate is a coupled object containing:

- `x`,
- `s`,
- `lambda_eq`,
- `lambda_ineq`.

The IPM also requires positivity handling for `s` and inequality multipliers before acceptance, and acceptance is based on a barrier objective rather than the raw objective function. Therefore the generic line search is not considered architecturally appropriate as a drop-in.

### Agreed v1 line search behavior

The user explicitly approved the following design:

- custom `linesearch.rs`,
- backtracking on a **single scalar** `alpha`,
- barrier objective only,
- no merit function,
- no filter,
- no second-order correction,
- no fallback tiny diagonal regularization,
- line-search failure should produce a clean solver exit.

### Barrier objective choice

The user chose the simplest acceptance criterion: **barrier objective only**.

That means the line search should be driven by something of the form

```text
phi_mu(x, s) = f(x) - mu * sum(log(s_i))
```

subject to trial slacks remaining strictly positive. If inequality multipliers also require strict positivity in the implementation, the trial step must respect that as well.

No residual penalty terms or external merit terms should be introduced in v1 unless the user explicitly changes direction.

## Barrier parameter update

The agreed design choice is:

- use a **fixed sigma** update strategy,
- prefer the simplest, easiest-to-explain barrier schedule,
- do not introduce adaptive sophistication unless necessary.

A reasonable simple form discussed conceptually was the usual complementarity-based pattern

```text
mu = sigma * (s^T lambda_ineq) / m
```

with fixed `sigma`, where `m` is the number of inequality constraints.

If there are no inequality constraints, the implementation will need a coherent convention; future contributors should keep that edge case explicit and documented rather than implicit.

## Linear algebra and solver assumptions

The user already added LU solver traits to `argmin-math` and has a nalgebra-backed LU solver implementation available. That is the intended linear solver for this first version.

Important consequences:

- LU is the only solver path that should be assumed for v1.
- No time should be spent trying to improve linear solver infrastructure.
- Cholesky exists in the project but is not expected to be useful for this KKT use case under current assumptions.
- No diagonal regularization retry should be added.
- On KKT solve failure, fail clearly rather than growing the architecture.

This is a conscious tradeoff: less robustness, more maintainability and delivery speed.

## Numerical realism and demo strategy

Future agents must understand that the target is **seminar-quality demonstration**, not broad solver coverage.

This has several implications:

- Do not optimize for adversarial cases.
- Do not spend time adding mechanisms for singular/ill-conditioned edge cases unless the user changes scope.
- Pick simple, smooth, visually explainable test problems.
- Favor traceable diagnostics and understandable behavior over sophistication.
- Keep implementation small enough that the user can explain it on slides.

The user explicitly said that if a maintainable working solution cannot be delivered quickly, they may abandon the idea of extending argmin. That makes schedule pressure a real project constraint, not just a preference.

## Testing expectations

The user explicitly asked for:

- function documentation,
- unit tests,
- documentation/comments for tests,
- especially explanations for numeric values in tests involving specific problems.

### Testing philosophy for this project

Tests should not be mysterious. If a test solves a particular small NLP or checks a KKT solve numerically, the test should explain:

- what the problem is,
- what the expected solution or iterate behavior is,
- why the asserted values are correct,
- why the chosen tolerances are reasonable.

The existing backend tests are a good model in spirit because they explicitly validate dimensions and assembled structures instead of burying meaning. Future tests for `iteration.rs`, `linesearch.rs`, and solver-level behavior should follow the same standard of interpretability.

### Recommended test categories

A future agent implementing the solver should aim for tests in categories like:

1. **Backend structure tests**
   - KKT dimensions and offsets,
   - validation failures on mismatched inputs,
   - matrix assembly correctness,
   - residual assembly correctness,
   - step splitting correctness.

2. **Line search tests**
   - positivity cap computation,
   - barrier objective decrease acceptance,
   - rejection of nonpositive trial slacks or multipliers,
   - failure behavior when no acceptable step exists.

3. **Iteration tests**
   - correct update of `mu`,
   - state diagnostics updated consistently,
   - accepted iterate remains feasible with respect to positivity,
   - both alpha fields set consistently from the single accepted scalar.

4. **Small end-to-end solver tests**
   - inequality-only toy problem,
   - equality-only toy problem,
   - mixed-constraint toy problem if time allows,
   - convergence to a known small solution with documented tolerances.

If time is limited, prioritize tests that protect architecture boundaries and demo-critical behavior over exhaustive coverage.

## Documentation expectations

The user asked specifically for documented functions and documented tests. Therefore:

- public and nontrivial internal functions should have concise Rust doc comments or explanatory comments where appropriate,
- new modules (`iteration.rs`, `linesearch.rs`) should begin with clear module-level documentation,
- tests should include comments that help a newcomer interpret the setup and the asserted numeric values.

The code should remain readable to a future contributor or presenter who needs to explain it quickly.

## Maintainability principles

This section captures the main maintainability lessons from the discussion.

### Keep `interiorpointmethod.rs` thin

Do not let it become the place where all the solver math lives. It should orchestrate, not micromanage.

### Use `iteration.rs` as the main complexity boundary

One IPM iteration is the natural unit of reasoning. If the implementation starts to mix residual assembly, linear solve, line search, state mutation, and logging in multiple places, refactor back toward a single iteration-centric boundary.

### Keep backend code backend-specific

If code only makes sense for dense nalgebra matrices or the current KKT layout, keep it in `nalgebra_backend.rs` rather than leaking it upward.

### Preserve future extensibility without implementing it now

Examples:

- keep `alpha_primal` and `alpha_dual` even though both are equal in v1,
- keep backend semantic ordering explicit,
- keep line search separate even though simple,
- keep module boundaries clean even if there is only one backend.

### Avoid premature abstraction

The user does **not** want complexity growth. Do not build speculative generic layers for sparse backends, alternative KKT formulations, or advanced globalization unless the code truly needs them now.

## Non-goals and anti-patterns

A future agent should **not** do the following unless explicitly instructed:

- Do not add sparse support.
- Do not add regularization “just in case.”
- Do not add filter methods or merit functions.
- Do not add second-order correction.
- Do not split accepted line-search parameters into separate primal/dual logic in v1.
- Do not chase production-grade numerical behavior.
- Do not replace LU with a more advanced solver path.
- Do not over-generalize the architecture around future possibilities.
- Do not bury the seminar/demo context; it is central to understanding the right tradeoffs.

## Practical implementation checklist

A future agent starting actual implementation work should assume the following checklist:

1. Add `iteration.rs` module.
2. Add `linesearch.rs` module.
3. Update the solver module `mod.rs` to expose/include them as needed.
4. Keep `nalgebra_backend.rs` focused on dense KKT details.
5. Refactor `interiorpointmethod.rs` so it orchestrates instead of containing all detail.
6. Use the existing `NonLinearProgramState` fields consistently.
7. Use one accepted scalar step length and write it into both alpha fields.
8. Implement barrier-only backtracking in `linesearch.rs`.
9. Use fixed `sigma` for `mu` updates.
10. Fail cleanly on line-search failure or KKT solve failure.
11. Add interpretable `KV` logging.
12. Add documented unit tests.
13. Prefer small smooth examples that make good seminar demonstrations.

## Summary for the next agent

The project is a first experimental dense interior point solver for `argmin`, gated behind `experimental_ipm`, using nalgebra and LU solve only. The real objective is to produce a maintainable, understandable, working solver for small smooth demo problems today, not a production solver.

The most important design decision is to isolate one solver iteration in `iteration.rs` and a custom barrier-based backtracking line search in `linesearch.rs`, while keeping dense KKT details in `nalgebra_backend.rs` and leaving `interiorpointmethod.rs` as the public orchestration layer.

When in doubt, choose the simpler design that preserves clarity, makes the math explainable on slides, and avoids growing complexity. That is not a compromise accidentally forced by time; it is the project’s explicit success criterion.

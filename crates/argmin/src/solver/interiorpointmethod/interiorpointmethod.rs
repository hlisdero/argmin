// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Primal-dual interior point method for basic nonlinear programs.
//!
//! This implementation is intentionally narrow in scope. It provides a dense,
//! nalgebra-backed primal-dual barrier method that can handle equality and
//! inequality constraints with slack variables. The algorithm is deliberately
//! simple:
//!
//! - dense KKT assembly and solve,
//! - fraction-to-boundary step size,
//! - backtracking line search on the barrier objective,
//! - convergence checks based on primal infeasibility, dual infeasibility, and
//!   complementarity.
//!
//! Advanced features such as second-order correction, filter line search,
//! restoration phases, inertia correction, and sparse linear algebra are out of
//! scope for this implementation.

use argmin_math::ArgminLuSolve;
use nalgebra::{ClosedAddAssign, ClosedMulAssign, ClosedSubAssign, DMatrix, DVector};

use super::iteration::{AcceptedIteration, InteriorPointIteration, IpmState, IterationEvaluation};
use super::linesearch::InteriorPointLineSearch;
use crate::core::{
    CostFunction, EqualityConstraint, EqualityConstraintJacobian, Error, Gradient,
    InequalityConstraint, InequalityConstraintJacobian, LagrangianHessian, Problem, Solver, State,
    TerminationReason, TerminationStatus, KV,
};

/// Basic primal-dual interior point method.
///
/// The solver is deliberately concrete in this first version: dense nalgebra
/// vectors/matrices with `f64` scalars.
#[derive(Clone, Debug)]
pub struct InteriorPointMethod {
    tol_primal: f64,
    tol_dual: f64,
    tol_complementarity: f64,
    mu0: f64,
    mu_min: f64,
    barrier_reduction: f64,
    fraction_to_boundary: f64,
    max_backtracking_steps: u64,
    iteration: InteriorPointIteration<f64>,
}

impl Default for InteriorPointMethod {
    fn default() -> Self {
        let line_search = InteriorPointLineSearch::new()
            .with_fraction_to_boundary(0.995)
            .expect("valid default fraction-to-boundary")
            .with_max_iters(16)
            .expect("valid default max backtracking steps");

        let iteration = InteriorPointIteration::new(line_search)
            .with_barrier_parameter_factor(0.2)
            .expect("valid default barrier reduction")
            .with_barrier_parameter_floor(1e-8)
            .expect("valid default barrier floor");

        Self {
            tol_primal: 1e-8,
            tol_dual: 1e-8,
            tol_complementarity: 1e-8,
            mu0: 1e-1,
            mu_min: 1e-8,
            barrier_reduction: 0.2,
            fraction_to_boundary: 0.995,
            max_backtracking_steps: 16,
            iteration,
        }
    }
}

impl InteriorPointMethod {
    /// Create a new interior point method solver with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn rebuild_iteration(&mut self) {
        let line_search = InteriorPointLineSearch::new()
            .with_fraction_to_boundary(self.fraction_to_boundary)
            .expect("fraction-to-boundary must satisfy 0 < tau < 1")
            .with_max_iters(self.max_backtracking_steps)
            .expect("max backtracking steps must be at least 1");

        self.iteration = InteriorPointIteration::new(line_search)
            .with_barrier_parameter_factor(self.barrier_reduction)
            .expect("barrier reduction must satisfy 0 < factor <= 1")
            .with_barrier_parameter_floor(self.mu_min)
            .expect("barrier floor must be > 0");
    }

    /// Set a common tolerance for primal infeasibility, dual infeasibility, and
    /// complementarity.
    #[must_use]
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol_primal = tol;
        self.tol_dual = tol;
        self.tol_complementarity = tol;
        self
    }

    /// Set the primal infeasibility tolerance.
    #[must_use]
    pub fn with_primal_tolerance(mut self, tol: f64) -> Self {
        self.tol_primal = tol;
        self
    }

    /// Set the dual infeasibility tolerance.
    #[must_use]
    pub fn with_dual_tolerance(mut self, tol: f64) -> Self {
        self.tol_dual = tol;
        self
    }

    /// Set the complementarity tolerance.
    #[must_use]
    pub fn with_complementarity_tolerance(mut self, tol: f64) -> Self {
        self.tol_complementarity = tol;
        self
    }

    /// Set the initial barrier parameter.
    #[must_use]
    pub fn with_mu0(mut self, mu0: f64) -> Self {
        self.mu0 = mu0;
        self
    }

    /// Set the minimum barrier parameter.
    #[must_use]
    pub fn with_mu_min(mut self, mu_min: f64) -> Self {
        self.mu_min = mu_min;
        self.rebuild_iteration();
        self
    }

    /// Set the monotone barrier reduction factor.
    #[must_use]
    pub fn with_barrier_reduction(mut self, factor: f64) -> Self {
        self.barrier_reduction = factor;
        self.rebuild_iteration();
        self
    }

    /// Set the fraction-to-boundary factor used to keep slacks and inequality
    /// multipliers strictly positive.
    #[must_use]
    pub fn with_fraction_to_boundary(mut self, tau: f64) -> Self {
        self.fraction_to_boundary = tau;
        self.rebuild_iteration();
        self
    }

    /// Set the maximum number of positivity backtracking steps.
    #[must_use]
    pub fn with_max_backtracking_steps(mut self, steps: u64) -> Self {
        self.max_backtracking_steps = steps;
        self.rebuild_iteration();
        self
    }

    fn evaluate_iterate<O>(
        problem: &mut Problem<O>,
        x: &DVector<f64>,
        lambda_eq: &DVector<f64>,
        lambda_ineq: &DVector<f64>,
    ) -> Result<IterationEvaluation<f64>, Error>
    where
        O: CostFunction<Param = DVector<f64>, Output = f64>
            + Gradient<Param = DVector<f64>, Gradient = DVector<f64>>
            + EqualityConstraint<Param = DVector<f64>, Output = DVector<f64>>
            + InequalityConstraint<Param = DVector<f64>, Output = DVector<f64>>
            + EqualityConstraintJacobian<Param = DVector<f64>, Jacobian = DMatrix<f64>>
            + InequalityConstraintJacobian<Param = DVector<f64>, Jacobian = DMatrix<f64>>
            + LagrangianHessian<
                Param = DVector<f64>,
                MultipliersEq = DVector<f64>,
                MultipliersIneq = DVector<f64>,
                Hessian = DMatrix<f64>,
            >,
    {
        let cost = problem.cost(x)?;
        let gradient = problem.gradient(x)?;
        let equality_constraints = problem.equality_constraint(x)?;
        let inequality_constraints = problem.inequality_constraint(x)?;
        let equality_constraint_jacobian = problem.equality_constraint_jacobian(x)?;
        let inequality_constraint_jacobian = problem.inequality_constraint_jacobian(x)?;
        let lagrangian_hessian = problem.lagrangian_hessian(x, lambda_eq, lambda_ineq)?;

        Ok(IterationEvaluation {
            cost,
            gradient,
            lagrangian_hessian,
            equality_constraints,
            inequality_constraints,
            equality_constraint_jacobian,
            inequality_constraint_jacobian,
        })
    }

    #[inline]
    fn converged(&self, inf_pr: f64, inf_du: f64, compl_inf: f64) -> bool {
        inf_pr <= self.tol_primal
            && inf_du <= self.tol_dual
            && compl_inf <= self.tol_complementarity
    }

    fn init_slacks(
        mut state: IpmState<f64>,
        inequality_constraints: &DVector<f64>,
    ) -> Result<(IpmState<f64>, DVector<f64>), Error> {
        let nineq = inequality_constraints.len();

        let slacks = match state.get_slacks().cloned() {
            Some(s) => {
                if s.len() != nineq {
                    return Err(argmin_error!(
                        InvalidParameter,
                        "InteriorPointMethod: initial slacks have incompatible dimension."
                    ));
                }
                if s.iter().any(|v| *v <= 0.0) {
                    return Err(argmin_error!(
                        InvalidParameter,
                        "InteriorPointMethod: initial slacks must be strictly positive."
                    ));
                }
                s
            }
            None => {
                let mut s = DVector::zeros(nineq);
                for i in 0..nineq {
                    s[i] = (-inequality_constraints[i]).max(1.0);
                }
                state = state.slacks(s.clone());
                s
            }
        };

        Ok((state, slacks))
    }

    fn init_lambda_eq(
        mut state: IpmState<f64>,
        neq: usize,
    ) -> Result<(IpmState<f64>, DVector<f64>), Error> {
        let lambda_eq = match state.get_lambda_eq().cloned() {
            Some(v) => {
                if v.len() != neq {
                    return Err(argmin_error!(
                        InvalidParameter,
                        "InteriorPointMethod: initial equality multipliers have incompatible dimension."
                    ));
                }
                v
            }
            None => {
                let v = DVector::zeros(neq);
                state = state.lambda_eq(v.clone());
                v
            }
        };

        Ok((state, lambda_eq))
    }

    fn init_lambda_ineq(
        mut state: IpmState<f64>,
        nineq: usize,
    ) -> Result<(IpmState<f64>, DVector<f64>), Error> {
        let lambda_ineq = match state.get_lambda_ineq().cloned() {
            Some(v) => {
                if v.len() != nineq {
                    return Err(argmin_error!(
                        InvalidParameter,
                        "InteriorPointMethod: initial inequality multipliers have incompatible dimension."
                    ));
                }
                if v.iter().any(|z| *z <= 0.0) {
                    return Err(argmin_error!(
                        InvalidParameter,
                        "InteriorPointMethod: initial inequality multipliers must be strictly positive."
                    ));
                }
                v
            }
            None => {
                let v = DVector::from_element(nineq, 1.0);
                state = state.lambda_ineq(v.clone());
                v
            }
        };

        Ok((state, lambda_ineq))
    }

    fn initialize_primal_dual(
        &self,
        mut state: IpmState<f64>,
        inequality_constraints: &DVector<f64>,
        equality_constraints: &DVector<f64>,
    ) -> Result<(IpmState<f64>, DVector<f64>, DVector<f64>, DVector<f64>), Error> {
        let nineq = inequality_constraints.len();
        let neq = equality_constraints.len();

        let (new_state, slacks) = Self::init_slacks(state, inequality_constraints)?;
        state = new_state;

        let (new_state, lambda_eq) = Self::init_lambda_eq(state, neq)?;
        state = new_state;

        let (new_state, lambda_ineq) = Self::init_lambda_ineq(state, nineq)?;
        state = new_state;

        let mu = state.get_mu().copied().unwrap_or(self.mu0).max(self.mu_min);
        state = state.mu(mu);

        Ok((state, slacks, lambda_eq, lambda_ineq))
    }
}

impl<O> Solver<O, IpmState<f64>> for InteriorPointMethod
where
    O: Clone
        + CostFunction<Param = DVector<f64>, Output = f64>
        + Gradient<Param = DVector<f64>, Gradient = DVector<f64>>
        + EqualityConstraint<Param = DVector<f64>, Output = DVector<f64>>
        + InequalityConstraint<Param = DVector<f64>, Output = DVector<f64>>
        + EqualityConstraintJacobian<Param = DVector<f64>, Jacobian = DMatrix<f64>>
        + InequalityConstraintJacobian<Param = DVector<f64>, Jacobian = DMatrix<f64>>
        + LagrangianHessian<
            Param = DVector<f64>,
            MultipliersEq = DVector<f64>,
            MultipliersIneq = DVector<f64>,
            Hessian = DMatrix<f64>,
        >,
    f64: ClosedAddAssign + ClosedSubAssign + ClosedMulAssign,
    DMatrix<f64>: ArgminLuSolve<DVector<f64>>,
{
    fn name(&self) -> &str {
        "InteriorPointMethod"
    }

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        state: IpmState<f64>,
    ) -> Result<(IpmState<f64>, Option<KV>), Error> {
        let x = state
            .get_param()
            .cloned()
            .ok_or_else(argmin_error_closure!(
                InvalidParameter,
                "InteriorPointMethod: missing initial parameter vector."
            ))?;

        let bootstrap_eval =
            Self::evaluate_iterate(problem, &x, &DVector::zeros(0), &DVector::zeros(0))?;

        let (state, s, lambda_eq, lambda_ineq) = self.initialize_primal_dual(
            state,
            &bootstrap_eval.inequality_constraints,
            &bootstrap_eval.equality_constraints,
        )?;

        let evaluation = Self::evaluate_iterate(problem, &x, &lambda_eq, &lambda_ineq)?;
        let context = self
            .iteration
            .build_context(x, s, lambda_eq, lambda_ineq, evaluation)?;

        let accepted = AcceptedIteration {
            next_x: context.x.clone(),
            next_s: context.s.clone(),
            next_lambda_eq: context.lambda_eq.clone(),
            next_lambda_ineq: context.lambda_ineq.clone(),
            alpha_primal: 0.0,
            alpha_dual: 0.0,
            barrier_cost: context.evaluation.cost,
        };

        let kv = self.iteration.kv(&context, &accepted);
        let state = self.iteration.write_state(state, accepted, context);

        Ok((state, Some(kv)))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        state: IpmState<f64>,
    ) -> Result<(IpmState<f64>, Option<KV>), Error> {
        let x = state
            .get_param()
            .cloned()
            .ok_or_else(argmin_error_closure!(
                PotentialBug,
                "InteriorPointMethod: missing current parameter vector."
            ))?;
        let s = state
            .get_slacks()
            .cloned()
            .ok_or_else(argmin_error_closure!(
                PotentialBug,
                "InteriorPointMethod: missing current slacks."
            ))?;
        let lambda_eq = state
            .get_lambda_eq()
            .cloned()
            .ok_or_else(argmin_error_closure!(
                PotentialBug,
                "InteriorPointMethod: missing current equality multipliers."
            ))?;
        let lambda_ineq = state
            .get_lambda_ineq()
            .cloned()
            .ok_or_else(argmin_error_closure!(
                PotentialBug,
                "InteriorPointMethod: missing current inequality multipliers."
            ))?;

        let evaluation = Self::evaluate_iterate(problem, &x, &lambda_eq, &lambda_ineq)?;
        let context = self
            .iteration
            .build_context(x, s, lambda_eq, lambda_ineq, evaluation)?;

        if self.converged(
            context.residuals.inf_pr,
            context.residuals.inf_du,
            context.residuals.compl_inf,
        ) {
            let accepted = AcceptedIteration {
                next_x: context.x.clone(),
                next_s: context.s.clone(),
                next_lambda_eq: context.lambda_eq.clone(),
                next_lambda_ineq: context.lambda_ineq.clone(),
                alpha_primal: 0.0,
                alpha_dual: 0.0,
                barrier_cost: context.evaluation.cost,
            };

            let mut next_state = self.iteration.write_state(state, accepted, context);
            next_state.termination_status =
                TerminationStatus::Terminated(TerminationReason::SolverConverged);
            return Ok((next_state, None));
        }

        let accepted = self
            .iteration
            .compute_step(&context, move |trial_x| problem.cost(trial_x))?;

        let kv = self.iteration.kv(&context, &accepted);
        let next_state = self.iteration.write_state(state, accepted, context);

        Ok((next_state, Some(kv)))
    }

    fn terminate(&mut self, state: &IpmState<f64>) -> TerminationStatus {
        if state.termination_status != TerminationStatus::NotTerminated {
            return state.termination_status.clone();
        }

        TerminationStatus::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Executor, State};
    use nalgebra::{dmatrix, dvector};

    #[derive(Clone)]
    struct TinyProblem;

    impl CostFunction for TinyProblem {
        type Param = DVector<f64>;
        type Output = f64;

        fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            Ok((p[0] - 1.0).powi(2) + (p[1] - 2.0).powi(2))
        }
    }

    impl Gradient for TinyProblem {
        type Param = DVector<f64>;
        type Gradient = DVector<f64>;

        fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
            Ok(dvector![2.0 * (p[0] - 1.0), 2.0 * (p[1] - 2.0)])
        }
    }

    impl EqualityConstraint for TinyProblem {
        type Param = DVector<f64>;
        type Output = DVector<f64>;

        fn equality_constraint(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            Ok(dvector![p[0] + p[1] - 3.0])
        }
    }

    impl InequalityConstraint for TinyProblem {
        type Param = DVector<f64>;
        type Output = DVector<f64>;

        fn inequality_constraint(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            Ok(dvector![-p[0]])
        }
    }

    impl EqualityConstraintJacobian for TinyProblem {
        type Param = DVector<f64>;
        type Jacobian = DMatrix<f64>;

        fn equality_constraint_jacobian(&self, _: &Self::Param) -> Result<Self::Jacobian, Error> {
            Ok(dmatrix![1.0, 1.0])
        }
    }

    impl InequalityConstraintJacobian for TinyProblem {
        type Param = DVector<f64>;
        type Jacobian = DMatrix<f64>;

        fn inequality_constraint_jacobian(&self, _: &Self::Param) -> Result<Self::Jacobian, Error> {
            Ok(dmatrix![-1.0, 0.0])
        }
    }

    impl LagrangianHessian for TinyProblem {
        type Param = DVector<f64>;
        type MultipliersEq = DVector<f64>;
        type MultipliersIneq = DVector<f64>;
        type Hessian = DMatrix<f64>;

        fn lagrangian_hessian(
            &self,
            _: &Self::Param,
            _: &Self::MultipliersEq,
            _: &Self::MultipliersIneq,
        ) -> Result<Self::Hessian, Error> {
            Ok(DMatrix::identity(2, 2) * 2.0)
        }
    }

    #[test]
    fn new_has_reasonable_defaults() {
        let solver = InteriorPointMethod::new();
        assert!(solver.tol_primal > 0.0);
        assert!(solver.tol_dual > 0.0);
        assert!(solver.tol_complementarity > 0.0);
        assert!(solver.mu0 > 0.0);
        assert!(solver.mu_min > 0.0);
        assert!(solver.fraction_to_boundary > 0.0);
        assert!(solver.fraction_to_boundary < 1.0);
    }

    #[test]
    fn init_populates_slacks_and_multipliers_when_missing() {
        let mut solver = InteriorPointMethod::new();
        let mut problem = Problem::new(TinyProblem);
        let state = IpmState::<f64>::new().param(dvector![0.5, 2.5]);

        let (state, _kv) = solver.init(&mut problem, state).unwrap();

        assert!(state.get_slacks().is_some());
        assert!(state.get_lambda_eq().is_some());
        assert!(state.get_lambda_ineq().is_some());
        assert!(state.get_mu().is_some());
    }

    #[test]
    fn solver_runs_one_iteration() {
        let solver = InteriorPointMethod::new();
        let problem = TinyProblem;
        let init = IpmState::<f64>::new()
            .param(dvector![0.5, 2.5])
            .max_iters(5);

        let result = Executor::new(problem, solver)
            .configure(|state| state.param(init.get_param().unwrap().clone()).max_iters(5))
            .run();

        assert!(result.is_ok());
    }
}

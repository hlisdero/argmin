// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! One-iteration data structures and helper routines for the experimental
//! nalgebra-based interior point method.
//!
//! This module intentionally sits between the high-level solver orchestration and
//! the low-level KKT backend. The goal is to keep a single IPM iteration
//! explicit and testable:
//!
//! 1. Evaluate the nonlinear program at the current iterate.
//! 2. Form primal/dual residuals and a barrier parameter.
//! 3. Ask the backend for a Newton step.
//! 4. Compute a coupled primal/dual trial step via line search.
//! 5. Write accepted values back to `NonLinearProgramState`.
//!
//! Version 1 is deliberately basic:
//! - nalgebra only
//! - exact derivatives assumed
//! - no merit function
//! - no filter
//! - no second-order correction
//! - a single accepted step length is used for both primal and dual variables

use std::ops::{AddAssign, MulAssign, SubAssign};

use crate::core::{ArgminFloat, Error, NonLinearProgramState, State, KV};
use argmin_math::ArgminLuSolve;
use nalgebra::{ClosedAddAssign, ClosedMulAssign, ClosedSubAssign, DMatrix, DVector};

use super::linesearch::{InteriorPointLineSearch, LineSearchResult};
use super::nalgebra_backend::NalgebraKKTSystem;

/// Concrete state type used by the nalgebra-based interior point method.
pub(super) type IpmState<F> = NonLinearProgramState<
    DVector<F>,
    DVector<F>,
    DMatrix<F>,
    F,
    DVector<F>,
    DVector<F>,
    DVector<F>,
    DMatrix<F>,
    DMatrix<F>,
    DVector<F>,
    DVector<F>,
>;

/// Nonlinear-program quantities evaluated at one iterate.
#[derive(Clone, Debug)]
pub(super) struct IterationEvaluation<F>
where
    F: ArgminFloat,
{
    pub cost: F,
    pub gradient: DVector<F>,
    pub lagrangian_hessian: DMatrix<F>,
    pub equality_constraints: DVector<F>,
    pub inequality_constraints: DVector<F>,
    pub equality_constraint_jacobian: DMatrix<F>,
    pub inequality_constraint_jacobian: DMatrix<F>,
}

/// Residuals and derived scalar diagnostics for one IPM iteration.
#[derive(Clone, Debug)]
pub(super) struct IterationResiduals<F>
where
    F: ArgminFloat + MulAssign + SubAssign + AddAssign,
{
    /// Dual residual:
    ///
    /// \[
    /// \nabla f(x) + J_{eq}(x)^T \lambda_{eq} + J_{ineq}(x)^T \lambda_{ineq}
    /// \]
    pub dual: DVector<F>,
    /// Equality residual:
    ///
    /// \[
    /// c_{eq}(x)
    /// \]
    pub equality: DVector<F>,
    /// Inequality residual in slack form:
    ///
    /// \[
    /// c_{ineq}(x) + s
    /// \]
    pub inequality: DVector<F>,
    /// Complementarity residual:
    ///
    /// \[
    /// S \Lambda_{ineq} e - \mu e
    /// \]
    pub complementarity: DVector<F>,
    /// Infinity norm of primal infeasibility.
    pub inf_pr: F,
    /// Infinity norm of dual infeasibility.
    pub inf_du: F,
    /// Infinity norm of complementarity residual.
    pub compl_inf: F,
}

/// All data required to compute one Newton step.
#[derive(Clone, Debug)]
pub(super) struct IterationContext<F>
where
    F: ArgminFloat + MulAssign + SubAssign + AddAssign,
{
    pub x: DVector<F>,
    pub s: DVector<F>,
    pub lambda_eq: DVector<F>,
    pub lambda_ineq: DVector<F>,
    pub mu: F,
    pub evaluation: IterationEvaluation<F>,
    pub residuals: IterationResiduals<F>,
}

/// Accepted step data from a single IPM iteration.
#[derive(Clone, Debug)]
pub(super) struct AcceptedIteration<F>
where
    F: ArgminFloat + MulAssign + SubAssign + AddAssign,
{
    pub next_x: DVector<F>,
    pub next_s: DVector<F>,
    pub next_lambda_eq: DVector<F>,
    pub next_lambda_ineq: DVector<F>,
    pub alpha_primal: F,
    pub alpha_dual: F,
    pub barrier_cost: F,
}

/// High-level helper which owns the backend and line search for one basic IPM
/// iteration.
#[derive(Clone, Debug)]
pub(super) struct InteriorPointIteration<F>
where
    F: ArgminFloat + MulAssign + SubAssign + AddAssign,
{
    line_search: InteriorPointLineSearch<F>,
    barrier_parameter_factor: F,
    barrier_parameter_floor: F,
}

impl<F> InteriorPointIteration<F>
where
    F: ArgminFloat
        + MulAssign
        + SubAssign
        + AddAssign
        + ClosedAddAssign
        + ClosedSubAssign
        + ClosedMulAssign,
    DMatrix<F>: ArgminLuSolve<DVector<F>>,
{
    /// Construct a new iteration helper from the solver-specific line search.
    pub(super) fn new(line_search: InteriorPointLineSearch<F>) -> Self {
        Self {
            line_search,
            barrier_parameter_factor: float!(0.1),
            barrier_parameter_floor: float!(1e-12),
        }
    }

    /// Set the multiplicative factor used in the simple complementarity-based
    /// barrier update.
    pub(super) fn with_barrier_parameter_factor(
        mut self,
        barrier_parameter_factor: F,
    ) -> Result<Self, Error> {
        if barrier_parameter_factor <= float!(0.0) || barrier_parameter_factor > float!(1.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "InteriorPointMethod: barrier parameter factor must satisfy 0 < factor <= 1."
            ));
        }
        self.barrier_parameter_factor = barrier_parameter_factor;
        Ok(self)
    }

    /// Set the lower bound on the barrier parameter.
    pub(super) fn with_barrier_parameter_floor(
        mut self,
        barrier_parameter_floor: F,
    ) -> Result<Self, Error> {
        if barrier_parameter_floor <= float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "InteriorPointMethod: barrier parameter floor must be > 0."
            ));
        }
        self.barrier_parameter_floor = barrier_parameter_floor;
        Ok(self)
    }

    /// Compute one full IPM iteration:
    /// evaluate -> residuals -> KKT direction -> line search -> accepted step.
    pub(super) fn compute_step<FCost>(
        &self,
        context: &IterationContext<F>,
        cost_function: FCost,
    ) -> Result<AcceptedIteration<F>, Error>
    where
        FCost: FnMut(&DVector<F>) -> Result<F, Error>,
    {
        let system = NalgebraKKTSystem::assemble(
            context.evaluation.lagrangian_hessian.clone(),
            context.evaluation.gradient.clone(),
            context.evaluation.equality_constraints.clone(),
            context.evaluation.inequality_constraints.clone(),
            context.evaluation.equality_constraint_jacobian.clone(),
            context.evaluation.inequality_constraint_jacobian.clone(),
            context.s.clone(),
            context.lambda_eq.clone(),
            context.lambda_ineq.clone(),
            context.mu,
        )?;

        let direction = system.solve_step()?;

        let line_search_result: LineSearchResult<F> = self.line_search.search(
            &context.x,
            &context.s,
            &context.lambda_eq,
            &context.lambda_ineq,
            &direction.dx,
            &direction.ds,
            &direction.dlambda_eq,
            &direction.dlambda_ineq,
            context.mu,
            context.evaluation.cost,
            cost_function,
        )?;

        Ok(AcceptedIteration {
            next_x: line_search_result.trial_point.x,
            next_s: line_search_result.trial_point.s,
            next_lambda_eq: line_search_result.trial_point.lambda_eq,
            next_lambda_ineq: line_search_result.trial_point.lambda_ineq,
            alpha_primal: line_search_result.alpha,
            alpha_dual: line_search_result.alpha,
            barrier_cost: line_search_result.barrier_value,
        })
    }

    /// Build the iteration context from already evaluated nonlinear-program
    /// quantities.
    pub(super) fn build_context(
        &self,
        x: DVector<F>,
        s: DVector<F>,
        lambda_eq: DVector<F>,
        lambda_ineq: DVector<F>,
        evaluation: IterationEvaluation<F>,
    ) -> Result<IterationContext<F>, Error> {
        validate_evaluation_dimensions(&x, &s, &lambda_eq, &lambda_ineq, &evaluation)?;

        let mu = self.compute_barrier_parameter(&s, &lambda_ineq)?;
        let residuals = compute_residuals(
            &s,
            &lambda_eq,
            &lambda_ineq,
            &evaluation.gradient,
            &evaluation.equality_constraints,
            &evaluation.inequality_constraints,
            &evaluation.equality_constraint_jacobian,
            &evaluation.inequality_constraint_jacobian,
            mu,
        )?;

        Ok(IterationContext {
            x,
            s,
            lambda_eq,
            lambda_ineq,
            mu,
            evaluation,
            residuals,
        })
    }

    /// Compute a simple complementarity-based barrier parameter:
    ///
    /// \[
    /// \mu = \max\left(\mu_{\min}, \sigma \frac{s^T \lambda_{ineq}}{m}\right)
    /// \]
    ///
    /// where `m` is the number of inequality constraints.
    pub(super) fn compute_barrier_parameter(
        &self,
        slacks: &DVector<F>,
        lambda_ineq: &DVector<F>,
    ) -> Result<F, Error> {
        if slacks.len() != lambda_ineq.len() {
            return Err(argmin_error!(
                InvalidParameter,
                "InteriorPointMethod: slacks and inequality multipliers must have the same length."
            ));
        }

        if slacks.is_empty() {
            return Ok(self.barrier_parameter_floor);
        }

        let m = F::from_u64(slacks.len() as u64).unwrap_or(float!(1.0));

        let complementarity = slacks.dot(lambda_ineq) / m;
        Ok((self.barrier_parameter_factor * complementarity).max(self.barrier_parameter_floor))
    }

    /// Write accepted values and diagnostics back to the shared NLP state.
    pub(super) fn write_state(
        &self,
        mut state: IpmState<F>,
        accepted: AcceptedIteration<F>,
        context: IterationContext<F>,
    ) -> IpmState<F> {
        state = state
            .param(accepted.next_x)
            .slacks(accepted.next_s)
            .cost(context.evaluation.cost)
            .gradient(context.evaluation.gradient)
            .lagrangian_hessian(context.evaluation.lagrangian_hessian)
            .equality_constraints(context.evaluation.equality_constraints)
            .inequality_constraints(context.evaluation.inequality_constraints)
            .equality_constraint_jacobian(context.evaluation.equality_constraint_jacobian)
            .inequality_constraint_jacobian(context.evaluation.inequality_constraint_jacobian)
            .lambda_eq(accepted.next_lambda_eq)
            .lambda_ineq(accepted.next_lambda_ineq)
            .mu(context.mu)
            .alpha_primal(accepted.alpha_primal)
            .alpha_dual(accepted.alpha_dual)
            .inf_pr(context.residuals.inf_pr)
            .inf_du(context.residuals.inf_du)
            .compl_inf(context.residuals.compl_inf);

        if state.get_best_param().is_none()
            || state.get_best_slacks().is_none()
            || context.evaluation.cost < state.get_best_cost()
        {
            state.best_param = state.param.clone();
            state.best_slacks = state.slacks.clone();
            state.best_cost = context.evaluation.cost;
        }

        state
    }

    /// Generate observer diagnostics for one accepted iteration.
    pub(super) fn kv(&self, context: &IterationContext<F>, accepted: &AcceptedIteration<F>) -> KV {
        kv!(
            "cost" => context.evaluation.cost;
            "barrier_cost" => accepted.barrier_cost;
            "mu" => context.mu;
            "inf_pr" => context.residuals.inf_pr;
            "inf_du" => context.residuals.inf_du;
            "compl_inf" => context.residuals.compl_inf;
            "alpha_primal" => accepted.alpha_primal;
            "alpha_dual" => accepted.alpha_dual;
        )
    }
}

/// Compute the residual blocks and scalar infeasibility measures for the basic
/// primal-dual system.
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_residuals<F>(
    slacks: &DVector<F>,
    lambda_eq: &DVector<F>,
    lambda_ineq: &DVector<F>,
    gradient: &DVector<F>,
    equality_constraints: &DVector<F>,
    inequality_constraints: &DVector<F>,
    equality_constraint_jacobian: &DMatrix<F>,
    inequality_constraint_jacobian: &DMatrix<F>,
    mu: F,
) -> Result<IterationResiduals<F>, Error>
where
    F: ArgminFloat + MulAssign + AddAssign + SubAssign,
{
    let n = gradient.len();
    let neq = equality_constraints.len();
    let nineq = inequality_constraints.len();

    if slacks.len() != nineq {
        return Err(argmin_error!(
            InvalidParameter,
            "InteriorPointMethod: number of slacks must equal the number of inequality constraints."
        ));
    }

    if lambda_ineq.len() != nineq {
        return Err(argmin_error!(
            InvalidParameter,
            "InteriorPointMethod: number of inequality multipliers must equal the number of inequality constraints."
        ));
    }

    if lambda_eq.len() != neq {
        return Err(argmin_error!(
            InvalidParameter,
            "InteriorPointMethod: number of equality multipliers must equal the number of equality constraints."
        ));
    }

    if equality_constraint_jacobian.nrows() != neq || equality_constraint_jacobian.ncols() != n {
        return Err(argmin_error!(
            InvalidParameter,
            "InteriorPointMethod: equality constraint Jacobian has incompatible dimensions."
        ));
    }

    if inequality_constraint_jacobian.nrows() != nineq
        || inequality_constraint_jacobian.ncols() != n
    {
        return Err(argmin_error!(
            InvalidParameter,
            "InteriorPointMethod: inequality constraint Jacobian has incompatible dimensions."
        ));
    }

    let dual = gradient
        + equality_constraint_jacobian.transpose() * lambda_eq
        + inequality_constraint_jacobian.transpose() * lambda_ineq;

    let equality = equality_constraints.clone();
    let inequality = inequality_constraints + slacks;

    let complementarity = slacks.component_mul(lambda_ineq) - DVector::from_element(nineq, mu);

    let inf_pr = infinity_norm_pair(&equality, &inequality);
    let inf_du = infinity_norm(&dual);
    let compl_inf = infinity_norm(&complementarity);

    Ok(IterationResiduals {
        dual,
        equality,
        inequality,
        complementarity,
        inf_pr,
        inf_du,
        compl_inf,
    })
}

/// Validate evaluated NLP dimensions against the current iterate.
fn validate_evaluation_dimensions<F>(
    x: &DVector<F>,
    s: &DVector<F>,
    lambda_eq: &DVector<F>,
    lambda_ineq: &DVector<F>,
    evaluation: &IterationEvaluation<F>,
) -> Result<(), Error>
where
    F: ArgminFloat,
{
    let n = x.len();
    let neq = evaluation.equality_constraints.len();
    let nineq = evaluation.inequality_constraints.len();

    if evaluation.gradient.len() != n {
        return Err(argmin_error!(
            InvalidParameter,
            "InteriorPointMethod: gradient dimension must match the number of parameters."
        ));
    }

    if evaluation.lagrangian_hessian.nrows() != n || evaluation.lagrangian_hessian.ncols() != n {
        return Err(argmin_error!(
            InvalidParameter,
            "InteriorPointMethod: Lagrangian Hessian must be square with parameter dimension."
        ));
    }

    if s.len() != nineq {
        return Err(argmin_error!(
            InvalidParameter,
            "InteriorPointMethod: slack dimension must match the number of inequality constraints."
        ));
    }

    if lambda_eq.len() != neq {
        return Err(argmin_error!(
            InvalidParameter,
            "InteriorPointMethod: equality multiplier dimension must match the number of equality constraints."
        ));
    }

    if lambda_ineq.len() != nineq {
        return Err(argmin_error!(
            InvalidParameter,
            "InteriorPointMethod: inequality multiplier dimension must match the number of inequality constraints."
        ));
    }

    if evaluation.equality_constraint_jacobian.nrows() != neq
        || evaluation.equality_constraint_jacobian.ncols() != n
    {
        return Err(argmin_error!(
            InvalidParameter,
            "InteriorPointMethod: equality Jacobian dimensions are inconsistent."
        ));
    }

    if evaluation.inequality_constraint_jacobian.nrows() != nineq
        || evaluation.inequality_constraint_jacobian.ncols() != n
    {
        return Err(argmin_error!(
            InvalidParameter,
            "InteriorPointMethod: inequality Jacobian dimensions are inconsistent."
        ));
    }

    Ok(())
}

/// Infinity norm of a dense vector. Returns zero for an empty vector.
fn infinity_norm<F>(values: &DVector<F>) -> F
where
    F: ArgminFloat,
{
    values.iter().copied().map(F::abs).fold(float!(0.0), F::max)
}

/// Maximum of two vector infinity norms.
fn infinity_norm_pair<F>(left: &DVector<F>, right: &DVector<F>) -> F
where
    F: ArgminFloat,
{
    infinity_norm(left).max(infinity_norm(right))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn basic_evaluation() -> IterationEvaluation<f64> {
        IterationEvaluation {
            cost: 2.0,
            gradient: DVector::from_vec(vec![1.0, -2.0]),
            lagrangian_hessian: DMatrix::identity(2, 2),
            equality_constraints: DVector::from_vec(vec![3.0]),
            inequality_constraints: DVector::from_vec(vec![-4.0]),
            equality_constraint_jacobian: DMatrix::from_row_slice(1, 2, &[1.0, 1.0]),
            inequality_constraint_jacobian: DMatrix::from_row_slice(1, 2, &[2.0, -1.0]),
        }
    }

    #[test]
    fn compute_barrier_parameter_matches_simple_average_complementarity_rule() {
        // With s = [2, 4], z = [3, 5], m = 2 and sigma = 0.1:
        //
        // s^T z / m = (2*3 + 4*5) / 2 = 13
        // mu = 0.1 * 13 = 1.3
        let helper = InteriorPointIteration::new(InteriorPointLineSearch::new());

        let mu = helper
            .compute_barrier_parameter(
                &DVector::from_vec(vec![2.0, 4.0]),
                &DVector::from_vec(vec![3.0, 5.0]),
            )
            .unwrap();

        assert_relative_eq!(mu, 1.3, epsilon = f64::EPSILON);
    }

    #[test]
    fn compute_residuals_matches_kkt_blocks() {
        // Data:
        // grad = [1, -2]
        // J_eq^T * lambda_eq = [1, 1] * 0.5 = [0.5, 0.5]
        // J_ineq^T * lambda_ineq = [2, -1] * 4 = [8, -4]
        //
        // Therefore
        // r_du = [1, -2] + [0.5, 0.5] + [8, -4] = [9.5, -5.5]
        //
        // Equality residual:
        // r_eq = c_eq = [3]
        //
        // Inequality residual:
        // r_ineq = c_ineq + s = [-4] + [5] = [1]
        //
        // Complementarity residual:
        // r_c = s .* z - mu e = [5 * 4 - 2] = [18]
        let residuals = compute_residuals(
            &DVector::from_vec(vec![5.0]),
            &DVector::from_vec(vec![0.5]),
            &DVector::from_vec(vec![4.0]),
            &DVector::from_vec(vec![1.0, -2.0]),
            &DVector::from_vec(vec![3.0]),
            &DVector::from_vec(vec![-4.0]),
            &DMatrix::from_row_slice(1, 2, &[1.0, 1.0]),
            &DMatrix::from_row_slice(1, 2, &[2.0, -1.0]),
            2.0,
        )
        .unwrap();

        assert_eq!(residuals.dual, DVector::from_vec(vec![9.5, -5.5]));
        assert_eq!(residuals.equality, DVector::from_vec(vec![3.0]));
        assert_eq!(residuals.inequality, DVector::from_vec(vec![1.0]));
        assert_eq!(residuals.complementarity, DVector::from_vec(vec![18.0]));
        assert_relative_eq!(residuals.inf_pr, 3.0, epsilon = f64::EPSILON);
        assert_relative_eq!(residuals.inf_du, 9.5, epsilon = f64::EPSILON);
        assert_relative_eq!(residuals.compl_inf, 18.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn build_context_validates_and_populates_mu_and_residuals() {
        let helper = InteriorPointIteration::new(InteriorPointLineSearch::new());

        let context = helper
            .build_context(
                DVector::from_vec(vec![1.0, 2.0]),
                DVector::from_vec(vec![5.0]),
                DVector::from_vec(vec![0.5]),
                DVector::from_vec(vec![4.0]),
                basic_evaluation(),
            )
            .unwrap();

        assert_relative_eq!(context.mu, 2.0, epsilon = f64::EPSILON);
        assert_eq!(context.residuals.inequality, DVector::from_vec(vec![1.0]));
        assert_eq!(context.residuals.equality, DVector::from_vec(vec![3.0]));
    }

    #[test]
    fn infinity_norm_returns_zero_for_empty_vectors() {
        let values = DVector::<f64>::from_vec(vec![]);
        assert_eq!(infinity_norm(&values), 0.0);
    }

    #[test]
    fn validate_evaluation_dimensions_rejects_bad_gradient_size() {
        let mut evaluation = basic_evaluation();
        evaluation.gradient = DVector::from_vec(vec![1.0]);

        let result = validate_evaluation_dimensions(
            &DVector::from_vec(vec![1.0, 2.0]),
            &DVector::from_vec(vec![1.0]),
            &DVector::from_vec(vec![0.0]),
            &DVector::from_vec(vec![1.0]),
            &evaluation,
        );

        assert!(result.is_err());
    }
}

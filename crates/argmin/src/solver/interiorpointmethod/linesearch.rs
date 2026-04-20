// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Solver-specific backtracking line search for the experimental interior point method.
//!
//! The line search operates on a coupled primal-dual trial step, but accepts a
//! single scalar step length `alpha` in version 1 of the solver. The accepted
//! step must preserve strict positivity of slack variables and inequality
//! multipliers, and it must decrease the barrier objective.
//!
//! The barrier objective used here is
//!
//! ```text
//! phi_mu(x, s) = f(x) - mu * sum(log(s_i))
//! ```
//!
//! Equality and inequality feasibility are handled by the primal-dual Newton
//! system rather than by a merit function or filter. This module therefore keeps
//! the acceptance logic intentionally simple and focused on positivity and
//! barrier decrease.

use std::ops::{AddAssign, MulAssign};

use crate::core::{ArgminFloat, Error};
use nalgebra::DVector;

/// Trial point along a primal-dual search direction.
#[derive(Clone, Debug)]
pub(super) struct TrialPoint<F>
where
    F: ArgminFloat,
{
    /// Trial primal variables.
    pub x: DVector<F>,
    /// Trial slack variables.
    pub s: DVector<F>,
    /// Trial equality multipliers.
    pub lambda_eq: DVector<F>,
    /// Trial inequality multipliers.
    pub lambda_ineq: DVector<F>,
}

/// Result of a successful line search.
#[derive(Clone, Debug)]
pub(super) struct LineSearchResult<F>
where
    F: ArgminFloat,
{
    /// Accepted step length.
    pub alpha: F,
    /// Accepted trial point.
    pub trial_point: TrialPoint<F>,
    /// Barrier objective at the accepted trial point.
    pub barrier_value: F,
}

/// Simple backtracking line search for the interior point method.
#[derive(Clone, Debug)]
pub(super) struct InteriorPointLineSearch<F>
where
    F: ArgminFloat,
{
    /// Initial step length before backtracking.
    alpha_init: F,
    /// Geometric reduction factor applied after each rejected trial step.
    contraction_factor: F,
    /// Fraction-to-the-boundary safety factor used to preserve strict positivity.
    fraction_to_boundary: F,
    /// Maximum number of backtracking iterations.
    max_iters: u64,
    /// Minimum acceptable trial step length before the search fails.
    alpha_min: F,
}

impl<F> Default for InteriorPointLineSearch<F>
where
    F: ArgminFloat,
{
    fn default() -> Self {
        Self {
            alpha_init: float!(1.0),
            contraction_factor: float!(0.5),
            fraction_to_boundary: float!(0.995),
            max_iters: 50,
            alpha_min: float!(1e-12),
        }
    }
}

impl<F> InteriorPointLineSearch<F>
where
    F: ArgminFloat,
{
    /// Construct a line search with conservative defaults suitable for a first
    /// dense interior-point implementation.
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Set the initial step length used before backtracking starts.
    pub(super) fn with_initial_step_length(mut self, alpha_init: F) -> Result<Self, Error> {
        if alpha_init <= float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "InteriorPointMethod: initial line-search step length must be > 0."
            ));
        }
        self.alpha_init = alpha_init;
        Ok(self)
    }

    /// Set the geometric contraction factor used after a rejected trial step.
    pub(super) fn with_contraction_factor(mut self, contraction_factor: F) -> Result<Self, Error> {
        if contraction_factor <= float!(0.0) || contraction_factor >= float!(1.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "InteriorPointMethod: line-search contraction factor must satisfy 0 < factor < 1."
            ));
        }
        self.contraction_factor = contraction_factor;
        Ok(self)
    }

    /// Set the fraction-to-the-boundary safety factor.
    pub(super) fn with_fraction_to_boundary(
        mut self,
        fraction_to_boundary: F,
    ) -> Result<Self, Error> {
        if fraction_to_boundary <= float!(0.0) || fraction_to_boundary >= float!(1.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "InteriorPointMethod: fraction-to-the-boundary must satisfy 0 < tau < 1."
            ));
        }
        self.fraction_to_boundary = fraction_to_boundary;
        Ok(self)
    }

    /// Set the maximum number of backtracking iterations.
    pub(super) fn with_max_iters(mut self, max_iters: u64) -> Result<Self, Error> {
        if max_iters == 0 {
            return Err(argmin_error!(
                InvalidParameter,
                "InteriorPointMethod: line-search max_iters must be at least 1."
            ));
        }
        self.max_iters = max_iters;
        Ok(self)
    }

    /// Set the minimum step length tolerated before the line search fails.
    pub(super) fn with_min_step_length(mut self, alpha_min: F) -> Result<Self, Error> {
        if alpha_min <= float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "InteriorPointMethod: minimum line-search step length must be > 0."
            ));
        }
        self.alpha_min = alpha_min;
        Ok(self)
    }

    /// Compute the largest admissible step length that preserves strict
    /// positivity of both slacks and inequality multipliers.
    pub(super) fn maximum_step_length(
        &self,
        slacks: &DVector<F>,
        delta_slacks: &DVector<F>,
        lambda_ineq: &DVector<F>,
        delta_lambda_ineq: &DVector<F>,
    ) -> Result<F, Error> {
        if slacks.len() != delta_slacks.len() {
            return Err(argmin_error!(
                InvalidParameter,
                "InteriorPointMethod: slack step dimension must match the number of slacks."
            ));
        }

        if lambda_ineq.len() != delta_lambda_ineq.len() {
            return Err(argmin_error!(
                InvalidParameter,
                "InteriorPointMethod: inequality multiplier step dimension must match the number of inequality multipliers."
            ));
        }

        let alpha_slacks =
            positivity_preserving_step_length(slacks, delta_slacks, self.fraction_to_boundary)?;
        let alpha_lambda = positivity_preserving_step_length(
            lambda_ineq,
            delta_lambda_ineq,
            self.fraction_to_boundary,
        )?;

        Ok(alpha_slacks.min(alpha_lambda).min(self.alpha_init))
    }

    /// Run a barrier-only backtracking line search.
    ///
    /// The closure `cost_function` is evaluated only after positivity has been
    /// enforced by the fraction-to-the-boundary limit. The accepted trial point
    /// must strictly preserve positive slacks and positive inequality
    /// multipliers, and it must not increase the barrier objective.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn search<FCost>(
        &self,
        x: &DVector<F>,
        s: &DVector<F>,
        lambda_eq: &DVector<F>,
        lambda_ineq: &DVector<F>,
        dx: &DVector<F>,
        ds: &DVector<F>,
        dlambda_eq: &DVector<F>,
        dlambda_ineq: &DVector<F>,
        mu: F,
        current_cost: F,
        mut cost_function: FCost,
    ) -> Result<LineSearchResult<F>, Error>
    where
        FCost: FnMut(&DVector<F>) -> Result<F, Error>,
        F: ArgminFloat + AddAssign + MulAssign,
    {
        validate_step_dimensions(
            x,
            dx,
            s,
            ds,
            lambda_eq,
            dlambda_eq,
            lambda_ineq,
            dlambda_ineq,
        )?;

        let current_barrier = barrier_objective(current_cost, s, mu)?;
        let mut alpha = self.maximum_step_length(s, ds, lambda_ineq, dlambda_ineq)?;

        for _ in 0..self.max_iters {
            if alpha < self.alpha_min {
                break;
            }

            let trial_point = trial_point(
                x,
                s,
                lambda_eq,
                lambda_ineq,
                dx,
                ds,
                dlambda_eq,
                dlambda_ineq,
                alpha,
            );

            if !is_strictly_positive(&trial_point.s)
                || !is_strictly_positive(&trial_point.lambda_ineq)
            {
                alpha = alpha * self.contraction_factor;
                continue;
            }

            let trial_cost = cost_function(&trial_point.x)?;
            let trial_barrier = barrier_objective(trial_cost, &trial_point.s, mu)?;

            if trial_barrier <= current_barrier {
                return Ok(LineSearchResult {
                    alpha,
                    trial_point,
                    barrier_value: trial_barrier,
                });
            }

            alpha = alpha * self.contraction_factor;
        }

        Err(argmin_error!(
            ConditionViolated,
            "InteriorPointMethod: barrier line search failed to find an acceptable step."
        ))
    }
}

/// Assemble a trial point from the current iterate, a primal-dual step, and a
/// scalar step length.
#[allow(clippy::too_many_arguments)]
fn trial_point<F>(
    x: &DVector<F>,
    s: &DVector<F>,
    lambda_eq: &DVector<F>,
    lambda_ineq: &DVector<F>,
    dx: &DVector<F>,
    ds: &DVector<F>,
    dlambda_eq: &DVector<F>,
    dlambda_ineq: &DVector<F>,
    alpha: F,
) -> TrialPoint<F>
where
    F: ArgminFloat + MulAssign + AddAssign,
{
    TrialPoint {
        x: x + dx * alpha,
        s: s + ds * alpha,
        lambda_eq: lambda_eq + dlambda_eq * alpha,
        lambda_ineq: lambda_ineq + dlambda_ineq * alpha,
    }
}

/// Compute the barrier objective `f(x) - mu * sum(log(s_i))`.
fn barrier_objective<F>(cost: F, slacks: &DVector<F>, mu: F) -> Result<F, Error>
where
    F: ArgminFloat,
{
    if !is_strictly_positive(slacks) {
        return Err(argmin_error!(
            ConditionViolated,
            "InteriorPointMethod: barrier objective is undefined for nonpositive slacks."
        ));
    }

    let log_barrier = slacks
        .iter()
        .copied()
        .fold(float!(0.0), |acc, si| acc + si.ln());

    Ok(cost - mu * log_barrier)
}

/// Check whether all entries in a vector are strictly positive.
fn is_strictly_positive<F>(values: &DVector<F>) -> bool
where
    F: ArgminFloat,
{
    values.iter().all(|&v| v > float!(0.0))
}

/// Compute the largest positivity-preserving step length for a vector update
/// `values + alpha * delta_values`.
fn positivity_preserving_step_length<F>(
    values: &DVector<F>,
    delta_values: &DVector<F>,
    fraction_to_boundary: F,
) -> Result<F, Error>
where
    F: ArgminFloat,
{
    if values.len() != delta_values.len() {
        return Err(argmin_error!(
            InvalidParameter,
            "InteriorPointMethod: positivity step-length inputs must have matching dimensions."
        ));
    }

    let mut alpha = float!(1.0);

    for (&value, &delta_value) in values.iter().zip(delta_values.iter()) {
        if delta_value < float!(0.0) {
            alpha = alpha.min(-fraction_to_boundary * value / delta_value);
        }
    }

    Ok(alpha)
}

/// Validate that all step blocks match the current iterate dimensions.
#[allow(clippy::too_many_arguments)]
fn validate_step_dimensions<F>(
    x: &DVector<F>,
    dx: &DVector<F>,
    s: &DVector<F>,
    ds: &DVector<F>,
    lambda_eq: &DVector<F>,
    dlambda_eq: &DVector<F>,
    lambda_ineq: &DVector<F>,
    dlambda_ineq: &DVector<F>,
) -> Result<(), Error>
where
    F: ArgminFloat,
{
    if x.len() != dx.len() {
        return Err(argmin_error!(
            InvalidParameter,
            "InteriorPointMethod: primal step dimension must match the number of parameters."
        ));
    }

    if s.len() != ds.len() {
        return Err(argmin_error!(
            InvalidParameter,
            "InteriorPointMethod: slack step dimension must match the number of slacks."
        ));
    }

    if lambda_eq.len() != dlambda_eq.len() {
        return Err(argmin_error!(
            InvalidParameter,
            "InteriorPointMethod: equality multiplier step dimension must match the number of equality multipliers."
        ));
    }

    if lambda_ineq.len() != dlambda_ineq.len() {
        return Err(argmin_error!(
            InvalidParameter,
            "InteriorPointMethod: inequality multiplier step dimension must match the number of inequality multipliers."
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn positivity_preserving_step_length_is_one_when_direction_is_nonnegative() {
        // If every directional derivative is nonnegative, moving with alpha = 1
        // cannot violate positivity. The admissible step should therefore remain 1.
        let values = DVector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let delta = DVector::<f64>::from_vec(vec![0.5, 0.0, 4.0]);

        let alpha = positivity_preserving_step_length(&values, &delta, 0.995).unwrap();

        assert_relative_eq!(alpha, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn positivity_preserving_step_length_respects_fraction_to_boundary() {
        // We test the scalar condition
        //
        // value + alpha * delta > 0
        //
        // with value = 2 and delta = -4. The boundary is hit at alpha = 0.5.
        // With tau = 0.9, the accepted maximum should be
        //
        // alpha_max = -tau * value / delta = 0.45.
        let values = DVector::<f64>::from_vec(vec![2.0]);
        let delta = DVector::<f64>::from_vec(vec![-4.0]);

        let alpha = positivity_preserving_step_length(&values, &delta, 0.9).unwrap();

        assert_relative_eq!(alpha, 0.45, epsilon = 1e-12);
    }

    #[test]
    fn maximum_step_length_takes_minimum_of_slack_and_multiplier_limits() {
        // Slack limit:
        //   s = 1, ds = -4, tau = 0.9 => alpha <= 0.225
        //
        // Multiplier limit:
        //   z = 3, dz = -2, tau = 0.9 => alpha <= 1.35
        //
        // The overall admissible step is therefore 0.225.
        let ls = InteriorPointLineSearch::<f64>::new()
            .with_fraction_to_boundary(0.9)
            .unwrap();

        let alpha = ls
            .maximum_step_length(
                &DVector::from_vec(vec![1.0]),
                &DVector::from_vec(vec![-4.0]),
                &DVector::from_vec(vec![3.0]),
                &DVector::from_vec(vec![-2.0]),
            )
            .unwrap();

        assert_relative_eq!(alpha, 0.225, epsilon = 1e-12);
    }

    #[test]
    fn barrier_objective_matches_manual_computation() {
        // For cost = 5, mu = 2, and s = [1, e],
        //
        // phi_mu = 5 - 2 * (ln(1) + ln(e)) = 5 - 2 * (0 + 1) = 3.
        let slacks = DVector::<f64>::from_vec(vec![1.0, std::f64::consts::E]);

        let phi = barrier_objective(5.0, &slacks, 2.0).unwrap();

        assert_relative_eq!(phi, 3.0, epsilon = 1e-12);
    }

    #[test]
    fn search_accepts_full_step_when_barrier_decreases() {
        // We choose a one-dimensional barrier problem with
        //
        // current x = 1, current s = 1, mu = 1,
        // step dx = -0.25, ds = 0.5.
        //
        // The test cost function is f(x) = x^2, so
        //
        // current phi = 1^2 - ln(1) = 1,
        // trial phi   = 0.75^2 - ln(1.5) = 0.5625 - 0.405465...
        //
        // which is strictly smaller than 1. The line search should accept
        // alpha = 1 immediately.
        let ls = InteriorPointLineSearch::<f64>::new();

        let result = ls
            .search(
                &DVector::from_vec(vec![1.0]),
                &DVector::from_vec(vec![1.0]),
                &DVector::from_vec(vec![]),
                &DVector::from_vec(vec![1.0]),
                &DVector::from_vec(vec![-0.25]),
                &DVector::from_vec(vec![0.5]),
                &DVector::from_vec(vec![]),
                &DVector::from_vec(vec![0.0]),
                1.0,
                1.0,
                |x| Ok(x[0] * x[0]),
            )
            .unwrap();

        assert_relative_eq!(result.alpha, 1.0, epsilon = 1e-12);
        assert_relative_eq!(result.trial_point.x[0], 0.75, epsilon = 1e-12);
        assert_relative_eq!(result.trial_point.s[0], 1.5, epsilon = 1e-12);
    }

    #[test]
    fn search_backtracks_until_barrier_decreases() {
        // Same test cost function f(x) = x^2, but we start with a step that is
        // too aggressive:
        //
        // x = 1, dx = -2, s = 1, ds = -0.2, mu = 0.1.
        //
        // At alpha = 1:
        //   trial x = -1, trial s = 0.8,
        //   phi = 1 - 0.1 ln(0.8) > current phi = 1.
        //
        // At alpha = 0.5:
        //   trial x = 0, trial s = 0.9,
        //   phi = 0 - 0.1 ln(0.9) < 1.
        //
        // Therefore the line search should reject alpha = 1 and accept alpha = 0.5.
        let ls = InteriorPointLineSearch::<f64>::new();

        let result = ls
            .search(
                &DVector::from_vec(vec![1.0]),
                &DVector::from_vec(vec![1.0]),
                &DVector::from_vec(vec![]),
                &DVector::from_vec(vec![1.0]),
                &DVector::from_vec(vec![-2.0]),
                &DVector::from_vec(vec![-0.2]),
                &DVector::from_vec(vec![]),
                &DVector::from_vec(vec![0.0]),
                0.1,
                1.0,
                |x| Ok(x[0] * x[0]),
            )
            .unwrap();

        assert_relative_eq!(result.alpha, 0.5, epsilon = 1e-12);
        assert_relative_eq!(result.trial_point.x[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(result.trial_point.s[0], 0.9, epsilon = 1e-12);
    }

    #[test]
    fn search_rejects_when_no_acceptable_step_exists() {
        // The barrier term can only help if slacks stay positive, but the test
        // cost function below is constant and strictly larger than the current
        // cost for every trial point. Since the barrier also increases because
        // ds = 0, every trial step is rejected and the line search must fail.
        let ls = InteriorPointLineSearch::<f64>::new()
            .with_max_iters(5)
            .unwrap()
            .with_min_step_length(1e-6)
            .unwrap();

        let result = ls.search(
            &DVector::from_vec(vec![1.0]),
            &DVector::from_vec(vec![1.0]),
            &DVector::from_vec(vec![]),
            &DVector::from_vec(vec![1.0]),
            &DVector::from_vec(vec![0.0]),
            &DVector::from_vec(vec![0.0]),
            &DVector::from_vec(vec![]),
            &DVector::from_vec(vec![0.0]),
            1.0,
            0.0,
            |_x| Ok(2.0),
        );

        assert!(result.is_err());
    }

    #[test]
    fn search_rejects_dimension_mismatch() {
        let ls = InteriorPointLineSearch::<f64>::new();

        let result = ls.search(
            &DVector::from_vec(vec![1.0, 2.0]),
            &DVector::from_vec(vec![1.0]),
            &DVector::from_vec(vec![]),
            &DVector::from_vec(vec![1.0]),
            &DVector::from_vec(vec![1.0]),
            &DVector::from_vec(vec![0.0]),
            &DVector::from_vec(vec![]),
            &DVector::from_vec(vec![0.0]),
            1.0,
            0.0,
            |_x| Ok(0.0),
        );

        assert!(result.is_err());
    }
}

// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Dense nalgebra-backed assembly and solve routines for the interior point KKT system.
//!
//! The semantic block ordering is fixed to
//!
//! `(x, s, lambda_ineq, lambda_eq)`
//!
//! where
//! - `x` are the primal variables,
//! - `s` are the slack variables,
//! - `lambda_ineq` are the inequality multipliers, and
//! - `lambda_eq` are the equality multipliers.
//!
//! The KKT matrix assembled here has the block form
//!
//! ```text
//! [ H        0        J_ineq^T   J_eq^T ]
//! [ 0     diag(z)    diag(s)      0     ]
//! [ J_ineq   I          0         0     ]
//! [ J_eq     0          0         0     ]
//! ```
//!
//! and the stacked residual vector is
//!
//! ```text
//! [ grad f(x) + J_ineq^T z + J_eq^T lambda_eq ]
//! [ s .* z - mu * 1                           ]
//! [ g(x) + s                                  ]
//! [ h(x)                                      ]
//! ```
//!
//! The Newton step `delta` is obtained from
//!
//! ```text
//! KKT * delta = -residuals
//! ```

use crate::core::{ArgminFloat, Error};
use argmin_math::ArgminLuSolve;
use nalgebra::{ClosedAddAssign, ClosedMulAssign, ClosedSubAssign, DMatrix, DVector};

/// Dimensions and offsets of the dense KKT system.
///
/// The canonical block ordering is
///
/// `(x, s, lambda_ineq, lambda_eq)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct KKTDimensions {
    /// Number of primal variables.
    pub n: usize,
    /// Number of inequality constraints.
    pub nineq: usize,
    /// Number of equality constraints.
    pub neq: usize,
}

impl KKTDimensions {
    /// Create a new dimension descriptor.
    #[inline]
    pub(super) fn new(n: usize, nineq: usize, neq: usize) -> Self {
        Self { n, nineq, neq }
    }

    /// Offset of the primal variable block `x`.
    #[inline]
    pub(super) fn x_offset(&self) -> usize {
        0
    }

    /// Offset of the slack block `s`.
    #[inline]
    pub(super) fn s_offset(&self) -> usize {
        self.n
    }

    /// Offset of the inequality multiplier block `lambda_ineq`.
    #[inline]
    pub(super) fn lambda_ineq_offset(&self) -> usize {
        self.n + self.nineq
    }

    /// Offset of the equality multiplier block `lambda_eq`.
    #[inline]
    pub(super) fn lambda_eq_offset(&self) -> usize {
        self.n + self.nineq + self.nineq
    }

    /// Total dimension of the KKT system.
    #[inline]
    pub(super) fn total_dim(&self) -> usize {
        self.n + self.nineq + self.nineq + self.neq
    }
}

/// Dense nalgebra realization of a KKT linear system.
///
/// Variable ordering:
/// `(x, s, lambda_ineq, lambda_eq)`.
#[derive(Clone, Debug)]
pub(super) struct NalgebraKKTSystem<F>
where
    F: ArgminFloat,
{
    matrix: DMatrix<F>,
    residuals: DVector<F>,
    dims: KKTDimensions,
}

/// Newton step split into semantic blocks.
///
/// The block ordering is
///
/// `(dx, ds, dlambda_ineq, dlambda_eq)`.
#[derive(Clone, Debug)]
pub(super) struct NewtonStep<F>
where
    F: ArgminFloat,
{
    /// Primal step.
    pub dx: DVector<F>,
    /// Slack step.
    pub ds: DVector<F>,
    /// Inequality multiplier step.
    pub dlambda_ineq: DVector<F>,
    /// Equality multiplier step.
    pub dlambda_eq: DVector<F>,
}

impl<F> NalgebraKKTSystem<F>
where
    F: ArgminFloat + ClosedAddAssign + ClosedSubAssign + ClosedMulAssign,
    DMatrix<F>: ArgminLuSolve<DVector<F>>,
{
    /// Assemble and validate the dense KKT system.
    ///
    /// Matrix layout:
    ///
    /// ```text
    /// [ H        0        J_ineq^T   J_eq^T ]
    /// [ 0     diag(z)    diag(s)      0     ]
    /// [ J_ineq   I          0         0     ]
    /// [ J_eq     0          0         0     ]
    /// ```
    ///
    /// Residual ordering:
    ///
    /// ```text
    /// [ grad f(x) + J_ineq^T z + J_eq^T lambda_eq ]
    /// [ s .* z - mu * 1                           ]
    /// [ g(x) + s                                  ]
    /// [ h(x)                                      ]
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub(super) fn assemble(
        lagrangian_hessian: DMatrix<F>,
        gradient: DVector<F>,
        equality_constraints: DVector<F>,
        inequality_constraints: DVector<F>,
        equality_constraint_jacobian: DMatrix<F>,
        inequality_constraint_jacobian: DMatrix<F>,
        slacks: DVector<F>,
        lambda_eq: DVector<F>,
        lambda_ineq: DVector<F>,
        mu: F,
    ) -> Result<Self, Error> {
        let dims = validate_inputs(
            &lagrangian_hessian,
            &gradient,
            &equality_constraints,
            &inequality_constraints,
            &equality_constraint_jacobian,
            &inequality_constraint_jacobian,
            &slacks,
            &lambda_eq,
            &lambda_ineq,
        )?;

        let matrix = assemble_matrix(
            &dims,
            &lagrangian_hessian,
            &equality_constraint_jacobian,
            &inequality_constraint_jacobian,
            &slacks,
            &lambda_ineq,
        );

        let residuals = assemble_residuals(
            &dims,
            gradient,
            equality_constraints,
            inequality_constraints,
            &equality_constraint_jacobian,
            &inequality_constraint_jacobian,
            slacks,
            lambda_eq,
            lambda_ineq,
            mu,
        );

        Ok(Self {
            matrix,
            residuals,
            dims,
        })
    }

    /// Construct a KKT system from a preassembled matrix and residual vector.
    ///
    /// This is mainly useful for tests or for backends that want to validate the
    /// final assembled system separately from the assembly process.
    pub(super) fn from_parts(
        matrix: DMatrix<F>,
        residuals: DVector<F>,
        dims: KKTDimensions,
    ) -> Result<Self, Error> {
        if matrix.nrows() != matrix.ncols() {
            return Err(argmin_error!(
                InvalidParameter,
                "InteriorPointMethod: KKT matrix must be square."
            ));
        }

        if matrix.nrows() != dims.total_dim() {
            return Err(argmin_error!(
                InvalidParameter,
                "InteriorPointMethod: KKT matrix dimension must match the semantic KKT dimension."
            ));
        }

        if residuals.len() != dims.total_dim() {
            return Err(argmin_error!(
                InvalidParameter,
                "InteriorPointMethod: residual vector dimension must match the semantic KKT dimension."
            ));
        }

        Ok(Self {
            matrix,
            residuals,
            dims,
        })
    }

    /// Solve the Newton system `KKT * delta = -residuals`.
    pub(super) fn solve(&self) -> Result<DVector<F>, Error> {
        if self.dims.total_dim() == 0 {
            return Ok(DVector::zeros(0));
        }

        let rhs = -&self.residuals;
        self.matrix.clone().lu_solve(&rhs)
    }

    /// Solve the Newton system and split the result into semantic blocks.
    pub(super) fn solve_step(&self) -> Result<NewtonStep<F>, Error> {
        let step = self.solve()?;
        Ok(split_step(&self.dims, step))
    }

    /// Return the semantic dimensions of the system.
    #[inline]
    pub(super) fn dims(&self) -> KKTDimensions {
        self.dims
    }

    #[cfg(test)]
    pub(super) fn matrix(&self) -> &DMatrix<F> {
        &self.matrix
    }

    #[cfg(test)]
    pub(super) fn residuals(&self) -> &DVector<F> {
        &self.residuals
    }
}

/// Validate dimensions of all KKT assembly inputs and derive the semantic block sizes.
#[allow(clippy::too_many_arguments)]
fn validate_inputs<F>(
    lagrangian_hessian: &DMatrix<F>,
    gradient: &DVector<F>,
    equality_constraints: &DVector<F>,
    inequality_constraints: &DVector<F>,
    equality_constraint_jacobian: &DMatrix<F>,
    inequality_constraint_jacobian: &DMatrix<F>,
    slacks: &DVector<F>,
    lambda_eq: &DVector<F>,
    lambda_ineq: &DVector<F>,
) -> Result<KKTDimensions, Error>
where
    F: ArgminFloat,
{
    let n = lagrangian_hessian.nrows();
    if lagrangian_hessian.ncols() != n {
        return Err(argmin_error!(
            InvalidParameter,
            "InteriorPointMethod: Lagrangian Hessian must be square."
        ));
    }

    if gradient.len() != n {
        return Err(argmin_error!(
            InvalidParameter,
            "InteriorPointMethod: gradient dimension must match the number of parameters."
        ));
    }

    let neq = equality_constraints.len();
    let nineq = inequality_constraints.len();

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

    if slacks.len() != nineq {
        return Err(argmin_error!(
            InvalidParameter,
            "InteriorPointMethod: number of slacks must match the number of inequality constraints."
        ));
    }

    if lambda_ineq.len() != nineq {
        return Err(argmin_error!(
            InvalidParameter,
            "InteriorPointMethod: number of inequality multipliers must match the number of inequality constraints."
        ));
    }

    if lambda_eq.len() != neq {
        return Err(argmin_error!(
            InvalidParameter,
            "InteriorPointMethod: number of equality multipliers must match the number of equality constraints."
        ));
    }

    Ok(KKTDimensions::new(n, nineq, neq))
}

/// Assemble the dense KKT matrix for the current iterate.
///
/// The block ordering is
///
/// `(x, s, lambda_ineq, lambda_eq)`.
fn assemble_matrix<F>(
    dims: &KKTDimensions,
    lagrangian_hessian: &DMatrix<F>,
    equality_constraint_jacobian: &DMatrix<F>,
    inequality_constraint_jacobian: &DMatrix<F>,
    slacks: &DVector<F>,
    lambda_ineq: &DVector<F>,
) -> DMatrix<F>
where
    F: ArgminFloat + ClosedAddAssign + ClosedSubAssign + ClosedMulAssign,
{
    let dim = dims.total_dim();
    let mut kkt = DMatrix::<F>::zeros(dim, dim);

    let xoff = dims.x_offset();
    let soff = dims.s_offset();
    let zoff = dims.lambda_ineq_offset();
    let leqoff = dims.lambda_eq_offset();

    let n = dims.n;
    let nineq = dims.nineq;
    let neq = dims.neq;

    kkt.view_mut((xoff, xoff), (n, n))
        .copy_from(lagrangian_hessian);

    if nineq > 0 {
        let jineq_t = inequality_constraint_jacobian.transpose();

        kkt.view_mut((xoff, zoff), (n, nineq)).copy_from(&jineq_t);

        kkt.view_mut((soff, soff), (nineq, nineq))
            .copy_from(&DMatrix::<F>::from_diagonal(lambda_ineq));

        kkt.view_mut((soff, zoff), (nineq, nineq))
            .copy_from(&DMatrix::<F>::from_diagonal(slacks));

        kkt.view_mut((zoff, xoff), (nineq, n))
            .copy_from(inequality_constraint_jacobian);

        kkt.view_mut((zoff, soff), (nineq, nineq))
            .copy_from(&DMatrix::<F>::identity(nineq, nineq));
    }

    if neq > 0 {
        let jeq_t = equality_constraint_jacobian.transpose();

        kkt.view_mut((xoff, leqoff), (n, neq)).copy_from(&jeq_t);

        kkt.view_mut((leqoff, xoff), (neq, n))
            .copy_from(equality_constraint_jacobian);
    }

    kkt
}

/// Assemble the stacked KKT residual vector for the current iterate.
///
/// The residual ordering is
///
/// `(r_dual, r_cent, r_ineq, r_eq)`,
///
/// which is aligned with the variable ordering
///
/// `(x, s, lambda_ineq, lambda_eq)`.
#[allow(clippy::too_many_arguments)]
fn assemble_residuals<F>(
    dims: &KKTDimensions,
    gradient: DVector<F>,
    equality_constraints: DVector<F>,
    inequality_constraints: DVector<F>,
    equality_constraint_jacobian: &DMatrix<F>,
    inequality_constraint_jacobian: &DMatrix<F>,
    slacks: DVector<F>,
    lambda_eq: DVector<F>,
    lambda_ineq: DVector<F>,
    mu: F,
) -> DVector<F>
where
    F: ArgminFloat + ClosedAddAssign + ClosedSubAssign + ClosedMulAssign,
{
    let dim = dims.total_dim();
    let mut residuals = DVector::<F>::zeros(dim);

    let xoff = dims.x_offset();
    let soff = dims.s_offset();
    let zoff = dims.lambda_ineq_offset();
    let leqoff = dims.lambda_eq_offset();

    let r_dual = gradient
        + inequality_constraint_jacobian.transpose() * lambda_ineq.clone()
        + equality_constraint_jacobian.transpose() * lambda_eq;

    let r_cent = slacks.component_mul(&lambda_ineq) - DVector::<F>::from_element(dims.nineq, mu);
    let r_ineq = inequality_constraints + slacks;
    let r_eq = equality_constraints;

    residuals.rows_mut(xoff, dims.n).copy_from(&r_dual);

    if dims.nineq > 0 {
        residuals.rows_mut(soff, dims.nineq).copy_from(&r_cent);
        residuals.rows_mut(zoff, dims.nineq).copy_from(&r_ineq);
    }

    if dims.neq > 0 {
        residuals.rows_mut(leqoff, dims.neq).copy_from(&r_eq);
    }

    residuals
}

/// Split a full Newton step according to the semantic KKT ordering.
///
/// Input ordering:
///
/// `(dx, ds, dlambda_ineq, dlambda_eq)`.
fn split_step<F>(dims: &KKTDimensions, step: DVector<F>) -> NewtonStep<F>
where
    F: ArgminFloat,
{
    let dx = step.rows(dims.x_offset(), dims.n).into_owned();
    let ds = step.rows(dims.s_offset(), dims.nineq).into_owned();
    let dlambda_ineq = step
        .rows(dims.lambda_ineq_offset(), dims.nineq)
        .into_owned();
    let dlambda_eq = step.rows(dims.lambda_eq_offset(), dims.neq).into_owned();

    NewtonStep {
        dx,
        ds,
        dlambda_ineq,
        dlambda_eq,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn validate_inputs_accepts_consistent_dimensions() {
        let h = DMatrix::<f64>::identity(2, 2);
        let grad = DVector::<f64>::from_vec(vec![1.0, 2.0]);
        let ceq = DVector::<f64>::from_vec(vec![3.0]);
        let cineq = DVector::<f64>::from_vec(vec![4.0, 5.0]);
        let jeq = DMatrix::<f64>::from_row_slice(1, 2, &[1.0, 0.0]);
        let jineq = DMatrix::<f64>::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let s = DVector::<f64>::from_vec(vec![6.0, 7.0]);
        let leq = DVector::<f64>::from_vec(vec![8.0]);
        let lineq = DVector::<f64>::from_vec(vec![9.0, 10.0]);

        let dims =
            validate_inputs(&h, &grad, &ceq, &cineq, &jeq, &jineq, &s, &leq, &lineq).unwrap();

        assert_eq!(dims, KKTDimensions::new(2, 2, 1));
    }

    #[test]
    fn validate_inputs_rejects_bad_gradient_dimension() {
        let h = DMatrix::<f64>::identity(2, 2);
        let grad = DVector::<f64>::from_vec(vec![1.0]);
        let ceq = DVector::<f64>::from_vec(vec![]);
        let cineq = DVector::<f64>::from_vec(vec![]);
        let jeq = DMatrix::<f64>::zeros(0, 2);
        let jineq = DMatrix::<f64>::zeros(0, 2);
        let s = DVector::<f64>::from_vec(vec![]);
        let leq = DVector::<f64>::from_vec(vec![]);
        let lineq = DVector::<f64>::from_vec(vec![]);

        let result = validate_inputs(&h, &grad, &ceq, &cineq, &jeq, &jineq, &s, &leq, &lineq);

        assert!(result.is_err());
    }

    #[test]
    fn validate_inputs_rejects_bad_equality_jacobian_dimension() {
        let h = DMatrix::<f64>::identity(2, 2);
        let grad = DVector::<f64>::from_vec(vec![1.0, 2.0]);
        let ceq = DVector::<f64>::from_vec(vec![3.0]);
        let cineq = DVector::<f64>::from_vec(vec![]);
        let jeq = DMatrix::<f64>::zeros(1, 3);
        let jineq = DMatrix::<f64>::zeros(0, 2);
        let s = DVector::<f64>::from_vec(vec![]);
        let leq = DVector::<f64>::from_vec(vec![4.0]);
        let lineq = DVector::<f64>::from_vec(vec![]);

        let result = validate_inputs(&h, &grad, &ceq, &cineq, &jeq, &jineq, &s, &leq, &lineq);

        assert!(result.is_err());
    }

    #[test]
    fn validate_inputs_rejects_bad_inequality_jacobian_dimension() {
        let h = DMatrix::<f64>::identity(2, 2);
        let grad = DVector::<f64>::from_vec(vec![1.0, 2.0]);
        let ceq = DVector::<f64>::from_vec(vec![]);
        let cineq = DVector::<f64>::from_vec(vec![3.0]);
        let jeq = DMatrix::<f64>::zeros(0, 2);
        let jineq = DMatrix::<f64>::zeros(1, 3);
        let s = DVector::<f64>::from_vec(vec![4.0]);
        let leq = DVector::<f64>::from_vec(vec![]);
        let lineq = DVector::<f64>::from_vec(vec![5.0]);

        let result = validate_inputs(&h, &grad, &ceq, &cineq, &jeq, &jineq, &s, &leq, &lineq);

        assert!(result.is_err());
    }

    #[test]
    fn validate_inputs_rejects_bad_slack_dimension() {
        let h = DMatrix::<f64>::identity(2, 2);
        let grad = DVector::<f64>::from_vec(vec![1.0, 2.0]);
        let ceq = DVector::<f64>::from_vec(vec![]);
        let cineq = DVector::<f64>::from_vec(vec![3.0, 4.0]);
        let jeq = DMatrix::<f64>::zeros(0, 2);
        let jineq = DMatrix::<f64>::zeros(2, 2);
        let s = DVector::<f64>::from_vec(vec![5.0]);
        let leq = DVector::<f64>::from_vec(vec![]);
        let lineq = DVector::<f64>::from_vec(vec![6.0, 7.0]);

        let result = validate_inputs(&h, &grad, &ceq, &cineq, &jeq, &jineq, &s, &leq, &lineq);

        assert!(result.is_err());
    }

    #[test]
    fn validate_inputs_rejects_bad_lambda_eq_dimension() {
        let h = DMatrix::<f64>::identity(2, 2);
        let grad = DVector::<f64>::from_vec(vec![1.0, 2.0]);
        let ceq = DVector::<f64>::from_vec(vec![3.0]);
        let cineq = DVector::<f64>::from_vec(vec![]);
        let jeq = DMatrix::<f64>::zeros(1, 2);
        let jineq = DMatrix::<f64>::zeros(0, 2);
        let s = DVector::<f64>::from_vec(vec![]);
        let leq = DVector::<f64>::from_vec(vec![4.0, 5.0]);
        let lineq = DVector::<f64>::from_vec(vec![]);

        let result = validate_inputs(&h, &grad, &ceq, &cineq, &jeq, &jineq, &s, &leq, &lineq);

        assert!(result.is_err());
    }

    #[test]
    fn validate_inputs_rejects_bad_lambda_ineq_dimension() {
        let h = DMatrix::<f64>::identity(2, 2);
        let grad = DVector::<f64>::from_vec(vec![1.0, 2.0]);
        let ceq = DVector::<f64>::from_vec(vec![]);
        let cineq = DVector::<f64>::from_vec(vec![3.0]);
        let jeq = DMatrix::<f64>::zeros(0, 2);
        let jineq = DMatrix::<f64>::zeros(1, 2);
        let s = DVector::<f64>::from_vec(vec![4.0]);
        let leq = DVector::<f64>::from_vec(vec![]);
        let lineq = DVector::<f64>::from_vec(vec![5.0, 6.0]);

        let result = validate_inputs(&h, &grad, &ceq, &cineq, &jeq, &jineq, &s, &leq, &lineq);

        assert!(result.is_err());
    }

    #[test]
    fn assemble_matrix_has_expected_entries() {
        let dims = KKTDimensions::new(2, 1, 1);

        let h = DMatrix::<f64>::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let jeq = DMatrix::<f64>::from_row_slice(1, 2, &[7.0, 8.0]);
        let jineq = DMatrix::<f64>::from_row_slice(1, 2, &[9.0, 10.0]);
        let s = DVector::<f64>::from_vec(vec![11.0]);
        let z = DVector::<f64>::from_vec(vec![13.0]);

        let kkt = assemble_matrix(&dims, &h, &jeq, &jineq, &s, &z);

        let expected = DMatrix::<f64>::from_row_slice(
            5,
            5,
            &[
                4.0, 1.0, 0.0, 9.0, 7.0, 1.0, 3.0, 0.0, 10.0, 8.0, 0.0, 0.0, 13.0, 11.0, 0.0, 9.0,
                10.0, 1.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0,
            ],
        );

        for i in 0..5 {
            for j in 0..5 {
                assert_relative_eq!(kkt[(i, j)], expected[(i, j)], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn assemble_residuals_has_expected_entries() {
        let dims = KKTDimensions::new(2, 1, 1);

        let grad = DVector::<f64>::from_vec(vec![1.0, 2.0]);
        let ceq = DVector::<f64>::from_vec(vec![5.0]);
        let cineq = DVector::<f64>::from_vec(vec![6.0]);
        let jeq = DMatrix::<f64>::from_row_slice(1, 2, &[7.0, 8.0]);
        let jineq = DMatrix::<f64>::from_row_slice(1, 2, &[9.0, 10.0]);
        let s = DVector::<f64>::from_vec(vec![11.0]);
        let z = DVector::<f64>::from_vec(vec![13.0]);
        let leq = DVector::<f64>::from_vec(vec![12.0]);

        let residuals = assemble_residuals(&dims, grad, ceq, cineq, &jeq, &jineq, s, leq, z, 14.0);

        let expected = DVector::<f64>::from_vec(vec![202.0, 228.0, 129.0, 17.0, 5.0]);

        for i in 0..expected.len() {
            assert_relative_eq!(residuals[i], expected[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn split_step_respects_ordering() {
        let dims = KKTDimensions::new(2, 2, 1);
        let step = DVector::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        let split = split_step(&dims, step);

        assert_eq!(split.dx, DVector::<f64>::from_vec(vec![1.0, 2.0]));
        assert_eq!(split.ds, DVector::<f64>::from_vec(vec![3.0, 4.0]));
        assert_eq!(split.dlambda_ineq, DVector::<f64>::from_vec(vec![5.0, 6.0]));
        assert_eq!(split.dlambda_eq, DVector::<f64>::from_vec(vec![7.0]));
    }

    #[test]
    fn from_parts_rejects_inconsistent_matrix_dimension() {
        let dims = KKTDimensions::new(2, 1, 0);
        let matrix = DMatrix::<f64>::identity(2, 2);
        let residuals = DVector::<f64>::zeros(4);

        let result = NalgebraKKTSystem::from_parts(matrix, residuals, dims);

        assert!(result.is_err());
    }

    #[test]
    fn from_parts_rejects_inconsistent_residual_dimension() {
        let dims = KKTDimensions::new(2, 1, 0);
        let matrix = DMatrix::<f64>::identity(4, 4);
        let residuals = DVector::<f64>::zeros(3);

        let result = NalgebraKKTSystem::from_parts(matrix, residuals, dims);

        assert!(result.is_err());
    }

    #[test]
    fn solve_step_without_constraints_matches_unconstrained_newton_step() {
        // Test problem:
        //
        // minimize f(x) = 1/2 * x^T H x + g^T x
        //
        // with
        //
        // H = [[4, 1],
        //      [1, 3]]
        //
        // g = [1, 2]
        //
        // At any iterate, the unconstrained Newton step solves
        //
        // H dx = -grad f(x).
        //
        // With no equality or inequality constraints, the KKT system reduces
        // exactly to this 2x2 Newton system. The exact solution is
        //
        // dx = [ -1/11,
        //        -7/11 ].
        let h = DMatrix::<f64>::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let grad = DVector::<f64>::from_vec(vec![1.0, 2.0]);
        let ceq = DVector::<f64>::from_vec(vec![]);
        let cineq = DVector::<f64>::from_vec(vec![]);
        let jeq = DMatrix::<f64>::zeros(0, 2);
        let jineq = DMatrix::<f64>::zeros(0, 2);
        let s = DVector::<f64>::from_vec(vec![]);
        let leq = DVector::<f64>::from_vec(vec![]);
        let lineq = DVector::<f64>::from_vec(vec![]);

        let system =
            NalgebraKKTSystem::assemble(h, grad, ceq, cineq, jeq, jineq, s, leq, lineq, 1.0)
                .unwrap();

        let step = system.solve_step().unwrap();

        assert_relative_eq!(step.dx[0], -1.0 / 11.0, epsilon = 1e-12);
        assert_relative_eq!(step.dx[1], -7.0 / 11.0, epsilon = 1e-12);
        assert_eq!(step.ds.len(), 0);
        assert_eq!(step.dlambda_ineq.len(), 0);
        assert_eq!(step.dlambda_eq.len(), 0);
    }

    #[test]
    fn solve_step_with_one_inequality_constraint_matches_exact_solution() {
        // Test problem:
        //
        // minimize f(x) = 1/2 * x^T x
        //
        // subject to
        //
        // g(x) = x1 + x2 - 1 <= 0.
        //
        // We test the primal-dual interior-point Newton system at the iterate
        //
        // x = (0, 0),
        // s = 1,
        // z = 1,
        // mu = 1.
        //
        // For this problem,
        //
        // grad f(x) = [0, 0],
        // H = I,
        // J_ineq = [1, 1],
        // g(x) = -1.
        //
        // The complementarity residual and primal inequality residual are both zero:
        //
        // s * z - mu = 1 * 1 - 1 = 0,
        // g(x) + s = -1 + 1 = 0.
        //
        // However, the dual residual is not zero:
        //
        // grad f(x) + J_ineq^T z = [0, 0] + [1, 1] = [1, 1].
        //
        // Therefore the Newton step is not the zero step. Solving
        //
        // [ 1  0  0  1 ] [dx1]   [ -1 ]
        // [ 0  1  0  1 ] [dx2] = [ -1 ]
        // [ 0  0  1  1 ] [ ds]   [  0 ]
        // [ 1  1  1  0 ] [ dz ]  [  0 ]
        //
        // yields the exact solution
        //
        // dx = (-1/3, -1/3),
        // ds =  2/3,
        // dz = -2/3.
        let h = DMatrix::<f64>::identity(2, 2);
        let grad = DVector::<f64>::from_vec(vec![0.0, 0.0]);
        let ceq = DVector::<f64>::from_vec(vec![]);
        let cineq = DVector::<f64>::from_vec(vec![-1.0]);
        let jeq = DMatrix::<f64>::zeros(0, 2);
        let jineq = DMatrix::<f64>::from_row_slice(1, 2, &[1.0, 1.0]);
        let s = DVector::<f64>::from_vec(vec![1.0]);
        let leq = DVector::<f64>::from_vec(vec![]);
        let lineq = DVector::<f64>::from_vec(vec![1.0]);

        let system =
            NalgebraKKTSystem::assemble(h, grad, ceq, cineq, jeq, jineq, s, leq, lineq, 1.0)
                .unwrap();

        let step = system.solve_step().unwrap();

        assert_relative_eq!(step.dx[0], -1.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(step.dx[1], -1.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(step.ds[0], 2.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(step.dlambda_ineq[0], -2.0 / 3.0, epsilon = 1e-12);
        assert_eq!(step.dlambda_eq.len(), 0);
    }
}

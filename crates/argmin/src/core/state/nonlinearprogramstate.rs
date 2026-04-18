// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{ArgminFloat, Problem, State, TerminationReason, TerminationStatus};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use web_time::Duration;

/// Maintains the state from iteration to iteration of a solver
///
/// This struct is passed from one iteration of an algorithm to the next.
///
/// Keeps track of
///
/// * parameter vector of current and previous iteration
/// * slack variables of current and previous iteration
/// * best parameter vector of current and previous iteration
/// * best slack variables of current and previous iteration
/// * gradient of current and previous iteration
/// * Lagrangian Hessian of current and previous iteration
/// * equality constraints of current and previous iteration
/// * inequality constraints of current and previous iteration
/// * slack variables of current and previous iteration
/// * Jacobian of equality constraints of current and previous iteration
/// * Jacobian of inequality constraints of current and previous iteration
/// * barrier parameter of current and previous iteration
/// * equality multipliers of current and previous iteration
/// * inequality multipliers of current and previous iteration
/// * primal step size of current and previous iteration
/// * dual step size of current and previous iteration
/// * primal infeasibility of current and previous iteration
/// * dual infeasibility of current and previous iteration
/// * complementarity infeasibility of current and previous iteration
/// * cost function value of current and previous iteration
/// * current and previous best cost function value
/// * target cost function value
/// * current iteration number
/// * iteration number where the last best parameter vector was found
/// * maximum number of iterations that will be executed
/// * problem function evaluation counts (cost function, gradient, Lagrangian Hessian, ...)
/// * elapsed time
/// * termination status
#[derive(Clone, Default, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct NonLinearProgramState<
    P,
    G,
    LH,
    F,
    EqC = (),
    IneqC = (),
    S = (),
    EqCJ = (),
    IneqCJ = (),
    LambdaEq = (),
    LambdaIneq = (),
> {
    /// Current parameter vector
    pub param: Option<P>,
    /// Previous parameter vector
    pub prev_param: Option<P>,
    /// Current slack variables
    pub slacks: Option<S>,
    /// Previous slack variables
    pub prev_slacks: Option<S>,
    /// Current best parameter vector
    pub best_param: Option<P>,
    /// Previous best parameter vector
    pub prev_best_param: Option<P>,
    /// Current best slack variables
    pub best_slacks: Option<S>,
    /// Previous best slack variables
    pub prev_best_slacks: Option<S>,
    /// Current cost function value
    pub cost: F,
    /// Previous cost function value
    pub prev_cost: F,
    /// Current best cost function value
    pub best_cost: F,
    /// Previous best cost function value
    pub prev_best_cost: F,
    /// Target cost function value
    pub target_cost: F,
    /// Current gradient
    pub grad: Option<G>,
    /// Previous gradient
    pub prev_grad: Option<G>,
    /// Current Lagrangian Hessian
    pub lagrangian_hessian: Option<LH>,
    /// Previous Lagrangian Hessian
    pub prev_lagrangian_hessian: Option<LH>,
    /// Current value of equality constraints
    pub equality_constraints: Option<EqC>,
    /// Previous value of equality constraints
    pub prev_equality_constraints: Option<EqC>,
    /// Current value of inequality constraints
    pub inequality_constraints: Option<IneqC>,
    /// Previous value of inequality constraints
    pub prev_inequality_constraints: Option<IneqC>,
    /// Current Jacobian of equality constraints
    pub equality_constraint_jacobian: Option<EqCJ>,
    /// Previous Jacobian of equality constraints
    pub prev_equality_constraint_jacobian: Option<EqCJ>,
    /// Current Jacobian of inequality constraints
    pub inequality_constraint_jacobian: Option<IneqCJ>,
    /// Previous Jacobian of inequality constraints
    pub prev_inequality_constraint_jacobian: Option<IneqCJ>,
    /// Current barrier parameter
    pub mu: Option<F>,
    /// Previous barrier parameter
    pub prev_mu: Option<F>,
    /// Current equality multipliers
    pub lambda_eq: Option<LambdaEq>,
    /// Previous equality multipliers
    pub prev_lambda_eq: Option<LambdaEq>,
    /// Current inequality multipliers
    pub lambda_ineq: Option<LambdaIneq>,
    /// Previous inequality multipliers
    pub prev_lambda_ineq: Option<LambdaIneq>,
    /// Current primal step size
    pub alpha_primal: Option<F>,
    /// Previous primal step size
    pub prev_alpha_primal: Option<F>,
    /// Current dual step size
    pub alpha_dual: Option<F>,
    /// Previous dual step size
    pub prev_alpha_dual: Option<F>,
    /// Current primal infeasibility
    pub inf_pr: Option<F>,
    /// Previous primal infeasibility
    pub prev_inf_pr: Option<F>,
    /// Current dual infeasibility
    pub inf_du: Option<F>,
    /// Previous dual infeasibility
    pub prev_inf_du: Option<F>,
    /// Current complementarity infeasibility
    pub compl_inf: Option<F>,
    /// Previous complementarity infeasibility
    pub prev_compl_inf: Option<F>,
    /// Current iteration
    pub iter: u64,
    /// Iteration number of last best cost
    pub last_best_iter: u64,
    /// Maximum number of iterations
    pub max_iters: u64,
    /// Evaluation counts
    pub counts: HashMap<String, u64>,
    /// Update evaluation counts?
    pub counting_enabled: bool,
    /// Time required so far
    pub time: Option<Duration>,
    /// Status of optimization execution
    pub termination_status: TerminationStatus,
}

impl<P, G, LH, F, EqC, IneqC, S, EqCJ, IneqCJ, LambdaEq, LambdaIneq>
    NonLinearProgramState<P, G, LH, F, EqC, IneqC, S, EqCJ, IneqCJ, LambdaEq, LambdaIneq>
where
    Self: State<Float = F>,
    F: ArgminFloat,
{
    /// Set parameter vector. This shifts the stored parameter vector to the previous parameter
    /// vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State};
    /// # let state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # let param_old = vec![1.0f64, 2.0f64];
    /// # let state = state.param(param_old);
    /// # assert!(state.prev_param.is_none());
    /// # assert_eq!(state.param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # let param = vec![0.0f64, 3.0f64];
    /// let state = state.param(param);
    /// # assert_eq!(state.prev_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.param.as_ref().unwrap()[0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// # assert_eq!(state.param.as_ref().unwrap()[1].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn param(mut self, param: P) -> Self {
        std::mem::swap(&mut self.prev_param, &mut self.param);
        self.param = Some(param);
        self
    }

    /// Set slack variables. This shifts the stored slack variables to the previous slack
    /// variables.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State};
    /// # let state: NonLinearProgramState<(), (), (), f64, (), (), Vec<f64>> = NonLinearProgramState::new();
    /// # let slacks_old = vec![1.0f64, 2.0f64];
    /// # let state = state.slacks(slacks_old);
    /// # assert!(state.prev_slacks.is_none());
    /// # assert_eq!(state.slacks.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.slacks.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # let slacks = vec![0.0f64, 3.0f64];
    /// let state = state.slacks(slacks);
    /// # assert_eq!(state.prev_slacks.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_slacks.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.slacks.as_ref().unwrap()[0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// # assert_eq!(state.slacks.as_ref().unwrap()[1].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn slacks(mut self, slacks: S) -> Self {
        std::mem::swap(&mut self.prev_slacks, &mut self.slacks);
        self.slacks = Some(slacks);
        self
    }

    /// Set gradient. This shifts the stored gradient to the previous gradient.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State};
    /// # let state: NonLinearProgramState<(), Vec<f64>, (), f64> = NonLinearProgramState::new();
    /// # let grad_old = vec![1.0f64, 2.0f64];
    /// # let state = state.gradient(grad_old);
    /// # assert!(state.prev_grad.is_none());
    /// # assert_eq!(state.grad.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.grad.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # let grad = vec![0.0f64, 3.0f64];
    /// let state = state.gradient(grad);
    /// # assert_eq!(state.prev_grad.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_grad.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.grad.as_ref().unwrap()[0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// # assert_eq!(state.grad.as_ref().unwrap()[1].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn gradient(mut self, gradient: G) -> Self {
        std::mem::swap(&mut self.prev_grad, &mut self.grad);
        self.grad = Some(gradient);
        self
    }

    /// Set Lagrangian Hessian. This shifts the stored Lagrangian Hessian to the previous Lagrangian Hessian.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State};
    /// # let state: NonLinearProgramState<(), (), Vec<f64>, f64> = NonLinearProgramState::new();
    /// # let lagrangian_hessian_old = vec![1.0f64, 2.0f64];
    /// # let state = state.lagrangian_hessian(lagrangian_hessian_old);
    /// # assert!(state.prev_lagrangian_hessian.is_none());
    /// # assert_eq!(state.lagrangian_hessian.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.lagrangian_hessian.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # let lagrangian_hessian = vec![0.0f64, 3.0f64];
    /// let state = state.lagrangian_hessian(lagrangian_hessian);
    /// # assert_eq!(state.prev_lagrangian_hessian.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_lagrangian_hessian.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.lagrangian_hessian.as_ref().unwrap()[0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// # assert_eq!(state.lagrangian_hessian.as_ref().unwrap()[1].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn lagrangian_hessian(mut self, lagrangian_hessian: LH) -> Self {
        std::mem::swap(
            &mut self.prev_lagrangian_hessian,
            &mut self.lagrangian_hessian,
        );
        self.lagrangian_hessian = Some(lagrangian_hessian);
        self
    }

    /// Set equality constraints. This shifts the stored equality constraints to the previous
    /// equality constraints.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State};
    /// # let state: NonLinearProgramState<(), (), (), f64, Vec<f64>> = NonLinearProgramState::new();
    /// # let equality_constraints_old = vec![1.0f64, 2.0f64];
    /// # let state = state.equality_constraints(equality_constraints_old);
    /// # assert!(state.prev_equality_constraints.is_none());
    /// # assert_eq!(state.equality_constraints.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.equality_constraints.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # let equality_constraints = vec![0.0f64, 3.0f64];
    /// let state = state.equality_constraints(equality_constraints);
    /// # assert_eq!(state.prev_equality_constraints.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_equality_constraints.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.equality_constraints.as_ref().unwrap()[0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// # assert_eq!(state.equality_constraints.as_ref().unwrap()[1].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn equality_constraints(mut self, equality_constraints: EqC) -> Self {
        std::mem::swap(
            &mut self.prev_equality_constraints,
            &mut self.equality_constraints,
        );
        self.equality_constraints = Some(equality_constraints);
        self
    }

    /// Set inequality constraints. This shifts the stored inequality constraints to the previous
    /// inequality constraints.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State};
    /// # let state: NonLinearProgramState<(), (), (), f64, (), Vec<f64>> = NonLinearProgramState::new();
    /// # let inequality_constraints_old = vec![1.0f64, 2.0f64];
    /// # let state = state.inequality_constraints(inequality_constraints_old);
    /// # assert!(state.prev_inequality_constraints.is_none());
    /// # assert_eq!(state.inequality_constraints.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.inequality_constraints.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # let inequality_constraints = vec![0.0f64, 3.0f64];
    /// let state = state.inequality_constraints(inequality_constraints);
    /// # assert_eq!(state.prev_inequality_constraints.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_inequality_constraints.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.inequality_constraints.as_ref().unwrap()[0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// # assert_eq!(state.inequality_constraints.as_ref().unwrap()[1].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn inequality_constraints(mut self, inequality_constraints: IneqC) -> Self {
        std::mem::swap(
            &mut self.prev_inequality_constraints,
            &mut self.inequality_constraints,
        );
        self.inequality_constraints = Some(inequality_constraints);
        self
    }

    /// Set the Jacobian of equality constraints. This shifts the stored Jacobian of equality
    /// constraints to the previous Jacobian of equality constraints.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State};
    /// # let state: NonLinearProgramState<(), (), (), f64, (), (), (), Vec<Vec<f64>>> = NonLinearProgramState::new();
    /// # let equality_constraint_jacobian_old = vec![vec![1.0f64, 2.0f64], vec![3.0f64, 4.0f64]];
    /// # let state = state.equality_constraint_jacobian(equality_constraint_jacobian_old);
    /// # assert!(state.prev_equality_constraint_jacobian.is_none());
    /// # assert_eq!(state.equality_constraint_jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.equality_constraint_jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.equality_constraint_jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.equality_constraint_jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// # let equality_constraint_jacobian = vec![vec![0.0f64, 3.0f64], vec![4.0f64, 5.0f64]];
    /// let state = state.equality_constraint_jacobian(equality_constraint_jacobian);
    /// # assert_eq!(state.prev_equality_constraint_jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_equality_constraint_jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_equality_constraint_jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_equality_constraint_jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// # assert_eq!(state.equality_constraint_jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// # assert_eq!(state.equality_constraint_jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.equality_constraint_jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// # assert_eq!(state.equality_constraint_jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 5.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn equality_constraint_jacobian(mut self, equality_constraint_jacobian: EqCJ) -> Self {
        std::mem::swap(
            &mut self.prev_equality_constraint_jacobian,
            &mut self.equality_constraint_jacobian,
        );
        self.equality_constraint_jacobian = Some(equality_constraint_jacobian);
        self
    }

    /// Set the Jacobian of inequality constraints. This shifts the stored Jacobian of inequality
    /// constraints to the previous Jacobian of inequality constraints.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State};
    /// # let state: NonLinearProgramState<(), (), (), f64, (), (), (), (), Vec<Vec<f64>>> = NonLinearProgramState::new();
    /// # let inequality_constraint_jacobian_old = vec![vec![1.0f64, 2.0f64], vec![3.0f64, 4.0f64]];
    /// # let state = state.inequality_constraint_jacobian(inequality_constraint_jacobian_old);
    /// # assert!(state.prev_inequality_constraint_jacobian.is_none());
    /// # assert_eq!(state.inequality_constraint_jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.inequality_constraint_jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.inequality_constraint_jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.inequality_constraint_jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// # let inequality_constraint_jacobian = vec![vec![0.0f64, 3.0f64], vec![4.0f64, 5.0f64]];
    /// let state = state.inequality_constraint_jacobian(inequality_constraint_jacobian);
    /// # assert_eq!(state.prev_inequality_constraint_jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_inequality_constraint_jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_inequality_constraint_jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_inequality_constraint_jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// # assert_eq!(state.inequality_constraint_jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// # assert_eq!(state.inequality_constraint_jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.inequality_constraint_jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// # assert_eq!(state.inequality_constraint_jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 5.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn inequality_constraint_jacobian(
        mut self,
        inequality_constraint_jacobian: IneqCJ,
    ) -> Self {
        std::mem::swap(
            &mut self.prev_inequality_constraint_jacobian,
            &mut self.inequality_constraint_jacobian,
        );
        self.inequality_constraint_jacobian = Some(inequality_constraint_jacobian);
        self
    }

    /// Set barrier parameter. This shifts the stored barrier parameter to the previous barrier
    /// parameter.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State};
    /// # let state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # let state = state.mu(1.0f64);
    /// # assert!(state.prev_mu.is_none());
    /// # assert_eq!(state.mu.unwrap().to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// let state = state.mu(0.5f64);
    /// # assert_eq!(state.prev_mu.unwrap().to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.mu.unwrap().to_ne_bytes(), 0.5f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn mu(mut self, mu: F) -> Self {
        std::mem::swap(&mut self.prev_mu, &mut self.mu);
        self.mu = Some(mu);
        self
    }

    /// Set equality multipliers. This shifts the stored equality multipliers to the previous
    /// equality multipliers.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State};
    /// # let state: NonLinearProgramState<(), (), (), f64, (), (), (), (), (), Vec<f64>> =
    /// #     NonLinearProgramState::new();
    /// # let lambda_eq_old = vec![1.0f64, 2.0f64];
    /// # let state = state.lambda_eq(lambda_eq_old);
    /// # assert!(state.prev_lambda_eq.is_none());
    /// # assert_eq!(state.lambda_eq.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.lambda_eq.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # let lambda_eq = vec![0.0f64, 3.0f64];
    /// let state = state.lambda_eq(lambda_eq);
    /// # assert_eq!(state.prev_lambda_eq.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_lambda_eq.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.lambda_eq.as_ref().unwrap()[0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// # assert_eq!(state.lambda_eq.as_ref().unwrap()[1].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn lambda_eq(mut self, lambda_eq: LambdaEq) -> Self {
        std::mem::swap(&mut self.prev_lambda_eq, &mut self.lambda_eq);
        self.lambda_eq = Some(lambda_eq);
        self
    }

    /// Set inequality multipliers. This shifts the stored inequality multipliers to the previous
    /// inequality multipliers.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State};
    /// # let state: NonLinearProgramState<(), (), (), f64, (), (), (), (), (), (), Vec<f64>> =
    /// #     NonLinearProgramState::new();
    /// # let lambda_ineq_old = vec![1.0f64, 2.0f64];
    /// # let state = state.lambda_ineq(lambda_ineq_old);
    /// # assert!(state.prev_lambda_ineq.is_none());
    /// # assert_eq!(state.lambda_ineq.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.lambda_ineq.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # let lambda_ineq = vec![0.0f64, 3.0f64];
    /// let state = state.lambda_ineq(lambda_ineq);
    /// # assert_eq!(state.prev_lambda_ineq.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_lambda_ineq.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.lambda_ineq.as_ref().unwrap()[0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// # assert_eq!(state.lambda_ineq.as_ref().unwrap()[1].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn lambda_ineq(mut self, lambda_ineq: LambdaIneq) -> Self {
        std::mem::swap(&mut self.prev_lambda_ineq, &mut self.lambda_ineq);
        self.lambda_ineq = Some(lambda_ineq);
        self
    }

    /// Set primal step size. This shifts the stored primal step size to the previous primal step
    /// size.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State};
    /// # let state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # let state = state.alpha_primal(1.0f64);
    /// # assert!(state.prev_alpha_primal.is_none());
    /// # assert_eq!(state.alpha_primal.unwrap().to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// let state = state.alpha_primal(0.75f64);
    /// # assert_eq!(state.prev_alpha_primal.unwrap().to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.alpha_primal.unwrap().to_ne_bytes(), 0.75f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn alpha_primal(mut self, alpha_primal: F) -> Self {
        std::mem::swap(&mut self.prev_alpha_primal, &mut self.alpha_primal);
        self.alpha_primal = Some(alpha_primal);
        self
    }

    /// Set dual step size. This shifts the stored dual step size to the previous dual step size.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State};
    /// # let state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # let state = state.alpha_dual(1.0f64);
    /// # assert!(state.prev_alpha_dual.is_none());
    /// # assert_eq!(state.alpha_dual.unwrap().to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// let state = state.alpha_dual(0.8f64);
    /// # assert_eq!(state.prev_alpha_dual.unwrap().to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.alpha_dual.unwrap().to_ne_bytes(), 0.8f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn alpha_dual(mut self, alpha_dual: F) -> Self {
        std::mem::swap(&mut self.prev_alpha_dual, &mut self.alpha_dual);
        self.alpha_dual = Some(alpha_dual);
        self
    }

    /// Set primal infeasibility. This shifts the stored primal infeasibility to the previous
    /// primal infeasibility.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State};
    /// # let state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # let state = state.inf_pr(1.0f64);
    /// # assert!(state.prev_inf_pr.is_none());
    /// # assert_eq!(state.inf_pr.unwrap().to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// let state = state.inf_pr(0.1f64);
    /// # assert_eq!(state.prev_inf_pr.unwrap().to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.inf_pr.unwrap().to_ne_bytes(), 0.1f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn inf_pr(mut self, inf_pr: F) -> Self {
        std::mem::swap(&mut self.prev_inf_pr, &mut self.inf_pr);
        self.inf_pr = Some(inf_pr);
        self
    }

    /// Set dual infeasibility. This shifts the stored dual infeasibility to the previous dual
    /// infeasibility.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State};
    /// # let state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # let state = state.inf_du(1.0f64);
    /// # assert!(state.prev_inf_du.is_none());
    /// # assert_eq!(state.inf_du.unwrap().to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// let state = state.inf_du(0.05f64);
    /// # assert_eq!(state.prev_inf_du.unwrap().to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.inf_du.unwrap().to_ne_bytes(), 0.05f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn inf_du(mut self, inf_du: F) -> Self {
        std::mem::swap(&mut self.prev_inf_du, &mut self.inf_du);
        self.inf_du = Some(inf_du);
        self
    }

    /// Set complementarity infeasibility. This shifts the stored complementarity infeasibility to
    /// the previous complementarity infeasibility.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State};
    /// # let state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # let state = state.compl_inf(1.0f64);
    /// # assert!(state.prev_compl_inf.is_none());
    /// # assert_eq!(state.compl_inf.unwrap().to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// let state = state.compl_inf(0.01f64);
    /// # assert_eq!(state.prev_compl_inf.unwrap().to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.compl_inf.unwrap().to_ne_bytes(), 0.01f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn compl_inf(mut self, compl_inf: F) -> Self {
        std::mem::swap(&mut self.prev_compl_inf, &mut self.compl_inf);
        self.compl_inf = Some(compl_inf);
        self
    }

    /// Set the current cost function value. This shifts the stored cost function value to the
    /// previous cost function value.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State};
    /// # let state: NonLinearProgramState<(), (), Vec<f64>, f64> = NonLinearProgramState::new();
    /// # let cost_old = 1.0f64;
    /// # let state = state.cost(cost_old);
    /// # assert_eq!(state.prev_cost.to_ne_bytes(), f64::INFINITY.to_ne_bytes());
    /// # assert_eq!(state.cost.to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # let cost = 0.0f64;
    /// let state = state.cost(cost);
    /// # assert_eq!(state.prev_cost.to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.cost.to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn cost(mut self, cost: F) -> Self {
        std::mem::swap(&mut self.prev_cost, &mut self.cost);
        self.cost = cost;
        self
    }

    /// Set target cost.
    ///
    /// When this cost is reached, the algorithm will stop. The default is
    /// `Self::Float::NEG_INFINITY`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # assert_eq!(state.target_cost.to_ne_bytes(), f64::NEG_INFINITY.to_ne_bytes());
    /// let state = state.target_cost(0.0);
    /// # assert_eq!(state.target_cost.to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn target_cost(mut self, target_cost: F) -> Self {
        self.target_cost = target_cost;
        self
    }

    /// Set maximum number of iterations
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # assert_eq!(state.max_iters, u64::MAX);
    /// let state = state.max_iters(1000);
    /// # assert_eq!(state.max_iters, 1000);
    /// ```
    #[must_use]
    pub fn max_iters(mut self, iters: u64) -> Self {
        self.max_iters = iters;
        self
    }

    /// Returns the current cost function value
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # let state = state.cost(2.0);
    /// let cost = state.get_cost();
    /// # assert_eq!(cost.to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_cost(&self) -> F {
        self.cost
    }

    /// Returns the previous cost function value
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # state.prev_cost = 2.0;
    /// let prev_cost = state.get_prev_cost();
    /// # assert_eq!(prev_cost.to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_cost(&self) -> F {
        self.prev_cost
    }

    /// Returns the current best cost function value
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # state.best_cost = 2.0;
    /// let best_cost = state.get_best_cost();
    /// # assert_eq!(best_cost.to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_best_cost(&self) -> F {
        self.best_cost
    }

    /// Returns the previous best cost function value
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # state.prev_best_cost = 2.0;
    /// let prev_best_cost = state.get_prev_best_cost();
    /// # assert_eq!(prev_best_cost.to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_best_cost(&self) -> F {
        self.prev_best_cost
    }

    /// Returns the target cost function value
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # assert_eq!(state.target_cost.to_ne_bytes(), f64::NEG_INFINITY.to_ne_bytes());
    /// # state.target_cost = 0.0;
    /// let target_cost = state.get_target_cost();
    /// # assert_eq!(target_cost.to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// ```
    pub fn get_target_cost(&self) -> F {
        self.target_cost
    }

    /// Moves the current parameter vector out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.take_param().is_none());
    /// # let mut state = state.param(vec![1.0, 2.0]);
    /// # assert_eq!(state.param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let param = state.take_param();  // Option<P>
    /// # assert!(state.take_param().is_none());
    /// # assert!(state.param.is_none());
    /// # assert_eq!(param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_param(&mut self) -> Option<P> {
        self.param.take()
    }

    /// Returns a reference to previous parameter vector
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.prev_param.is_none());
    /// # state.prev_param = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.prev_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let prev_param = state.get_prev_param();  // Option<&P>
    /// # assert_eq!(prev_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_param(&self) -> Option<&P> {
        self.prev_param.as_ref()
    }

    /// Moves the previous parameter vector out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.take_prev_param().is_none());
    /// # state.prev_param = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.prev_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let prev_param = state.take_prev_param();  // Option<P>
    /// # assert!(state.take_prev_param().is_none());
    /// # assert!(state.prev_param.is_none());
    /// # assert_eq!(prev_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_prev_param(&mut self) -> Option<P> {
        self.prev_param.take()
    }

    /// Returns a reference to previous best parameter vector
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.prev_best_param.is_none());
    /// # state.prev_best_param = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.prev_best_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_best_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let prev_best_param = state.get_prev_best_param();  // Option<&P>
    /// # assert_eq!(prev_best_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_best_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_best_param(&self) -> Option<&P> {
        self.prev_best_param.as_ref()
    }

    /// Moves the best parameter vector out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.take_best_param().is_none());
    /// # state.best_param = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.best_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.best_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let best_param = state.take_best_param();  // Option<P>
    /// # assert!(state.take_best_param().is_none());
    /// # assert!(state.best_param.is_none());
    /// # assert_eq!(best_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(best_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_best_param(&mut self) -> Option<P> {
        self.best_param.take()
    }

    /// Moves the previous best parameter vector out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.take_prev_best_param().is_none());
    /// # state.prev_best_param = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.prev_best_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_best_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let prev_best_param = state.take_prev_best_param();  // Option<P>
    /// # assert!(state.take_prev_best_param().is_none());
    /// # assert!(state.prev_best_param.is_none());
    /// # assert_eq!(prev_best_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_best_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_prev_best_param(&mut self) -> Option<P> {
        self.prev_best_param.take()
    }

    /// Returns a reference to the slack variables
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), (), Vec<f64>> = NonLinearProgramState::new();
    /// # assert!(state.slacks.is_none());
    /// # state.slacks = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.slacks.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.slacks.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let slacks = state.get_slacks();  // Option<&S>
    /// # assert_eq!(slacks.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(slacks.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_slacks(&self) -> Option<&S> {
        self.slacks.as_ref()
    }

    /// Moves the slack variables out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), (), Vec<f64>> = NonLinearProgramState::new();
    /// # assert!(state.take_slacks().is_none());
    /// # state.slacks = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.slacks.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.slacks.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let slacks = state.take_slacks();  // Option<S>
    /// # assert!(state.take_slacks().is_none());
    /// # assert!(state.slacks.is_none());
    /// # assert_eq!(slacks.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(slacks.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_slacks(&mut self) -> Option<S> {
        self.slacks.take()
    }

    /// Returns a reference to the previous slack variables
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), (), Vec<f64>> = NonLinearProgramState::new();
    /// # assert!(state.prev_slacks.is_none());
    /// # state.prev_slacks = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.prev_slacks.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_slacks.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let prev_slacks = state.get_prev_slacks();  // Option<&S>
    /// # assert_eq!(prev_slacks.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_slacks.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_slacks(&self) -> Option<&S> {
        self.prev_slacks.as_ref()
    }

    /// Moves the previous slack variables out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), (), Vec<f64>> = NonLinearProgramState::new();
    /// # assert!(state.take_prev_slacks().is_none());
    /// # state.prev_slacks = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.prev_slacks.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_slacks.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let prev_slacks = state.take_prev_slacks();  // Option<S>
    /// # assert!(state.take_prev_slacks().is_none());
    /// # assert!(state.prev_slacks.is_none());
    /// # assert_eq!(prev_slacks.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_slacks.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_prev_slacks(&mut self) -> Option<S> {
        self.prev_slacks.take()
    }

    /// Returns a reference to the current best slack variables
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), (), Vec<f64>> = NonLinearProgramState::new();
    /// # assert!(state.best_slacks.is_none());
    /// # state.best_slacks = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.best_slacks.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.best_slacks.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let best_slacks = state.get_best_slacks();  // Option<&S>
    /// # assert_eq!(best_slacks.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(best_slacks.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_best_slacks(&self) -> Option<&S> {
        self.best_slacks.as_ref()
    }

    /// Moves the best slack variables out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), (), Vec<f64>> = NonLinearProgramState::new();
    /// # assert!(state.take_best_slacks().is_none());
    /// # state.best_slacks = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.best_slacks.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.best_slacks.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let best_slacks = state.take_best_slacks();  // Option<S>
    /// # assert!(state.take_best_slacks().is_none());
    /// # assert!(state.best_slacks.is_none());
    /// # assert_eq!(best_slacks.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(best_slacks.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_best_slacks(&mut self) -> Option<S> {
        self.best_slacks.take()
    }

    /// Returns a reference to the previous best slack variables
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), (), Vec<f64>> = NonLinearProgramState::new();
    /// # assert!(state.prev_best_slacks.is_none());
    /// # state.prev_best_slacks = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.prev_best_slacks.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_best_slacks.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let prev_best_slacks = state.get_prev_best_slacks();  // Option<&S>
    /// # assert_eq!(prev_best_slacks.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_best_slacks.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_best_slacks(&self) -> Option<&S> {
        self.prev_best_slacks.as_ref()
    }

    /// Moves the previous best slack variables out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), (), Vec<f64>> = NonLinearProgramState::new();
    /// # assert!(state.take_prev_best_slacks().is_none());
    /// # state.prev_best_slacks = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.prev_best_slacks.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_best_slacks.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let prev_best_slacks = state.take_prev_best_slacks();  // Option<S>
    /// # assert!(state.take_prev_best_slacks().is_none());
    /// # assert!(state.prev_best_slacks.is_none());
    /// # assert_eq!(prev_best_slacks.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_best_slacks.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_prev_best_slacks(&mut self) -> Option<S> {
        self.prev_best_slacks.take()
    }

    /// Returns a reference to the gradient
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), Vec<f64>, (), f64> = NonLinearProgramState::new();
    /// # assert!(state.grad.is_none());
    /// # assert!(state.get_gradient().is_none());
    /// # state.grad = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.grad.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.grad.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let grad = state.get_gradient();  // Option<&G>
    /// # assert_eq!(grad.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(grad.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_gradient(&self) -> Option<&G> {
        self.grad.as_ref()
    }

    /// Moves the gradient out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), Vec<f64>, (), f64> = NonLinearProgramState::new();
    /// # assert!(state.take_gradient().is_none());
    /// # state.grad = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.grad.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.grad.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let grad = state.take_gradient();  // Option<G>
    /// # assert!(state.take_gradient().is_none());
    /// # assert!(state.grad.is_none());
    /// # assert_eq!(grad.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(grad.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_gradient(&mut self) -> Option<G> {
        self.grad.take()
    }

    /// Returns a reference to the previous gradient
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), Vec<f64>, (), f64> = NonLinearProgramState::new();
    /// # assert!(state.prev_grad.is_none());
    /// # assert!(state.get_prev_gradient().is_none());
    /// # state.prev_grad = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.prev_grad.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_grad.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let prev_grad = state.get_prev_gradient();  // Option<&G>
    /// # assert_eq!(prev_grad.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_grad.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_gradient(&self) -> Option<&G> {
        self.prev_grad.as_ref()
    }

    /// Moves the gradient out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), Vec<f64>, (), f64> = NonLinearProgramState::new();
    /// # assert!(state.take_prev_gradient().is_none());
    /// # state.prev_grad = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.prev_grad.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_grad.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let prev_grad = state.take_prev_gradient();  // Option<G>
    /// # assert!(state.take_prev_gradient().is_none());
    /// # assert!(state.prev_grad.is_none());
    /// # assert_eq!(prev_grad.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_grad.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_prev_gradient(&mut self) -> Option<G> {
        self.prev_grad.take()
    }

    /// Returns a reference to the current Lagrangian Hessian
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), Vec<Vec<f64>>, f64> = NonLinearProgramState::new();
    /// # assert!(state.lagrangian_hessian.is_none());
    /// # assert!(state.get_lagrangian_hessian().is_none());
    /// # state.lagrangian_hessian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(state.lagrangian_hessian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.lagrangian_hessian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.lagrangian_hessian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.lagrangian_hessian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// let lagrangian_hessian = state.get_lagrangian_hessian();  // Option<&LH>
    /// # assert_eq!(lagrangian_hessian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(lagrangian_hessian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(lagrangian_hessian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(lagrangian_hessian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// ```
    pub fn get_lagrangian_hessian(&self) -> Option<&LH> {
        self.lagrangian_hessian.as_ref()
    }

    /// Moves the Lagrangian Hessian out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), Vec<Vec<f64>>, f64> = NonLinearProgramState::new();
    /// # assert!(state.lagrangian_hessian.is_none());
    /// # assert!(state.take_lagrangian_hessian().is_none());
    /// # state.lagrangian_hessian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(state.lagrangian_hessian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.lagrangian_hessian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.lagrangian_hessian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.lagrangian_hessian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// let lagrangian_hessian = state.take_lagrangian_hessian();  // Option<LH>
    /// # assert!(state.take_lagrangian_hessian().is_none());
    /// # assert!(state.lagrangian_hessian.is_none());
    /// # assert_eq!(lagrangian_hessian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(lagrangian_hessian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(lagrangian_hessian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(lagrangian_hessian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// ```
    pub fn take_lagrangian_hessian(&mut self) -> Option<LH> {
        self.lagrangian_hessian.take()
    }

    /// Returns a reference to the previous Lagrangian Hessian
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), Vec<Vec<f64>>, f64> = NonLinearProgramState::new();
    /// # assert!(state.prev_lagrangian_hessian.is_none());
    /// # assert!(state.get_prev_lagrangian_hessian().is_none());
    /// # state.prev_lagrangian_hessian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(state.prev_lagrangian_hessian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_lagrangian_hessian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_lagrangian_hessian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_lagrangian_hessian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// let prev_lagrangian_hessian = state.get_prev_lagrangian_hessian();  // Option<&LH>
    /// # assert_eq!(prev_lagrangian_hessian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_lagrangian_hessian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(prev_lagrangian_hessian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(prev_lagrangian_hessian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_lagrangian_hessian(&self) -> Option<&LH> {
        self.prev_lagrangian_hessian.as_ref()
    }

    /// Moves the previous Lagrangian Hessian out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), Vec<Vec<f64>>, f64> = NonLinearProgramState::new();
    /// # assert!(state.prev_lagrangian_hessian.is_none());
    /// # assert!(state.take_prev_lagrangian_hessian().is_none());
    /// # state.prev_lagrangian_hessian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(state.prev_lagrangian_hessian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_lagrangian_hessian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_lagrangian_hessian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_lagrangian_hessian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// let prev_lagrangian_hessian = state.take_prev_lagrangian_hessian();  // Option<LH>
    /// # assert!(state.take_prev_lagrangian_hessian().is_none());
    /// # assert!(state.prev_lagrangian_hessian.is_none());
    /// # assert_eq!(prev_lagrangian_hessian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_lagrangian_hessian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(prev_lagrangian_hessian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(prev_lagrangian_hessian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// ```
    pub fn take_prev_lagrangian_hessian(&mut self) -> Option<LH> {
        self.prev_lagrangian_hessian.take()
    }

    /// Returns a reference to the equality constraints
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, Vec<f64>> = NonLinearProgramState::new();
    /// # assert!(state.equality_constraints.is_none());
    /// # state.equality_constraints = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.equality_constraints.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.equality_constraints.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let equality_constraints = state.get_equality_constraints();  // Option<&EqC>
    /// # assert_eq!(equality_constraints.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(equality_constraints.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_equality_constraints(&self) -> Option<&EqC> {
        self.equality_constraints.as_ref()
    }

    /// Moves the equality constraints out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, Vec<f64>> = NonLinearProgramState::new();
    /// # assert!(state.take_equality_constraints().is_none());
    /// # state.equality_constraints = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.equality_constraints.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.equality_constraints.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let equality_constraints = state.take_equality_constraints();  // Option<EqC>
    /// # assert!(state.take_equality_constraints().is_none());
    /// # assert!(state.equality_constraints.is_none());
    /// # assert_eq!(equality_constraints.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(equality_constraints.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_equality_constraints(&mut self) -> Option<EqC> {
        self.equality_constraints.take()
    }

    /// Returns a reference to the previous equality constraints
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, Vec<f64>> = NonLinearProgramState::new();
    /// # assert!(state.prev_equality_constraints.is_none());
    /// # state.prev_equality_constraints = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.prev_equality_constraints.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_equality_constraints.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let equality_constraints = state.get_prev_equality_constraints();  // Option<&EqC>
    /// # assert_eq!(equality_constraints.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(equality_constraints.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_equality_constraints(&self) -> Option<&EqC> {
        self.prev_equality_constraints.as_ref()
    }

    /// Moves the previous equality constraints out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, Vec<f64>> = NonLinearProgramState::new();
    /// # assert!(state.take_prev_equality_constraints().is_none());
    /// # state.prev_equality_constraints = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.prev_equality_constraints.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_equality_constraints.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let equality_constraints = state.take_prev_equality_constraints();  // Option<EqC>
    /// # assert!(state.take_prev_equality_constraints().is_none());
    /// # assert!(state.prev_equality_constraints.is_none());
    /// # assert_eq!(equality_constraints.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(equality_constraints.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_prev_equality_constraints(&mut self) -> Option<EqC> {
        self.prev_equality_constraints.take()
    }

    /// Returns a reference to the inequality constraints
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), Vec<f64>> = NonLinearProgramState::new();
    /// # assert!(state.inequality_constraints.is_none());
    /// # state.inequality_constraints = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.inequality_constraints.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.inequality_constraints.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let inequality_constraints = state.get_inequality_constraints();  // Option<&IneqC>
    /// # assert_eq!(inequality_constraints.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(inequality_constraints.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_inequality_constraints(&self) -> Option<&IneqC> {
        self.inequality_constraints.as_ref()
    }

    /// Moves the inequality constraints out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), Vec<f64>> = NonLinearProgramState::new();
    /// # assert!(state.take_inequality_constraints().is_none());
    /// # state.inequality_constraints = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.inequality_constraints.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.inequality_constraints.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let inequality_constraints = state.take_inequality_constraints();  // Option<IneqC>
    /// # assert!(state.take_inequality_constraints().is_none());
    /// # assert!(state.inequality_constraints.is_none());
    /// # assert_eq!(inequality_constraints.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(inequality_constraints.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_inequality_constraints(&mut self) -> Option<IneqC> {
        self.inequality_constraints.take()
    }

    /// Returns a reference to the previous inequality constraints
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), Vec<f64>> = NonLinearProgramState::new();
    /// # assert!(state.prev_inequality_constraints.is_none());
    /// # state.prev_inequality_constraints = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.prev_inequality_constraints.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_inequality_constraints.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let inequality_constraints = state.get_prev_inequality_constraints();  // Option<&IneqC>
    /// # assert_eq!(inequality_constraints.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(inequality_constraints.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_inequality_constraints(&self) -> Option<&IneqC> {
        self.prev_inequality_constraints.as_ref()
    }

    /// Moves the previous inequality constraints out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), Vec<f64>> = NonLinearProgramState::new();
    /// # assert!(state.take_prev_inequality_constraints().is_none());
    /// # state.prev_inequality_constraints = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.prev_inequality_constraints.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_inequality_constraints.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let inequality_constraints = state.take_prev_inequality_constraints();  // Option<IneqC>
    /// # assert!(state.take_prev_inequality_constraints().is_none());
    /// # assert!(state.prev_inequality_constraints.is_none());
    /// # assert_eq!(inequality_constraints.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(inequality_constraints.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_prev_inequality_constraints(&mut self) -> Option<IneqC> {
        self.prev_inequality_constraints.take()
    }

    /// Returns a reference to the Jacobian of equality constraints
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), (), (), Vec<Vec<f64>>> = NonLinearProgramState::new();
    /// # assert!(state.equality_constraint_jacobian.is_none());
    /// # state.equality_constraint_jacobian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(state.equality_constraint_jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.equality_constraint_jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.equality_constraint_jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.equality_constraint_jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// let jacobian = state.get_equality_constraint_jacobian();  // Option<&EqCJ>
    /// # assert_eq!(jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// ```
    pub fn get_equality_constraint_jacobian(&self) -> Option<&EqCJ> {
        self.equality_constraint_jacobian.as_ref()
    }

    /// Moves the Jacobian of equality constraints out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), (), (), Vec<Vec<f64>>> = NonLinearProgramState::new();
    /// # assert!(state.take_equality_constraint_jacobian().is_none());
    /// # state.equality_constraint_jacobian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(state.equality_constraint_jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.equality_constraint_jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.equality_constraint_jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.equality_constraint_jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// let jacobian = state.take_equality_constraint_jacobian();  // Option<EqCJ>
    /// # assert!(state.take_equality_constraint_jacobian().is_none());
    /// # assert!(state.equality_constraint_jacobian.is_none());
    /// # assert_eq!(jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// ```
    pub fn take_equality_constraint_jacobian(&mut self) -> Option<EqCJ> {
        self.equality_constraint_jacobian.take()
    }

    /// Returns a reference to the previous Jacobian of equality constraints
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), (), (), Vec<Vec<f64>>> = NonLinearProgramState::new();
    /// # assert!(state.prev_equality_constraint_jacobian.is_none());
    /// # state.prev_equality_constraint_jacobian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(state.prev_equality_constraint_jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_equality_constraint_jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_equality_constraint_jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_equality_constraint_jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// let jacobian = state.get_prev_equality_constraint_jacobian();  // Option<&EqCJ>
    /// # assert_eq!(jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_equality_constraint_jacobian(&self) -> Option<&EqCJ> {
        self.prev_equality_constraint_jacobian.as_ref()
    }

    /// Moves the previous Jacobian of equality constraints out and replaces it internally with
    /// `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), (), (), Vec<Vec<f64>>> = NonLinearProgramState::new();
    /// # assert!(state.take_prev_equality_constraint_jacobian().is_none());
    /// # state.prev_equality_constraint_jacobian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(state.prev_equality_constraint_jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_equality_constraint_jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_equality_constraint_jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_equality_constraint_jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// let jacobian = state.take_prev_equality_constraint_jacobian();  // Option<EqCJ>
    /// # assert!(state.take_prev_equality_constraint_jacobian().is_none());
    /// # assert!(state.prev_equality_constraint_jacobian.is_none());
    /// # assert_eq!(jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// ```
    pub fn take_prev_equality_constraint_jacobian(&mut self) -> Option<EqCJ> {
        self.prev_equality_constraint_jacobian.take()
    }

    /// Returns a reference to the Jacobian of inequality constraints
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), (), (), (), Vec<Vec<f64>>> = NonLinearProgramState::new();
    /// # assert!(state.inequality_constraint_jacobian.is_none());
    /// # state.inequality_constraint_jacobian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(state.inequality_constraint_jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.inequality_constraint_jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.inequality_constraint_jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.inequality_constraint_jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// let jacobian = state.get_inequality_constraint_jacobian();  // Option<&IneqCJ>
    /// # assert_eq!(jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// ```
    pub fn get_inequality_constraint_jacobian(&self) -> Option<&IneqCJ> {
        self.inequality_constraint_jacobian.as_ref()
    }

    /// Moves the Jacobian of inequality constraints out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), (), (), (), Vec<Vec<f64>>> = NonLinearProgramState::new();
    /// # assert!(state.take_inequality_constraint_jacobian().is_none());
    /// # state.inequality_constraint_jacobian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(state.inequality_constraint_jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.inequality_constraint_jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.inequality_constraint_jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.inequality_constraint_jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// let jacobian = state.take_inequality_constraint_jacobian();  // Option<IneqCJ>
    /// # assert!(state.take_inequality_constraint_jacobian().is_none());
    /// # assert!(state.inequality_constraint_jacobian.is_none());
    /// # assert_eq!(jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// ```
    pub fn take_inequality_constraint_jacobian(&mut self) -> Option<IneqCJ> {
        self.inequality_constraint_jacobian.take()
    }

    /// Returns a reference to the previous Jacobian of inequality constraints
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), (), (), (), Vec<Vec<f64>>> = NonLinearProgramState::new();
    /// # assert!(state.prev_inequality_constraint_jacobian.is_none());
    /// # state.prev_inequality_constraint_jacobian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(state.prev_inequality_constraint_jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_inequality_constraint_jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_inequality_constraint_jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_inequality_constraint_jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// let jacobian = state.get_prev_inequality_constraint_jacobian();  // Option<&IneqCJ>
    /// # assert_eq!(jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_inequality_constraint_jacobian(&self) -> Option<&IneqCJ> {
        self.prev_inequality_constraint_jacobian.as_ref()
    }

    /// Moves the previous Jacobian of inequality constraints out and replaces it internally with
    /// `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), (), (), (), Vec<Vec<f64>>> = NonLinearProgramState::new();
    /// # assert!(state.take_prev_inequality_constraint_jacobian().is_none());
    /// # state.prev_inequality_constraint_jacobian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(state.prev_inequality_constraint_jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_inequality_constraint_jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_inequality_constraint_jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_inequality_constraint_jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// let jacobian = state.take_prev_inequality_constraint_jacobian();  // Option<IneqCJ>
    /// # assert!(state.take_prev_inequality_constraint_jacobian().is_none());
    /// # assert!(state.prev_inequality_constraint_jacobian.is_none());
    /// # assert_eq!(jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// ```
    pub fn take_prev_inequality_constraint_jacobian(&mut self) -> Option<IneqCJ> {
        self.prev_inequality_constraint_jacobian.take()
    }

    /// Returns a reference to the current barrier parameter
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # let state = state.mu(0.5);
    /// let mu = state.get_mu();  // Option<&F>
    /// # assert_eq!(mu.unwrap().to_ne_bytes(), 0.5f64.to_ne_bytes());
    /// ```
    pub fn get_mu(&self) -> Option<&F> {
        self.mu.as_ref()
    }

    /// Moves the current barrier parameter out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.take_mu().is_none());
    /// # let mut state = state.mu(0.5);
    /// let mu = state.take_mu();  // Option<F>
    /// # assert!(state.take_mu().is_none());
    /// # assert!(state.mu.is_none());
    /// # assert_eq!(mu.unwrap().to_ne_bytes(), 0.5f64.to_ne_bytes());
    /// ```
    pub fn take_mu(&mut self) -> Option<F> {
        self.mu.take()
    }

    /// Returns a reference to the previous barrier parameter
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.prev_mu.is_none());
    /// # state.prev_mu = Some(0.5);
    /// let prev_mu = state.get_prev_mu();  // Option<&F>
    /// # assert_eq!(prev_mu.unwrap().to_ne_bytes(), 0.5f64.to_ne_bytes());
    /// ```
    pub fn get_prev_mu(&self) -> Option<&F> {
        self.prev_mu.as_ref()
    }

    /// Moves the previous barrier parameter out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.take_prev_mu().is_none());
    /// # state.prev_mu = Some(0.5);
    /// let prev_mu = state.take_prev_mu();  // Option<F>
    /// # assert!(state.take_prev_mu().is_none());
    /// # assert!(state.prev_mu.is_none());
    /// # assert_eq!(prev_mu.unwrap().to_ne_bytes(), 0.5f64.to_ne_bytes());
    /// ```
    pub fn take_prev_mu(&mut self) -> Option<F> {
        self.prev_mu.take()
    }

    /// Returns a reference to the current equality multipliers
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let state: NonLinearProgramState<(), (), (), f64, (), (), (), (), (), Vec<f64>> =
    /// #     NonLinearProgramState::new();
    /// # let state = state.lambda_eq(vec![1.0, 2.0]);
    /// let lambda_eq = state.get_lambda_eq();  // Option<&LambdaEq>
    /// # assert_eq!(lambda_eq.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(lambda_eq.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_lambda_eq(&self) -> Option<&LambdaEq> {
        self.lambda_eq.as_ref()
    }

    /// Moves the current equality multipliers out and replaces them internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), (), (), (), (), Vec<f64>> =
    /// #     NonLinearProgramState::new();
    /// # assert!(state.take_lambda_eq().is_none());
    /// # let mut state = state.lambda_eq(vec![1.0, 2.0]);
    /// let lambda_eq = state.take_lambda_eq();  // Option<LambdaEq>
    /// # assert!(state.take_lambda_eq().is_none());
    /// # assert!(state.lambda_eq.is_none());
    /// # assert_eq!(lambda_eq.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(lambda_eq.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_lambda_eq(&mut self) -> Option<LambdaEq> {
        self.lambda_eq.take()
    }

    /// Returns a reference to the previous equality multipliers
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), (), (), (), (), Vec<f64>> =
    /// #     NonLinearProgramState::new();
    /// # assert!(state.prev_lambda_eq.is_none());
    /// # state.prev_lambda_eq = Some(vec![1.0, 2.0]);
    /// let prev_lambda_eq = state.get_prev_lambda_eq();  // Option<&LambdaEq>
    /// # assert_eq!(prev_lambda_eq.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_lambda_eq.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_lambda_eq(&self) -> Option<&LambdaEq> {
        self.prev_lambda_eq.as_ref()
    }

    /// Moves the previous equality multipliers out and replaces them internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), (), (), (), (), Vec<f64>> =
    /// #     NonLinearProgramState::new();
    /// # assert!(state.take_prev_lambda_eq().is_none());
    /// # state.prev_lambda_eq = Some(vec![1.0, 2.0]);
    /// let prev_lambda_eq = state.take_prev_lambda_eq();  // Option<LambdaEq>
    /// # assert!(state.take_prev_lambda_eq().is_none());
    /// # assert!(state.prev_lambda_eq.is_none());
    /// # assert_eq!(prev_lambda_eq.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_lambda_eq.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_prev_lambda_eq(&mut self) -> Option<LambdaEq> {
        self.prev_lambda_eq.take()
    }

    /// Returns a reference to the current inequality multipliers
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let state: NonLinearProgramState<(), (), (), f64, (), (), (), (), (), (), Vec<f64>> =
    /// #     NonLinearProgramState::new();
    /// # let state = state.lambda_ineq(vec![1.0, 2.0]);
    /// let lambda_ineq = state.get_lambda_ineq();  // Option<&LambdaIneq>
    /// # assert_eq!(lambda_ineq.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(lambda_ineq.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_lambda_ineq(&self) -> Option<&LambdaIneq> {
        self.lambda_ineq.as_ref()
    }

    /// Moves the current inequality multipliers out and replaces them internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), (), (), (), (), (), Vec<f64>> =
    /// #     NonLinearProgramState::new();
    /// # assert!(state.take_lambda_ineq().is_none());
    /// # let mut state = state.lambda_ineq(vec![1.0, 2.0]);
    /// let lambda_ineq = state.take_lambda_ineq();  // Option<LambdaIneq>
    /// # assert!(state.take_lambda_ineq().is_none());
    /// # assert!(state.lambda_ineq.is_none());
    /// # assert_eq!(lambda_ineq.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(lambda_ineq.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_lambda_ineq(&mut self) -> Option<LambdaIneq> {
        self.lambda_ineq.take()
    }

    /// Returns a reference to the previous inequality multipliers
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), (), (), (), (), (), Vec<f64>> =
    /// #     NonLinearProgramState::new();
    /// # assert!(state.prev_lambda_ineq.is_none());
    /// # state.prev_lambda_ineq = Some(vec![1.0, 2.0]);
    /// let prev_lambda_ineq = state.get_prev_lambda_ineq();  // Option<&LambdaIneq>
    /// # assert_eq!(prev_lambda_ineq.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_lambda_ineq.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_lambda_ineq(&self) -> Option<&LambdaIneq> {
        self.prev_lambda_ineq.as_ref()
    }

    /// Moves the previous inequality multipliers out and replaces them internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64, (), (), (), (), (), (), Vec<f64>> =
    /// #     NonLinearProgramState::new();
    /// # assert!(state.take_prev_lambda_ineq().is_none());
    /// # state.prev_lambda_ineq = Some(vec![1.0, 2.0]);
    /// let prev_lambda_ineq = state.take_prev_lambda_ineq();  // Option<LambdaIneq>
    /// # assert!(state.take_prev_lambda_ineq().is_none());
    /// # assert!(state.prev_lambda_ineq.is_none());
    /// # assert_eq!(prev_lambda_ineq.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_lambda_ineq.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_prev_lambda_ineq(&mut self) -> Option<LambdaIneq> {
        self.prev_lambda_ineq.take()
    }

    /// Returns a reference to the current primal step size
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # let state = state.alpha_primal(0.75);
    /// let alpha_primal = state.get_alpha_primal();  // Option<&F>
    /// # assert_eq!(alpha_primal.unwrap().to_ne_bytes(), 0.75f64.to_ne_bytes());
    /// ```
    pub fn get_alpha_primal(&self) -> Option<&F> {
        self.alpha_primal.as_ref()
    }

    /// Moves the current primal step size out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.take_alpha_primal().is_none());
    /// # let mut state = state.alpha_primal(0.75);
    /// let alpha_primal = state.take_alpha_primal();  // Option<F>
    /// # assert!(state.take_alpha_primal().is_none());
    /// # assert!(state.alpha_primal.is_none());
    /// # assert_eq!(alpha_primal.unwrap().to_ne_bytes(), 0.75f64.to_ne_bytes());
    /// ```
    pub fn take_alpha_primal(&mut self) -> Option<F> {
        self.alpha_primal.take()
    }

    /// Returns a reference to the previous primal step size
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.prev_alpha_primal.is_none());
    /// # state.prev_alpha_primal = Some(0.75);
    /// let prev_alpha_primal = state.get_prev_alpha_primal();  // Option<&F>
    /// # assert_eq!(prev_alpha_primal.unwrap().to_ne_bytes(), 0.75f64.to_ne_bytes());
    /// ```
    pub fn get_prev_alpha_primal(&self) -> Option<&F> {
        self.prev_alpha_primal.as_ref()
    }

    /// Moves the previous primal step size out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.take_prev_alpha_primal().is_none());
    /// # state.prev_alpha_primal = Some(0.75);
    /// let prev_alpha_primal = state.take_prev_alpha_primal();  // Option<F>
    /// # assert!(state.take_prev_alpha_primal().is_none());
    /// # assert!(state.prev_alpha_primal.is_none());
    /// # assert_eq!(prev_alpha_primal.unwrap().to_ne_bytes(), 0.75f64.to_ne_bytes());
    /// ```
    pub fn take_prev_alpha_primal(&mut self) -> Option<F> {
        self.prev_alpha_primal.take()
    }

    /// Returns a reference to the current dual step size
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # let state = state.alpha_dual(0.8);
    /// let alpha_dual = state.get_alpha_dual();  // Option<&F>
    /// # assert_eq!(alpha_dual.unwrap().to_ne_bytes(), 0.8f64.to_ne_bytes());
    /// ```
    pub fn get_alpha_dual(&self) -> Option<&F> {
        self.alpha_dual.as_ref()
    }

    /// Moves the current dual step size out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.take_alpha_dual().is_none());
    /// # let mut state = state.alpha_dual(0.8);
    /// let alpha_dual = state.take_alpha_dual();  // Option<F>
    /// # assert!(state.take_alpha_dual().is_none());
    /// # assert!(state.alpha_dual.is_none());
    /// # assert_eq!(alpha_dual.unwrap().to_ne_bytes(), 0.8f64.to_ne_bytes());
    /// ```
    pub fn take_alpha_dual(&mut self) -> Option<F> {
        self.alpha_dual.take()
    }

    /// Returns a reference to the previous dual step size
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.prev_alpha_dual.is_none());
    /// # state.prev_alpha_dual = Some(0.8);
    /// let prev_alpha_dual = state.get_prev_alpha_dual();  // Option<&F>
    /// # assert_eq!(prev_alpha_dual.unwrap().to_ne_bytes(), 0.8f64.to_ne_bytes());
    /// ```
    pub fn get_prev_alpha_dual(&self) -> Option<&F> {
        self.prev_alpha_dual.as_ref()
    }

    /// Moves the previous dual step size out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.take_prev_alpha_dual().is_none());
    /// # state.prev_alpha_dual = Some(0.8);
    /// let prev_alpha_dual = state.take_prev_alpha_dual();  // Option<F>
    /// # assert!(state.take_prev_alpha_dual().is_none());
    /// # assert!(state.prev_alpha_dual.is_none());
    /// # assert_eq!(prev_alpha_dual.unwrap().to_ne_bytes(), 0.8f64.to_ne_bytes());
    /// ```
    pub fn take_prev_alpha_dual(&mut self) -> Option<F> {
        self.prev_alpha_dual.take()
    }

    /// Returns a reference to the current primal infeasibility
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # let state = state.inf_pr(0.1);
    /// let inf_pr = state.get_inf_pr();  // Option<&F>
    /// # assert_eq!(inf_pr.unwrap().to_ne_bytes(), 0.1f64.to_ne_bytes());
    /// ```
    pub fn get_inf_pr(&self) -> Option<&F> {
        self.inf_pr.as_ref()
    }

    /// Moves the current primal infeasibility out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.take_inf_pr().is_none());
    /// # let mut state = state.inf_pr(0.1);
    /// let inf_pr = state.take_inf_pr();  // Option<F>
    /// # assert!(state.take_inf_pr().is_none());
    /// # assert!(state.inf_pr.is_none());
    /// # assert_eq!(inf_pr.unwrap().to_ne_bytes(), 0.1f64.to_ne_bytes());
    /// ```
    pub fn take_inf_pr(&mut self) -> Option<F> {
        self.inf_pr.take()
    }

    /// Returns a reference to the previous primal infeasibility
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.prev_inf_pr.is_none());
    /// # state.prev_inf_pr = Some(0.1);
    /// let prev_inf_pr = state.get_prev_inf_pr();  // Option<&F>
    /// # assert_eq!(prev_inf_pr.unwrap().to_ne_bytes(), 0.1f64.to_ne_bytes());
    /// ```
    pub fn get_prev_inf_pr(&self) -> Option<&F> {
        self.prev_inf_pr.as_ref()
    }

    /// Moves the previous primal infeasibility out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.take_prev_inf_pr().is_none());
    /// # state.prev_inf_pr = Some(0.1);
    /// let prev_inf_pr = state.take_prev_inf_pr();  // Option<F>
    /// # assert!(state.take_prev_inf_pr().is_none());
    /// # assert!(state.prev_inf_pr.is_none());
    /// # assert_eq!(prev_inf_pr.unwrap().to_ne_bytes(), 0.1f64.to_ne_bytes());
    /// ```
    pub fn take_prev_inf_pr(&mut self) -> Option<F> {
        self.prev_inf_pr.take()
    }

    /// Returns a reference to the current dual infeasibility
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # let state = state.inf_du(0.05);
    /// let inf_du = state.get_inf_du();  // Option<&F>
    /// # assert_eq!(inf_du.unwrap().to_ne_bytes(), 0.05f64.to_ne_bytes());
    /// ```
    pub fn get_inf_du(&self) -> Option<&F> {
        self.inf_du.as_ref()
    }

    /// Moves the current dual infeasibility out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.take_inf_du().is_none());
    /// # let mut state = state.inf_du(0.05);
    /// let inf_du = state.take_inf_du();  // Option<F>
    /// # assert!(state.take_inf_du().is_none());
    /// # assert!(state.inf_du.is_none());
    /// # assert_eq!(inf_du.unwrap().to_ne_bytes(), 0.05f64.to_ne_bytes());
    /// ```
    pub fn take_inf_du(&mut self) -> Option<F> {
        self.inf_du.take()
    }

    /// Returns a reference to the previous dual infeasibility
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.prev_inf_du.is_none());
    /// # state.prev_inf_du = Some(0.05);
    /// let prev_inf_du = state.get_prev_inf_du();  // Option<&F>
    /// # assert_eq!(prev_inf_du.unwrap().to_ne_bytes(), 0.05f64.to_ne_bytes());
    /// ```
    pub fn get_prev_inf_du(&self) -> Option<&F> {
        self.prev_inf_du.as_ref()
    }

    /// Moves the previous dual infeasibility out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.take_prev_inf_du().is_none());
    /// # state.prev_inf_du = Some(0.05);
    /// let prev_inf_du = state.take_prev_inf_du();  // Option<F>
    /// # assert!(state.take_prev_inf_du().is_none());
    /// # assert!(state.prev_inf_du.is_none());
    /// # assert_eq!(prev_inf_du.unwrap().to_ne_bytes(), 0.05f64.to_ne_bytes());
    /// ```
    pub fn take_prev_inf_du(&mut self) -> Option<F> {
        self.prev_inf_du.take()
    }

    /// Returns a reference to the current complementarity infeasibility
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # let state = state.compl_inf(0.01);
    /// let compl_inf = state.get_compl_inf();  // Option<&F>
    /// # assert_eq!(compl_inf.unwrap().to_ne_bytes(), 0.01f64.to_ne_bytes());
    /// ```
    pub fn get_compl_inf(&self) -> Option<&F> {
        self.compl_inf.as_ref()
    }

    /// Moves the current complementarity infeasibility out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.take_compl_inf().is_none());
    /// # let mut state = state.compl_inf(0.01);
    /// let compl_inf = state.take_compl_inf();  // Option<F>
    /// # assert!(state.take_compl_inf().is_none());
    /// # assert!(state.compl_inf.is_none());
    /// # assert_eq!(compl_inf.unwrap().to_ne_bytes(), 0.01f64.to_ne_bytes());
    /// ```
    pub fn take_compl_inf(&mut self) -> Option<F> {
        self.compl_inf.take()
    }

    /// Returns a reference to the previous complementarity infeasibility
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.prev_compl_inf.is_none());
    /// # state.prev_compl_inf = Some(0.01);
    /// let prev_compl_inf = state.get_prev_compl_inf();  // Option<&F>
    /// # assert_eq!(prev_compl_inf.unwrap().to_ne_bytes(), 0.01f64.to_ne_bytes());
    /// ```
    pub fn get_prev_compl_inf(&self) -> Option<&F> {
        self.prev_compl_inf.as_ref()
    }

    /// Moves the previous complementarity infeasibility out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.take_prev_compl_inf().is_none());
    /// # state.prev_compl_inf = Some(0.01);
    /// let prev_compl_inf = state.take_prev_compl_inf();  // Option<F>
    /// # assert!(state.take_prev_compl_inf().is_none());
    /// # assert!(state.prev_compl_inf.is_none());
    /// # assert_eq!(prev_compl_inf.unwrap().to_ne_bytes(), 0.01f64.to_ne_bytes());
    /// ```
    pub fn take_prev_compl_inf(&mut self) -> Option<F> {
        self.prev_compl_inf.take()
    }

    /// Overrides state of counting function executions (default: false)
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State};
    /// # let mut state: NonLinearProgramState<(), (), (), f64> = NonLinearProgramState::new();
    /// # assert!(!state.counting_enabled);
    /// let state = state.counting(true);
    /// # assert!(state.counting_enabled);
    /// ```
    #[must_use]
    pub fn counting(mut self, mode: bool) -> Self {
        self.counting_enabled = mode;
        self
    }
}

impl<P, G, LH, F, EqC, IneqC, S, EqCJ, IneqCJ, LambdaEq, LambdaIneq> State
    for NonLinearProgramState<P, G, LH, F, EqC, IneqC, S, EqCJ, IneqCJ, LambdaEq, LambdaIneq>
where
    P: Clone,
    S: Clone,
    F: ArgminFloat,
{
    /// Type of parameter vector
    type Param = P;
    /// Floating point precision
    type Float = F;

    /// Create a new NonLinearProgramState instance
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate web_time;
    /// # use web_time::Duration;
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat, TerminationStatus};
    /// let state: NonLinearProgramState<Vec<f64>, Vec<f64>, Vec<Vec<f64>>, f64> = NonLinearProgramState::new();
    /// # assert!(state.param.is_none());
    /// # assert!(state.prev_param.is_none());
    /// # assert!(state.slacks.is_none());
    /// # assert!(state.prev_slacks.is_none());
    /// # assert!(state.best_param.is_none());
    /// # assert!(state.prev_best_param.is_none());
    /// # assert!(state.best_slacks.is_none());
    /// # assert!(state.prev_best_slacks.is_none());
    /// # assert_eq!(state.cost.to_ne_bytes(), f64::INFINITY.to_ne_bytes());
    /// # assert_eq!(state.prev_cost.to_ne_bytes(), f64::INFINITY.to_ne_bytes());
    /// # assert_eq!(state.best_cost.to_ne_bytes(), f64::INFINITY.to_ne_bytes());
    /// # assert_eq!(state.prev_best_cost.to_ne_bytes(), f64::INFINITY.to_ne_bytes());
    /// # assert_eq!(state.target_cost.to_ne_bytes(), f64::NEG_INFINITY.to_ne_bytes());
    /// # assert!(state.grad.is_none());
    /// # assert!(state.prev_grad.is_none());
    /// # assert!(state.lagrangian_hessian.is_none());
    /// # assert!(state.prev_lagrangian_hessian.is_none());
    /// # assert!(state.equality_constraints.is_none());
    /// # assert!(state.prev_equality_constraints.is_none());
    /// # assert!(state.inequality_constraints.is_none());
    /// # assert!(state.prev_inequality_constraints.is_none());
    /// # assert!(state.equality_constraint_jacobian.is_none());
    /// # assert!(state.prev_equality_constraint_jacobian.is_none());
    /// # assert!(state.inequality_constraint_jacobian.is_none());
    /// # assert!(state.prev_inequality_constraint_jacobian.is_none());
    /// # assert!(state.mu.is_none());
    /// # assert!(state.prev_mu.is_none());
    /// # assert!(state.lambda_eq.is_none());
    /// # assert!(state.prev_lambda_eq.is_none());
    /// # assert!(state.lambda_ineq.is_none());
    /// # assert!(state.prev_lambda_ineq.is_none());
    /// # assert!(state.alpha_primal.is_none());
    /// # assert!(state.prev_alpha_primal.is_none());
    /// # assert!(state.alpha_dual.is_none());
    /// # assert!(state.prev_alpha_dual.is_none());
    /// # assert!(state.inf_pr.is_none());
    /// # assert!(state.prev_inf_pr.is_none());
    /// # assert!(state.inf_du.is_none());
    /// # assert!(state.prev_inf_du.is_none());
    /// # assert!(state.compl_inf.is_none());
    /// # assert!(state.prev_compl_inf.is_none());
    /// # assert_eq!(state.iter, 0);
    /// # assert_eq!(state.last_best_iter, 0);
    /// # assert_eq!(state.max_iters, u64::MAX);
    /// # assert_eq!(state.counts.len(), 0);
    /// # assert_eq!(state.time.unwrap(), Duration::ZERO);
    /// # assert_eq!(state.termination_status, TerminationStatus::NotTerminated);
    /// ```
    fn new() -> Self {
        NonLinearProgramState {
            param: None,
            prev_param: None,
            slacks: None,
            prev_slacks: None,
            best_param: None,
            prev_best_param: None,
            best_slacks: None,
            prev_best_slacks: None,
            cost: F::infinity(),
            prev_cost: F::infinity(),
            best_cost: F::infinity(),
            prev_best_cost: F::infinity(),
            target_cost: F::neg_infinity(),
            grad: None,
            prev_grad: None,
            lagrangian_hessian: None,
            prev_lagrangian_hessian: None,
            equality_constraints: None,
            prev_equality_constraints: None,
            inequality_constraints: None,
            prev_inequality_constraints: None,
            equality_constraint_jacobian: None,
            prev_equality_constraint_jacobian: None,
            inequality_constraint_jacobian: None,
            prev_inequality_constraint_jacobian: None,
            mu: None,
            prev_mu: None,
            lambda_eq: None,
            prev_lambda_eq: None,
            lambda_ineq: None,
            prev_lambda_ineq: None,
            alpha_primal: None,
            prev_alpha_primal: None,
            alpha_dual: None,
            prev_alpha_dual: None,
            inf_pr: None,
            prev_inf_pr: None,
            inf_du: None,
            prev_inf_du: None,
            compl_inf: None,
            prev_compl_inf: None,
            iter: 0,
            last_best_iter: 0,
            max_iters: u64::MAX,
            counts: HashMap::new(),
            counting_enabled: false,
            time: Some(Duration::ZERO),
            termination_status: TerminationStatus::NotTerminated,
        }
    }

    /// Checks if the current parameter vector and slack variables should become the new best
    /// feasible iterate. The caller is responsible for updating `best_cost` beforehand and for
    /// ensuring that `param` and `slacks` are feasible.
    ///
    /// This method performs only a simple consistency check and delegates all tolerance- and
    /// solver-specific acceptance logic to the solver.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// let mut state: NonLinearProgramState<Vec<f64>, (), (), f64, (), (), Vec<f64>> =
    ///     NonLinearProgramState::new();
    ///
    /// // Simulating a new feasible incumbent selected by the solver
    /// state.best_cost = 5.0;
    /// state.param = Some(vec![2.0f64]);
    /// state.slacks = Some(vec![3.0f64]);
    /// state.cost = 5.0;
    ///
    /// // Calling update
    /// state.update();
    ///
    /// // Check if update was successful
    /// assert_eq!(state.best_param.as_ref().unwrap()[0], 2.0f64);
    /// assert_eq!(state.best_slacks.as_ref().unwrap()[0], 3.0f64);
    /// assert_eq!(state.best_cost.to_ne_bytes(), 5.0f64.to_ne_bytes());
    /// assert!(state.is_best());
    /// ```
    fn update(&mut self) {
        let Some(param) = self.param.as_ref().cloned() else {
            return;
        };

        let Some(slacks) = self.slacks.as_ref().cloned() else {
            return;
        };

        let no_best_stored = self.best_param.is_none() || self.best_slacks.is_none();
        let best_cost_matches_current_cost = self.best_cost == self.cost;

        if no_best_stored || best_cost_matches_current_cost {
            std::mem::swap(&mut self.prev_best_param, &mut self.best_param);
            self.best_param = Some(param);

            std::mem::swap(&mut self.prev_best_slacks, &mut self.best_slacks);
            self.best_slacks = Some(slacks);

            std::mem::swap(&mut self.prev_best_cost, &mut self.best_cost);
            self.best_cost = self.cost;
            self.last_best_iter = self.iter;
        }
    }

    /// Returns a reference to the current parameter vector
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.param.is_none());
    /// # state.param = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let param = state.get_param();  // Option<&P>
    /// # assert_eq!(param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    fn get_param(&self) -> Option<&P> {
        self.param.as_ref()
    }

    /// Returns a reference to the current best parameter vector
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # assert!(state.best_param.is_none());
    /// # state.best_param = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.best_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.best_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let best_param = state.get_best_param();  // Option<&P>
    /// # assert_eq!(best_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(best_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    fn get_best_param(&self) -> Option<&P> {
        self.best_param.as_ref()
    }

    /// Sets the termination status to [`Terminated`](`TerminationStatus::Terminated`) with the given reason
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat, TerminationReason, TerminationStatus};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # assert_eq!(state.termination_status, TerminationStatus::NotTerminated);
    /// let state = state.terminate_with(TerminationReason::MaxItersReached);
    /// # assert_eq!(state.termination_status, TerminationStatus::Terminated(TerminationReason::MaxItersReached));
    /// ```
    fn terminate_with(mut self, reason: TerminationReason) -> Self {
        self.termination_status = TerminationStatus::Terminated(reason);
        self
    }

    /// Sets the time required so far.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate web_time;
    /// # use web_time::Duration;
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat, TerminationReason};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// let state = state.time(Some(Duration::from_nanos(12)));
    /// # assert_eq!(state.time.unwrap(), Duration::from_nanos(12));
    /// ```
    fn time(&mut self, time: Option<Duration>) -> &mut Self {
        self.time = time;
        self
    }

    /// Returns current cost function value.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # state.cost = 12.0;
    /// let cost = state.get_cost();
    /// # assert_eq!(cost.to_ne_bytes(), 12.0f64.to_ne_bytes());
    /// ```
    fn get_cost(&self) -> Self::Float {
        self.cost
    }

    /// Returns current best cost function value.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # state.best_cost = 12.0;
    /// let best_cost = state.get_best_cost();
    /// # assert_eq!(best_cost.to_ne_bytes(), 12.0f64.to_ne_bytes());
    /// ```
    fn get_best_cost(&self) -> Self::Float {
        self.best_cost
    }

    /// Returns target cost function value.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # state.target_cost = 12.0;
    /// let target_cost = state.get_target_cost();
    /// # assert_eq!(target_cost.to_ne_bytes(), 12.0f64.to_ne_bytes());
    /// ```
    fn get_target_cost(&self) -> Self::Float {
        self.target_cost
    }

    /// Returns current number of iterations.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # state.iter = 12;
    /// let iter = state.get_iter();
    /// # assert_eq!(iter, 12);
    /// ```
    fn get_iter(&self) -> u64 {
        self.iter
    }

    /// Returns iteration number of last best parameter vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # state.last_best_iter = 12;
    /// let last_best_iter = state.get_last_best_iter();
    /// # assert_eq!(last_best_iter, 12);
    /// ```
    fn get_last_best_iter(&self) -> u64 {
        self.last_best_iter
    }

    /// Returns the maximum number of iterations.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # state.max_iters = 12;
    /// let max_iters = state.get_max_iters();
    /// # assert_eq!(max_iters, 12);
    /// ```
    fn get_max_iters(&self) -> u64 {
        self.max_iters
    }

    /// Returns the termination status.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat, TerminationStatus};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// let termination_status = state.get_termination_status();
    /// # assert_eq!(*termination_status, TerminationStatus::NotTerminated);
    /// ```
    fn get_termination_status(&self) -> &TerminationStatus {
        &self.termination_status
    }

    /// Returns the termination reason if terminated, otherwise None.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat, TerminationReason};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// let termination_reason = state.get_termination_reason();
    /// # assert_eq!(termination_reason, None);
    /// ```
    fn get_termination_reason(&self) -> Option<&TerminationReason> {
        match &self.termination_status {
            TerminationStatus::Terminated(reason) => Some(reason),
            TerminationStatus::NotTerminated => None,
        }
    }

    /// Returns the time elapsed since the start of the optimization.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate web_time;
    /// # use web_time::Duration;
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// let time = state.get_time();
    /// # assert_eq!(time.unwrap(), Duration::ZERO);
    /// ```
    fn get_time(&self) -> Option<Duration> {
        self.time
    }

    /// Increments the number of iterations by one
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # assert_eq!(state.iter, 0);
    /// state.increment_iter();
    /// # assert_eq!(state.iter, 1);
    /// ```
    fn increment_iter(&mut self) {
        self.iter += 1;
    }

    /// Set all function evaluation counts to the evaluation counts of another `Problem`.
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// # use argmin::core::{Problem, NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new().counting(true);
    /// # assert_eq!(state.counts, HashMap::new());
    /// # state.counts.insert("test2".to_string(), 10u64);
    /// #
    /// # #[derive(Eq, PartialEq, Debug)]
    /// # struct UserDefinedProblem {};
    /// #
    /// # let mut problem = Problem::new(UserDefinedProblem {});
    /// # problem.counts.insert("test1", 10u64);
    /// # problem.counts.insert("test2", 2);
    /// state.func_counts(&problem);
    /// # let mut hm = HashMap::new();
    /// # hm.insert("test1".to_string(), 10u64);
    /// # hm.insert("test2".to_string(), 2u64);
    /// # assert_eq!(state.counts, hm);
    /// ```
    fn func_counts<O>(&mut self, problem: &Problem<O>) {
        if self.counting_enabled {
            for (k, &v) in problem.counts.iter() {
                let count = self.counts.entry(k.to_string()).or_insert(0);
                *count = v
            }
        }
    }

    /// Returns function evaluation counts
    ///
    /// # Example
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # assert_eq!(state.counts, HashMap::new());
    /// # state.counts.insert("test2".to_string(), 10u64);
    /// let counts = state.get_func_counts();
    /// # let mut hm = HashMap::new();
    /// # hm.insert("test2".to_string(), 10u64);
    /// # assert_eq!(*counts, hm);
    /// ```
    fn get_func_counts(&self) -> &HashMap<String, u64> {
        &self.counts
    }

    /// Returns whether the current parameter vector is also the best parameter vector found so
    /// far.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{NonLinearProgramState, State, ArgminFloat};
    /// # let mut state: NonLinearProgramState<Vec<f64>, (), (), f64> = NonLinearProgramState::new();
    /// # state.last_best_iter = 12;
    /// # state.iter = 12;
    /// let is_best = state.is_best();
    /// # assert!(is_best);
    /// # state.last_best_iter = 12;
    /// # state.iter = 21;
    /// # let is_best = state.is_best();
    /// # assert!(!is_best);
    /// ```
    fn is_best(&self) -> bool {
        self.last_best_iter == self.iter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::type_complexity)]
    fn test_nonlinearprogramstate() {
        let param = vec![1.0f64, 2.0];
        let cost: f64 = 42.0;

        let mut state: NonLinearProgramState<
            Vec<f64>,
            Vec<f64>,
            Vec<Vec<f64>>,
            f64,
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
            Vec<Vec<f64>>,
            Vec<Vec<f64>>,
            Vec<f64>,
            Vec<f64>,
        > = NonLinearProgramState::new();

        assert!(state.get_param().is_none());
        assert!(state.get_prev_param().is_none());
        assert!(state.get_slacks().is_none());
        assert!(state.get_prev_slacks().is_none());
        assert!(state.get_best_param().is_none());
        assert!(state.get_prev_best_param().is_none());
        assert!(state.get_best_slacks().is_none());
        assert!(state.get_prev_best_slacks().is_none());

        state = state.param(param.clone());

        assert_eq!(*state.get_param().unwrap(), param);
        assert!(state.get_prev_param().is_none());
        assert!(state.get_best_param().is_none());
        assert!(state.get_prev_best_param().is_none());

        assert!(state.get_cost().is_infinite());
        assert!(state.get_cost().is_sign_positive());

        assert!(state.get_prev_cost().is_infinite());
        assert!(state.get_prev_cost().is_sign_positive());

        assert!(state.get_best_cost().is_infinite());
        assert!(state.get_best_cost().is_sign_positive());

        assert!(state.get_prev_best_cost().is_infinite());
        assert!(state.get_prev_best_cost().is_sign_positive());

        assert!(state.get_target_cost().is_infinite());
        assert!(state.get_target_cost().is_sign_negative());

        assert!(state.get_gradient().is_none());
        assert!(state.get_prev_gradient().is_none());
        assert!(state.get_lagrangian_hessian().is_none());
        assert!(state.get_prev_lagrangian_hessian().is_none());

        assert!(state.get_equality_constraints().is_none());
        assert!(state.get_prev_equality_constraints().is_none());
        assert!(state.get_inequality_constraints().is_none());
        assert!(state.get_prev_inequality_constraints().is_none());
        assert!(state.get_equality_constraint_jacobian().is_none());
        assert!(state.get_prev_equality_constraint_jacobian().is_none());
        assert!(state.get_inequality_constraint_jacobian().is_none());
        assert!(state.get_prev_inequality_constraint_jacobian().is_none());

        assert!(state.get_mu().is_none());
        assert!(state.get_prev_mu().is_none());
        assert!(state.get_lambda_eq().is_none());
        assert!(state.get_prev_lambda_eq().is_none());
        assert!(state.get_lambda_ineq().is_none());
        assert!(state.get_prev_lambda_ineq().is_none());
        assert!(state.get_alpha_primal().is_none());
        assert!(state.get_prev_alpha_primal().is_none());
        assert!(state.get_alpha_dual().is_none());
        assert!(state.get_prev_alpha_dual().is_none());
        assert!(state.get_inf_pr().is_none());
        assert!(state.get_prev_inf_pr().is_none());
        assert!(state.get_inf_du().is_none());
        assert!(state.get_prev_inf_du().is_none());
        assert!(state.get_compl_inf().is_none());
        assert!(state.get_prev_compl_inf().is_none());

        assert_eq!(state.get_iter(), 0);

        assert!(state.is_best());

        assert_eq!(state.get_max_iters(), u64::MAX);
        let func_counts = state.get_func_counts().clone();
        assert!(!func_counts.contains_key("cost_count"));
        assert!(!func_counts.contains_key("operator_count"));
        assert!(!func_counts.contains_key("gradient_count"));
        assert!(!func_counts.contains_key("lagrangian_hessian_count"));
        assert!(!func_counts.contains_key("equality_constraint_count"));
        assert!(!func_counts.contains_key("inequality_constraint_count"));
        assert!(!func_counts.contains_key("equality_constraint_jacobian_count"));
        assert!(!func_counts.contains_key("inequality_constraint_jacobian_count"));
        assert!(!func_counts.contains_key("modify_count"));

        state = state.max_iters(42);

        assert_eq!(state.get_max_iters(), 42);

        let mut state = state.cost(cost);

        assert_eq!(state.get_cost().to_ne_bytes(), cost.to_ne_bytes());
        assert!(state.get_prev_cost().is_infinite());
        assert!(state.get_prev_cost().is_sign_positive());

        let new_param = vec![2.0, 1.0];

        state = state.param(new_param.clone());

        assert_eq!(*state.get_param().unwrap(), new_param);
        assert_eq!(*state.get_prev_param().unwrap(), param);

        let slacks = vec![5.0f64, 6.0];

        let state = state.slacks(slacks.clone());
        assert_eq!(*state.get_slacks().unwrap(), slacks);
        assert!(state.get_prev_slacks().is_none());

        let new_slacks = vec![6.0f64, 5.0];

        let state = state.slacks(new_slacks.clone());

        assert_eq!(*state.get_slacks().unwrap(), new_slacks);
        assert_eq!(*state.get_prev_slacks().unwrap(), slacks);

        let new_cost: f64 = 21.0;

        let mut state = state.cost(new_cost);

        assert_eq!(state.get_cost().to_ne_bytes(), new_cost.to_ne_bytes());
        assert_eq!(state.get_prev_cost().to_ne_bytes(), cost.to_ne_bytes());

        state.increment_iter();

        assert_eq!(state.get_iter(), 1);

        assert!(!state.is_best());

        state.last_best_iter = state.iter;

        assert!(state.is_best());

        let grad = vec![1.0, 2.0];

        let state = state.gradient(grad.clone());
        assert_eq!(*state.get_gradient().unwrap(), grad);
        assert!(state.get_prev_gradient().is_none());

        let new_grad = vec![2.0, 1.0];

        let state = state.gradient(new_grad.clone());

        assert_eq!(*state.get_gradient().unwrap(), new_grad);
        assert_eq!(*state.get_prev_gradient().unwrap(), grad);

        let lagrangian_hessian = vec![vec![1.0, 2.0], vec![2.0, 1.0]];

        let state = state.lagrangian_hessian(lagrangian_hessian.clone());
        assert_eq!(*state.get_lagrangian_hessian().unwrap(), lagrangian_hessian);
        assert!(state.get_prev_lagrangian_hessian().is_none());

        let new_lagrangian_hessian = vec![vec![2.0, 1.0], vec![1.0, 2.0]];

        let state = state.lagrangian_hessian(new_lagrangian_hessian.clone());

        assert_eq!(
            *state.get_lagrangian_hessian().unwrap(),
            new_lagrangian_hessian
        );
        assert_eq!(
            *state.get_prev_lagrangian_hessian().unwrap(),
            lagrangian_hessian
        );

        let equality_constraints = vec![1.0f64, 2.0];

        let state = state.equality_constraints(equality_constraints.clone());
        assert_eq!(
            *state.get_equality_constraints().unwrap(),
            equality_constraints
        );
        assert!(state.get_prev_equality_constraints().is_none());

        let new_equality_constraints = vec![2.0f64, 1.0];

        let state = state.equality_constraints(new_equality_constraints.clone());

        assert_eq!(
            *state.get_equality_constraints().unwrap(),
            new_equality_constraints
        );
        assert_eq!(
            *state.get_prev_equality_constraints().unwrap(),
            equality_constraints
        );

        let inequality_constraints = vec![3.0f64, 4.0];

        let state = state.inequality_constraints(inequality_constraints.clone());
        assert_eq!(
            *state.get_inequality_constraints().unwrap(),
            inequality_constraints
        );
        assert!(state.get_prev_inequality_constraints().is_none());

        let new_inequality_constraints = vec![4.0f64, 3.0];

        let state = state.inequality_constraints(new_inequality_constraints.clone());

        assert_eq!(
            *state.get_inequality_constraints().unwrap(),
            new_inequality_constraints
        );
        assert_eq!(
            *state.get_prev_inequality_constraints().unwrap(),
            inequality_constraints
        );

        let equality_constraint_jacobian = vec![vec![1.0f64, 2.0], vec![3.0, 4.0]];

        let state = state.equality_constraint_jacobian(equality_constraint_jacobian.clone());
        assert_eq!(
            *state.get_equality_constraint_jacobian().unwrap(),
            equality_constraint_jacobian
        );
        assert!(state.get_prev_equality_constraint_jacobian().is_none());

        let new_equality_constraint_jacobian = vec![vec![2.0f64, 1.0], vec![4.0, 3.0]];

        let state = state.equality_constraint_jacobian(new_equality_constraint_jacobian.clone());

        assert_eq!(
            *state.get_equality_constraint_jacobian().unwrap(),
            new_equality_constraint_jacobian
        );
        assert_eq!(
            *state.get_prev_equality_constraint_jacobian().unwrap(),
            equality_constraint_jacobian
        );

        let inequality_constraint_jacobian = vec![vec![5.0f64, 6.0], vec![7.0, 8.0]];

        let state = state.inequality_constraint_jacobian(inequality_constraint_jacobian.clone());
        assert_eq!(
            *state.get_inequality_constraint_jacobian().unwrap(),
            inequality_constraint_jacobian
        );
        assert!(state.get_prev_inequality_constraint_jacobian().is_none());

        let new_inequality_constraint_jacobian = vec![vec![6.0f64, 5.0], vec![8.0, 7.0]];

        let state =
            state.inequality_constraint_jacobian(new_inequality_constraint_jacobian.clone());

        assert_eq!(
            *state.get_inequality_constraint_jacobian().unwrap(),
            new_inequality_constraint_jacobian
        );
        assert_eq!(
            *state.get_prev_inequality_constraint_jacobian().unwrap(),
            inequality_constraint_jacobian
        );

        let mu = 1.0f64;

        let state = state.mu(mu);
        assert_eq!(state.get_mu().unwrap().to_ne_bytes(), 1.0f64.to_ne_bytes());
        assert!(state.get_prev_mu().is_none());

        let new_mu = 0.5f64;

        let state = state.mu(new_mu);
        assert_eq!(state.get_mu().unwrap().to_ne_bytes(), 0.5f64.to_ne_bytes());
        assert_eq!(
            state.get_prev_mu().unwrap().to_ne_bytes(),
            1.0f64.to_ne_bytes()
        );

        let lambda_eq = vec![1.0f64, 2.0];

        let state = state.lambda_eq(lambda_eq.clone());
        assert_eq!(*state.get_lambda_eq().unwrap(), lambda_eq);
        assert!(state.get_prev_lambda_eq().is_none());

        let new_lambda_eq = vec![2.0f64, 1.0];

        let state = state.lambda_eq(new_lambda_eq.clone());
        assert_eq!(*state.get_lambda_eq().unwrap(), new_lambda_eq);
        assert_eq!(*state.get_prev_lambda_eq().unwrap(), lambda_eq);

        let lambda_ineq = vec![3.0f64, 4.0];

        let state = state.lambda_ineq(lambda_ineq.clone());
        assert_eq!(*state.get_lambda_ineq().unwrap(), lambda_ineq);
        assert!(state.get_prev_lambda_ineq().is_none());

        let new_lambda_ineq = vec![4.0f64, 3.0];

        let state = state.lambda_ineq(new_lambda_ineq.clone());
        assert_eq!(*state.get_lambda_ineq().unwrap(), new_lambda_ineq);
        assert_eq!(*state.get_prev_lambda_ineq().unwrap(), lambda_ineq);

        let alpha_primal = 1.0f64;

        let state = state.alpha_primal(alpha_primal);
        assert_eq!(
            state.get_alpha_primal().unwrap().to_ne_bytes(),
            alpha_primal.to_ne_bytes()
        );
        assert!(state.get_prev_alpha_primal().is_none());

        let new_alpha_primal = 0.75f64;

        let state = state.alpha_primal(new_alpha_primal);
        assert_eq!(
            state.get_alpha_primal().unwrap().to_ne_bytes(),
            new_alpha_primal.to_ne_bytes()
        );
        assert_eq!(
            state.get_prev_alpha_primal().unwrap().to_ne_bytes(),
            alpha_primal.to_ne_bytes()
        );

        let alpha_dual = 1.0f64;

        let state = state.alpha_dual(alpha_dual);
        assert_eq!(
            state.get_alpha_dual().unwrap().to_ne_bytes(),
            alpha_dual.to_ne_bytes()
        );
        assert!(state.get_prev_alpha_dual().is_none());

        let new_alpha_dual = 0.8f64;

        let state = state.alpha_dual(new_alpha_dual);
        assert_eq!(
            state.get_alpha_dual().unwrap().to_ne_bytes(),
            new_alpha_dual.to_ne_bytes()
        );
        assert_eq!(
            state.get_prev_alpha_dual().unwrap().to_ne_bytes(),
            alpha_dual.to_ne_bytes()
        );

        let inf_pr = 1.0f64;

        let state = state.inf_pr(inf_pr);
        assert_eq!(
            state.get_inf_pr().unwrap().to_ne_bytes(),
            inf_pr.to_ne_bytes()
        );
        assert!(state.get_prev_inf_pr().is_none());

        let new_inf_pr = 0.1f64;

        let state = state.inf_pr(new_inf_pr);
        assert_eq!(
            state.get_inf_pr().unwrap().to_ne_bytes(),
            new_inf_pr.to_ne_bytes()
        );
        assert_eq!(
            state.get_prev_inf_pr().unwrap().to_ne_bytes(),
            inf_pr.to_ne_bytes()
        );

        let inf_du = 1.0f64;

        let state = state.inf_du(inf_du);
        assert_eq!(
            state.get_inf_du().unwrap().to_ne_bytes(),
            inf_du.to_ne_bytes()
        );
        assert!(state.get_prev_inf_du().is_none());

        let new_inf_du = 0.05f64;

        let state = state.inf_du(new_inf_du);
        assert_eq!(
            state.get_inf_du().unwrap().to_ne_bytes(),
            new_inf_du.to_ne_bytes()
        );
        assert_eq!(
            state.get_prev_inf_du().unwrap().to_ne_bytes(),
            inf_du.to_ne_bytes()
        );

        let compl_inf = 1.0f64;

        let state = state.compl_inf(compl_inf);
        assert_eq!(
            state.get_compl_inf().unwrap().to_ne_bytes(),
            compl_inf.to_ne_bytes()
        );
        assert!(state.get_prev_compl_inf().is_none());

        let new_compl_inf = 0.01f64;

        let mut state = state.compl_inf(new_compl_inf);
        assert_eq!(
            state.get_compl_inf().unwrap().to_ne_bytes(),
            new_compl_inf.to_ne_bytes()
        );
        assert_eq!(
            state.get_prev_compl_inf().unwrap().to_ne_bytes(),
            compl_inf.to_ne_bytes()
        );

        state.increment_iter();

        assert_eq!(state.get_iter(), 2);
        assert_eq!(state.get_last_best_iter(), 1);
        assert!(!state.is_best());

        // check again!
        assert_eq!(state.get_iter(), 2);
        assert_eq!(state.get_last_best_iter(), 1);
        assert_eq!(state.get_max_iters(), 42);

        assert!(!state.is_best());

        assert_eq!(state.get_cost().to_ne_bytes(), new_cost.to_ne_bytes());
        assert_eq!(state.get_prev_cost().to_ne_bytes(), cost.to_ne_bytes());
        assert_eq!(state.get_prev_cost().to_ne_bytes(), cost.to_ne_bytes());

        assert_eq!(*state.get_param().unwrap(), new_param);
        assert_eq!(*state.get_prev_param().unwrap(), param);

        assert_eq!(*state.get_gradient().unwrap(), new_grad);
        assert_eq!(*state.get_prev_gradient().unwrap(), grad);

        assert_eq!(
            *state.get_lagrangian_hessian().unwrap(),
            new_lagrangian_hessian
        );
        assert_eq!(
            *state.get_prev_lagrangian_hessian().unwrap(),
            lagrangian_hessian
        );

        assert_eq!(
            *state.get_equality_constraints().unwrap(),
            new_equality_constraints
        );
        assert_eq!(
            *state.get_prev_equality_constraints().unwrap(),
            equality_constraints
        );

        assert_eq!(
            *state.get_inequality_constraints().unwrap(),
            new_inequality_constraints
        );
        assert_eq!(
            *state.get_prev_inequality_constraints().unwrap(),
            inequality_constraints
        );

        assert_eq!(*state.get_slacks().unwrap(), new_slacks);
        assert_eq!(*state.get_prev_slacks().unwrap(), slacks);

        assert_eq!(
            *state.get_equality_constraint_jacobian().unwrap(),
            new_equality_constraint_jacobian
        );
        assert_eq!(
            *state.get_prev_equality_constraint_jacobian().unwrap(),
            equality_constraint_jacobian
        );

        assert_eq!(
            *state.get_inequality_constraint_jacobian().unwrap(),
            new_inequality_constraint_jacobian
        );
        assert_eq!(
            *state.get_prev_inequality_constraint_jacobian().unwrap(),
            inequality_constraint_jacobian
        );

        assert_eq!(state.get_mu().unwrap().to_ne_bytes(), new_mu.to_ne_bytes());
        assert_eq!(state.get_prev_mu().unwrap().to_ne_bytes(), mu.to_ne_bytes());

        assert_eq!(*state.get_lambda_eq().unwrap(), new_lambda_eq);
        assert_eq!(*state.get_prev_lambda_eq().unwrap(), lambda_eq);

        assert_eq!(*state.get_lambda_ineq().unwrap(), new_lambda_ineq);
        assert_eq!(*state.get_prev_lambda_ineq().unwrap(), lambda_ineq);

        assert_eq!(
            state.get_alpha_primal().unwrap().to_ne_bytes(),
            new_alpha_primal.to_ne_bytes()
        );
        assert_eq!(
            state.get_prev_alpha_primal().unwrap().to_ne_bytes(),
            alpha_primal.to_ne_bytes()
        );

        assert_eq!(
            state.get_alpha_dual().unwrap().to_ne_bytes(),
            new_alpha_dual.to_ne_bytes()
        );
        assert_eq!(
            state.get_prev_alpha_dual().unwrap().to_ne_bytes(),
            alpha_dual.to_ne_bytes()
        );

        assert_eq!(
            state.get_inf_pr().unwrap().to_ne_bytes(),
            new_inf_pr.to_ne_bytes()
        );
        assert_eq!(
            state.get_prev_inf_pr().unwrap().to_ne_bytes(),
            inf_pr.to_ne_bytes()
        );

        assert_eq!(
            state.get_inf_du().unwrap().to_ne_bytes(),
            new_inf_du.to_ne_bytes()
        );
        assert_eq!(
            state.get_prev_inf_du().unwrap().to_ne_bytes(),
            inf_du.to_ne_bytes()
        );

        assert_eq!(
            state.get_compl_inf().unwrap().to_ne_bytes(),
            new_compl_inf.to_ne_bytes()
        );
        assert_eq!(
            state.get_prev_compl_inf().unwrap().to_ne_bytes(),
            compl_inf.to_ne_bytes()
        );

        assert_eq!(state.take_slacks().unwrap(), new_slacks);
        assert_eq!(state.take_prev_slacks().unwrap(), slacks);

        assert_eq!(state.take_gradient().unwrap(), new_grad);
        assert_eq!(state.take_prev_gradient().unwrap(), grad);

        assert_eq!(
            state.take_lagrangian_hessian().unwrap(),
            new_lagrangian_hessian
        );
        assert_eq!(
            state.take_prev_lagrangian_hessian().unwrap(),
            lagrangian_hessian
        );

        assert_eq!(
            state.take_equality_constraints().unwrap(),
            new_equality_constraints
        );
        assert_eq!(
            state.take_prev_equality_constraints().unwrap(),
            equality_constraints
        );

        assert_eq!(
            state.take_inequality_constraints().unwrap(),
            new_inequality_constraints
        );
        assert_eq!(
            state.take_prev_inequality_constraints().unwrap(),
            inequality_constraints
        );

        assert_eq!(
            state.take_equality_constraint_jacobian().unwrap(),
            new_equality_constraint_jacobian
        );
        assert_eq!(
            state.take_prev_equality_constraint_jacobian().unwrap(),
            equality_constraint_jacobian
        );

        assert_eq!(
            state.take_inequality_constraint_jacobian().unwrap(),
            new_inequality_constraint_jacobian
        );
        assert_eq!(
            state.take_prev_inequality_constraint_jacobian().unwrap(),
            inequality_constraint_jacobian
        );

        assert_eq!(state.take_mu().unwrap().to_ne_bytes(), new_mu.to_ne_bytes());
        assert_eq!(
            state.take_prev_mu().unwrap().to_ne_bytes(),
            mu.to_ne_bytes()
        );

        assert_eq!(state.take_lambda_eq().unwrap(), new_lambda_eq);
        assert_eq!(state.take_prev_lambda_eq().unwrap(), lambda_eq);

        assert_eq!(state.take_lambda_ineq().unwrap(), new_lambda_ineq);
        assert_eq!(state.take_prev_lambda_ineq().unwrap(), lambda_ineq);

        assert_eq!(
            state.take_alpha_primal().unwrap().to_ne_bytes(),
            new_alpha_primal.to_ne_bytes()
        );
        assert_eq!(
            state.take_prev_alpha_primal().unwrap().to_ne_bytes(),
            alpha_primal.to_ne_bytes()
        );

        assert_eq!(
            state.take_alpha_dual().unwrap().to_ne_bytes(),
            new_alpha_dual.to_ne_bytes()
        );
        assert_eq!(
            state.take_prev_alpha_dual().unwrap().to_ne_bytes(),
            alpha_dual.to_ne_bytes()
        );

        assert_eq!(
            state.take_inf_pr().unwrap().to_ne_bytes(),
            new_inf_pr.to_ne_bytes()
        );
        assert_eq!(
            state.take_prev_inf_pr().unwrap().to_ne_bytes(),
            inf_pr.to_ne_bytes()
        );

        assert_eq!(
            state.take_inf_du().unwrap().to_ne_bytes(),
            new_inf_du.to_ne_bytes()
        );
        assert_eq!(
            state.take_prev_inf_du().unwrap().to_ne_bytes(),
            inf_du.to_ne_bytes()
        );

        assert_eq!(
            state.take_compl_inf().unwrap().to_ne_bytes(),
            new_compl_inf.to_ne_bytes()
        );
        assert_eq!(
            state.take_prev_compl_inf().unwrap().to_ne_bytes(),
            compl_inf.to_ne_bytes()
        );

        let func_counts = state.get_func_counts().clone();
        assert!(!func_counts.contains_key("cost_count"));
        assert!(!func_counts.contains_key("operator_count"));
        assert!(!func_counts.contains_key("gradient_count"));
        assert!(!func_counts.contains_key("lagrangian_hessian_count"));
        assert!(!func_counts.contains_key("equality_constraint_count"));
        assert!(!func_counts.contains_key("inequality_constraint_count"));
        assert!(!func_counts.contains_key("equality_constraint_jacobian_count"));
        assert!(!func_counts.contains_key("inequality_constraint_jacobian_count"));
        assert!(!func_counts.contains_key("modify_count"));
    }
}

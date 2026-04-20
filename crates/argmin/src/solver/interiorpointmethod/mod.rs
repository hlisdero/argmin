// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Interior Point Method
//!
//! * [`InteriorPointMethod`]
//!
//! This solver is currently experimental and requires the
//! `experimental_ipm` cargo feature.
//!
//! The current implementation uses a nalgebra-backed dense KKT system.
//!
//! # Reference
//!
//! Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

mod interiorpointmethod;
mod iteration;
mod linesearch;
mod nalgebra_backend;

pub use self::interiorpointmethod::InteriorPointMethod;

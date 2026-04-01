// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::{Allocator, ArgminLuSolve, Error};
use nalgebra::{
    base::{dimension::Dim, storage::Storage},
    ComplexField, DefaultAllocator, DimMin, OMatrix, SquareMatrix, LU,
};
use std::fmt;

#[derive(Debug, thiserror::Error, PartialEq)]
struct LuSolveError;

impl fmt::Display for LuSolveError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Non-invertible matrix")
    }
}

impl<N, D, C, S> ArgminLuSolve<OMatrix<N, D, C>> for SquareMatrix<N, D, S>
where
    N: ComplexField,
    D: Dim + DimMin<D, Output = D>,
    C: Dim,
    S: Storage<N, D, D>,
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D, C> + Allocator<N, D>,
{
    #[inline]
    fn lu_solve(&self, rhs: &OMatrix<N, D, C>) -> Result<OMatrix<N, D, C>, Error> {
        let lu = LU::new(self.clone_owned());
        lu.solve(rhs).ok_or_else(|| LuSolveError {}.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::Matrix2;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_lu_solve_ $t>]() {
                    let a = Matrix2::new(
                        2 as $t, 5 as $t,
                        1 as $t, 3 as $t,
                    );
                    // solve A * x = b
                    let b = nalgebra::Vector2::new(1 as $t, 2 as $t);
                    // compute expected using explicit inverse (for test only)
                    let inv = Matrix2::new(
                        3 as $t, -5 as $t,
                        -1 as $t, 2 as $t,
                    );
                    let target = inv * b;
                    let res = <Matrix2<$t> as ArgminLuSolve<nalgebra::Vector2<$t>>>::lu_solve(&a, &b).unwrap();
                    for i in 0..2 {
                        assert_relative_eq!(res[i], target[i], epsilon = $t::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_lu_solve_error_ $t>]() {
                    let a = Matrix2::new(
                        2 as $t, 5 as $t,
                        4 as $t, 10 as $t,
                    );
                    let b = nalgebra::Vector2::new(1 as $t, 2 as $t);
                    let err = <Matrix2<$t> as ArgminLuSolve<nalgebra::Vector2<$t>>>::lu_solve(&a, &b)
                        .unwrap_err()
                        .downcast::<LuSolveError>()
                        .unwrap();
                    assert_eq!(err, LuSolveError {});
                    assert_eq!(format!("{}", err), "Non-invertible matrix");
                    assert_eq!(format!("{:?}", err), "LuSolveError");
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}

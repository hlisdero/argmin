use crate::{Allocator, ArgminCholeskySolve, Error};
use nalgebra::{
    base::{dimension::Dim, storage::Storage},
    Cholesky, ComplexField, DefaultAllocator, DimMin, OMatrix, SquareMatrix,
};
use std::fmt;

#[derive(Debug, thiserror::Error, PartialEq)]
struct CholeskySolveError;

impl fmt::Display for CholeskySolveError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Matrix is not symmetric positive definite")
    }
}

impl<N, D, C, S> ArgminCholeskySolve<OMatrix<N, D, C>> for SquareMatrix<N, D, S>
where
    N: ComplexField,
    D: Dim + DimMin<D, Output = D>,
    C: Dim,
    S: Storage<N, D, D>,
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D, C>,
{
    #[inline]
    fn cholesky_solve(&self, rhs: &OMatrix<N, D, C>) -> Result<OMatrix<N, D, C>, Error> {
        let cholesky =
            Cholesky::new(self.clone_owned()).ok_or_else(|| Error::from(CholeskySolveError {}))?;
        Ok(cholesky.solve(rhs))
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
                fn [<test_cholesky_solve_ $t>]() {
                    let a = Matrix2::new(
                        4 as $t, 1 as $t,
                        1 as $t, 3 as $t,
                    );
                    let b = nalgebra::Vector2::new(1 as $t, 2 as $t);

                    // A^{-1} = 1/11 * [ 3 -1 ]
                    //                 [ -1 4 ]
                    let target = nalgebra::Vector2::new(
                        (1 as $t) / (11 as $t),
                        (7 as $t) / (11 as $t),
                    );

                    let res = <Matrix2<$t> as ArgminCholeskySolve<nalgebra::Vector2<$t>>>::cholesky_solve(&a, &b).unwrap();

                    for i in 0..2 {
                        assert_relative_eq!(res[i], target[i], epsilon = 10.0 as $t * $t::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_cholesky_solve_error_ $t>]() {
                    let a = Matrix2::new(
                        1 as $t, 2 as $t,
                        2 as $t, 1 as $t,
                    );
                    let b = nalgebra::Vector2::new(1 as $t, 2 as $t);

                    let err = <Matrix2<$t> as ArgminCholeskySolve<nalgebra::Vector2<$t>>>::cholesky_solve(&a, &b)
                        .unwrap_err()
                        .downcast::<CholeskySolveError>()
                        .unwrap();

                    assert_eq!(err, CholeskySolveError {});
                    assert_eq!(format!("{}", err), "Matrix is not symmetric positive definite");
                    assert_eq!(format!("{:?}", err), "CholeskySolveError");
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}

use rand::prelude::{Distribution, thread_rng};
use rand_distr::Normal;
use rayon::prelude::*;
use crate::common::utils;

use super::activation::*;

const MEAN: f64 = 0.0;
const DEVIATION: f64 = 5.0;


#[derive(Clone, PartialEq)]
pub(crate) struct Layer {
    nrow: usize,
    ncol: usize,
    pub(super) matrix: Vec<Vec<f64>>,
    pub(super) bias: Vec<f64>,
    pub(super) activation: Activation,
}


impl Layer {
    #[inline(always)]
    pub(crate) fn new(nrow: usize, ncol: usize, activation: Activation)
        -> Self
    {
        let mut rng = thread_rng();
        let dist = Normal::<f64>::new(MEAN, DEVIATION).unwrap();
        let matrix = (0..nrow).map(|_|
                dist.sample_iter(&mut rng)
                    .take(ncol)
                    .collect::<Vec<_>>()
            )
            .collect::<Vec<_>>();
        let bias = dist.sample_iter(&mut rng).take(nrow).collect();

        Self { nrow, ncol, matrix, bias, activation, }
    }


    #[inline(always)]
    pub(crate) fn output_dim(&self) -> usize {
        self.nrow
    }


    #[inline(always)]
    pub(crate) fn shape(&self) -> (usize, usize) {
        (self.nrow, self.ncol)
    }


    #[inline(always)]
    pub(crate) fn affine<T: AsRef<[f64]>>(&self, x: T) -> Vec<f64> {
        let x = x.as_ref();
        assert_eq!(self.ncol, x.len());

        self.matrix.par_iter()
            .zip(&self.bias)
            .map(|(w, b)| utils::inner_product(w, x) + b)
            .collect::<Vec<f64>>()
    }


    #[inline(always)]
    pub(crate) fn nonlinear<T: AsRef<[f64]>>(&self, u: T) -> Vec<f64> {
        let u = u.as_ref();
        self.activation.eval(u)
    }


    #[inline(always)]
    pub(crate) fn forward<T: AsRef<[f64]>>(&self, x: T) -> Vec<f64> {
        let u = self.affine(x);
        self.nonlinear(u)
    }


    /// Update `self.matrix` and `self.bias`
    /// and returns the next `delta`.
    /// Note:
    /// - The `i`-th row of `batch_u` corresponds to `W z[i] + b`.
    #[inline(always)]
    pub(crate) fn backward(
        &mut self,
        rate: f64,
        dw: Vec<Vec<f64>>,
        db: Vec<f64>,
    )
    {
        assert_eq!(self.matrix.len(), dw.len());
        assert_eq!(self.matrix[0].len(), dw[0].len());
        assert_eq!(self.bias.len(), db.len());
        self.matrix.iter_mut()
            .zip(dw)
            .for_each(|(row, drow)| {
                row.iter_mut()
                    .zip(drow)
                    .for_each(|(r, dr)| { *r -= rate * dr; });
            });

        self.bias.iter_mut()
            .zip(db)
            .for_each(|(b, db)| { *b -= rate * db; });
    }
}

/// Computes `A^T B` for matrices `A` and `B`.
#[inline(always)]
pub(super) fn matrix_inner_product(m1: &[Vec<f64>], m2: &[Vec<f64>])
    -> Vec<Vec<f64>>
{
    // Check the shape condition.
    assert_eq!(m1.len(), m2.len());

    let nrow = m1[0].len();
    let ncol = m2[0].len();
    let nmid = m1.len();

    let mut ans = vec![vec![0.0; ncol]; nrow];
    for i in 0..nrow {
        for j in 0..ncol {
            for k in 0..nmid {
                ans[i][j] += m1[k][i] * m2[k][j];
            }
        }
    }
    ans
}



/// Computes `A B^T` for matrices `A` and `B`.
#[inline(always)]
pub(super) fn matrix_product(m1: &[Vec<f64>], m2: &[Vec<f64>])
    -> Vec<Vec<f64>>
{
    // Check the shape condition.
    assert_eq!(m1[0].len(), m2.len());

    let nrow = m1.len();
    let ncol = m2[0].len();
    let nmid = m1[0].len();

    let mut ans = vec![vec![0.0; ncol]; nrow];
    for i in 0..nrow {
        for j in 0..ncol {
            for k in 0..nmid {
                ans[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
    ans
}


#[inline(always)]
pub(super) fn column_sum(matrix: &[Vec<f64>]) -> Vec<f64> {
    let ncol = matrix[0].len();
    let mut columns = vec![0.0; ncol];

    matrix.iter()
        .for_each(|row| {
            columns.iter_mut()
                .zip(row)
                .for_each(|(c, r)| {
                    *c += r;
                });
        });
    columns
}

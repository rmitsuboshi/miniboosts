use polars::prelude::*;

use crate::Classifier;



pub(super) fn edge_of<C>(
    data: &DataFrame,
    target: &Series,
    dist: &[f64],
    classifiers: &[C],
    weights: &[f64]
) -> f64
    where C: Classifier
{
    target.i64()
        .expect("The target is not a dtype i64")
        .into_iter()
        .zip(dist.iter().copied())
        .enumerate()
        .map(|(i, (y, d))| {
            let y = y.unwrap() as f64;
            let p = confidence(i, data, classifiers, weights);
            d * y * p
        })
        .sum::<f64>()
}


pub(super) fn edge_of_h<C>(
    data: &DataFrame,
    target: &Series,
    dist: &[f64],
    h: &C
) -> f64
    where C: Classifier
{
    target.i64()
        .expect("The target is not a dtype i64")
        .into_iter()
        .zip(dist.iter().copied())
        .enumerate()
        .map(|(i, (y, d))|
            d * y.unwrap() as f64 * h.confidence(data, i)
        )
        .sum::<f64>()
}


pub(super) fn confidence<C>(
    index: usize,
    data: &DataFrame,
    classifiers: &[C],
    weights: &[f64]
) -> f64
    where C: Classifier
{
    classifiers.iter()
        .zip(weights)
        .map(|(h, w)| w * h.confidence(data, index))
        .sum::<f64>()
}



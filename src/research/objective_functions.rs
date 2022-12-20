use polars::prelude::*;

use crate::Classifier;


/// Compute the soft-margin objective value
pub fn soft_margin_objective<C>(
    data: &DataFrame,
    target: &Series,
    weights: &[f64],
    hypotheses: &[C],
    nu: f64,
) -> f64
    where C: Classifier,
{
    let n_sample = data.shape().0 as f64;
    assert!((1.0..=n_sample).contains(&nu));

    let mut margins = target.i64()
        .expect("The class is not a dtype i64")
        .into_iter()
        .enumerate()
        .map(|(i, y)| {
            let y = y.unwrap() as f64;

            let p = weights.into_iter()
                .zip(hypotheses)
                .map(|(w, h)| w * h.confidence(data, i))
                .sum::<f64>();
            y * p
        })
        .collect::<Vec<f64>>();

    // Sort the margin vector in ascending order.
    margins.sort_by(|a, b| a.partial_cmp(&b).unwrap());

    let unit = 1.0 / nu;
    let mut total = 1.0_f64;

    let mut objective = 0.0_f64;

    for margin in margins {
        if total < unit {
            objective += total * margin;
            break;
        } else {
            objective += unit * margin;
            total -= unit;

            // total might be negative by a numerical error.
            if total < 0.0 {
                break;
            }
        }
    }


    objective
}

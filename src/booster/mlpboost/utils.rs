use crate::{Sample, Classifier};



pub(super) fn edge_of<C>(
    sample: &Sample,
    dist: &[f64],
    classifiers: &[C],
    weights: &[f64]
) -> f64
    where C: Classifier
{
    sample.target()
        .into_iter()
        .zip(dist.iter().copied())
        .enumerate()
        .map(|(i, (y, d))| {
            d * y * confidence(i, sample, classifiers, weights)
        })
        .sum::<f64>()
}


pub(super) fn edge_of_h<C>(
    sample: &Sample,
    dist: &[f64],
    h: &C
) -> f64
    where C: Classifier
{
    sample.target()
        .into_iter()
        .zip(dist.iter().copied())
        .enumerate()
        .map(|(i, (y, d))| d * y * h.confidence(sample, i))
        .sum::<f64>()
}


pub(super) fn confidence<C>(
    index: usize,
    sample: &Sample,
    classifiers: &[C],
    weights: &[f64]
) -> f64
    where C: Classifier
{
    classifiers.iter()
        .zip(weights)
        .map(|(h, w)| w * h.confidence(sample, index))
        .sum::<f64>()
}



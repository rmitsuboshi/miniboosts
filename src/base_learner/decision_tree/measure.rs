//! Defines some functions that measure the node impurity.
use crate::Sample;


use std::collections::HashMap;


/// Compute the binary entropy of the given subsample.
#[inline]
pub(super) fn entropic_impurity<D, L>(sample:  &Sample<D, L>,
                                      dist:    &[f64],
                                      indices: &[usize])
    -> f64
    where L: Eq + std::hash::Hash + Clone
{
    let mut grouped = HashMap::new();
    for &i in indices.iter() {
        let (_, l) = &sample[i];
        let cnt = grouped.entry(l).or_insert(0.0);
        *cnt += dist[i];
    }


    let total = grouped.values().sum::<f64>();


    grouped.into_iter()
        .map(|(_, p)| {
            let p = p / total;
            -p * p.ln()
        })
        .sum::<f64>()
}


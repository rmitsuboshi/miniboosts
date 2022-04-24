
pub struct Node<D> {
    split_rule: SplitRule<D>,
    left_node:  Box<Node<D>>,
    right_node: Box<Node<D>>,
    prediction: Option<Label>,
}


/// This function computes the binary entropy 
/// `-(p/m) * log2(p/m) - ((m-p)/m) * log2((m-p)/m)`,
/// where
/// * `m = total_size` is the number of examples, and
/// * `p = positive_size` is the number of examples classififed as positive.
/// 
#[inline]
fn binary_entropy(positive_size: usize, total_size: usize) -> f64 {
    let p = positive_size as f64;
    let m = total_size as f64;

    - (p / m) * (p / m).log2() - ((m - p) / m) * ((m - p) / m).ln()
}


/// This function computes the binary gini impurity
/// `1 - ((p/m)^2 + ((m-p)/m)^2`,
/// * `m = total_size` is the number of examples, and
/// * `p = positive_size` is the number of examples classififed as positive.
/// 
#[inline]
fn binary_gini(positive_size: usize, total_size: usize) -> f64 {
    let p = positive_size as f64;
    let m = total_size as f64;

    1.0 - ((p / m).powi(2) + ((m - p) / m).powi(2))
}




/// Returns the evaluated score of `predictor` based on `criterion`.
#[inline]
fn score<D>(sample:    &Sample<D>,
            dist:      &[f64],
            indices:   &[usize],
            predictor: &InnerPredictor,
            criterion: Criterion)
    -> f64
    where D: Data<Output = f64>
{
    let total_size = indices.len();
    let positive_size = indices.iter()
        .filter_map(|&i| {
            let (data, _) = &sample[i];
            match predictor.transit(data) {
                Child::Positive => Some(()),
                Child::Negative => None
            }
        })
        .count();

    match criterion {
        Criterion::Entropy => binary_entropy(positive_size, total_size),
        Criterion::Gini => binary_gini(positive_size, total_size)
    }

}

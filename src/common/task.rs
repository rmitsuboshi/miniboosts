
/// Defines machine learning tasks.
/// This enum is defined for neural network.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Task {
    /// Binary classification. The target labels are `+1` or `-1`.
    Binary,
    /// Multi-class classification.
    /// Labels are `0`, `1`, ..., `K-1`.
    MultiClass(usize),
    /// Regression
    Regression,
}


/// Convert the given vector into a scalar in `[1, +1]`.
/// ```text
/// index -> label
/// 0     -> -1
/// 1     -> +1
/// ```
#[inline(always)]
pub(crate) fn binarize<T: AsRef<[f64]>>(y: T) -> f64 {
    let y = y.as_ref();
    // If `y.len()` is a scalar,
    // this code interprets it 
    // as the confidence to be `+1`
    if y.len() == 1 {
        1.0 - 2.0 * y[0]
        // if y[0] >= 0.5 { y[0] } else { y[0] - 1.0 }
    } else if y.len() == 2 {
        if y[0] > y[1] { -y[0] } else { y[1] }
    } else {
        panic!("Cannot convert a vector of length >= 3 into a scalar");
    }
}


#[inline(always)]
pub(crate) fn discretize<T: AsRef<[f64]>>(y: T, n_class: usize) -> f64 {
    let y = y.as_ref();

    assert_eq!(y.len(), n_class);

    let k = y.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap().0;

    k as f64
}


/// Convert a label into a vector.
#[inline(always)]
pub(crate) fn vectorize(y: f64, n_class: usize) -> Vec<f64> {
    if n_class == 1 { return vec![y]; }
    let y = if y <= 0.0 { 0_usize } else { y as usize };
    let mut vec = vec![0.0; n_class];

    vec[y] = 1.0;
    vec
}

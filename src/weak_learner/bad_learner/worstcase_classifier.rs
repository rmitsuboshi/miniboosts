use crate::{
    Sample,
    Classifier
};


/// A hypothesis returned by `BadBaseLearner.`
/// This struct is user for demonstrating the worst-case LPBoost behavior.
#[derive(Debug, Clone, PartialEq)]
pub struct BadClassifier {
    tag: i8,
    index: usize,
    shift: usize,
    gap: f64,
    eps: f64,
}


impl BadClassifier {
    /// Construct a new instance of `Self.`
    pub(super) fn new(
        tag: i8,
        index: usize,
        shift: usize,
        gap: f64,
        eps: f64,
    ) -> Self
    {
        Self { tag, index, shift, gap, eps, }
    }
}


impl Classifier for BadClassifier {
    /// Let `h` be an instance of `BadClassifier`.
    /// `h` predicts according to the following rule.
    /// 1. `self.tag == 0`
    ///     ```txt
    ///     h.confidence(sample, i) = +1,                 if i < self.index
    ///                               -1 + (gap - 1)*eps, if i == self.index
    ///                               -1 + gap*eps,       if i > self.index
    ///     ```
    /// 2. `self.tag == 1`
    ///     ```txt
    ///     h.confidence(sample, i) = +1,                 if i < self.index
    ///                               -1 + (gap - 1)*eps, if i == self.index
    ///                               -1 + gap*eps,       if i > self.index
    ///     ```
    /// 3. Otherwise (`self.tag > 1`)
    ///     ```txt
    ///     h.confidence(sample, i) = -1 + eps,           if i <= self.index
    ///                                1 - eps,           if i > self.index
    ///     ```
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        let target = sample.target();
        let y = target[row];
        if self.tag == 0 {
            if row < self.index {
                1f64 * y
            } else if (self.index..self.index+self.shift).contains(&row) {
                (-1f64 + 2f64 * self.eps) * y
            } else {
                (-1f64 + 3f64 * self.eps) * y
            }
        } else if self.tag == 1 {
            let lef = self.index;
            let mid = lef + self.shift;
            let rig = mid + self.shift;
            if (lef..mid).contains(&row) {
                1f64 * y
            } else if (mid..rig).contains(&row) {
                (-1f64 + (self.gap - 1f64) * self.eps) * y
            } else {
                (-1f64 + self.gap * self.eps) * y
            }
        } else {
            if row < self.index {
                (-1f64 + self.eps) * y
            } else {
                (1f64 - self.eps) * y
            }
        }
    }
}

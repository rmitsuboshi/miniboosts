use std::fmt;

/// Loss functions available to Neural networks.
#[derive(Clone, Copy)]
pub enum NNLoss {
    /// Least squared loss
    L2,

    /// Cross-entropy loss
    CrossEntropy,
}


impl fmt::Display for NNLoss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::L2 => "L2",
            Self::CrossEntropy => "Cross Entropy",
        };

        write!(f, "{name}")
    }
}


impl NNLoss {
    /// Returns a differentiate for a given minibatch.
    #[inline(always)]
    pub fn diff(&self, p: Vec<f64>, y: Vec<f64>) -> Vec<f64> {
        match self {
            Self::L2 => l2_diff(p, y),
            Self::CrossEntropy => cross_entropy_diff(p, y),
        }
    }
}


#[inline(always)]
fn l2_diff(p: Vec<f64>, y: Vec<f64>) -> Vec<f64> {
    p.into_iter()
        .zip(y)
        .map(|(pi, yi)| pi - yi)
        .collect()
}

#[inline(always)]
fn cross_entropy_diff(p: Vec<f64>, y: Vec<f64>) -> Vec<f64> {
    p.into_iter()
        .zip(y)
        .map(|(pi, yi)| pi - yi)
        .collect()
}

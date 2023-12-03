/// This trait defines the loss functions.
pub trait LossFunction {
    /// The name of the loss function.
    fn name(&self) -> &str;
    /// Loss value for a single point.
    fn eval_at_point(&self, prediction: f64, true_value: f64) -> f64;


    /// Loss value for a single point.
    fn eval(&self, predictions: &[f64], target: &[f64]) -> f64 {
        let n_items = predictions.len();

        assert_eq!(n_items, target.len());


        predictions.iter()
            .zip(target)
            .map(|(&p, &y)| self.eval_at_point(p, y))
            .sum::<f64>()
            / n_items as f64
    }

    /// Gradient vector at current point.
    fn gradient(&self, predictions: &[f64], target: &[f64]) -> Vec<f64>;


    /// Best coffecient for the newly-attained hypothesis.
    fn best_coefficient(
        &self, 
        residuals: &[f64],
        predictions: &[f64],
    ) -> f64;
}


/// Some well-known loss functions.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GBMLoss {
    /// `L1`-loss.
    /// This loss function is also known as
    /// **Least Absolute Deviation (LAD)**.
    L1,

    /// `L2`-loss.
    /// This loss function is also known as
    /// **Mean Squared Error (MSE)**.
    L2,
}


impl LossFunction for GBMLoss {
    fn name(&self) -> &str {
        match self {
            Self::L1 => "L1 loss",
            Self::L2 => "L2 loss",
        }
    }


    fn eval_at_point(&self, prediction: f64, true_value: f64) -> f64 {
        match self {
            Self::L1 => (prediction - true_value).abs(),
            Self::L2 => (prediction - true_value).powi(2),
        }
    }


    fn gradient(&self, predictions: &[f64], target: &[f64]) -> Vec<f64>
    {
        let n_sample = predictions.len() as f64;
        assert_eq!(n_sample as usize, target.len());


        match self {
            Self::L1 => {
                predictions.iter()
                    .zip(target)
                    .map(|(p, y)| (p - y).signum() / n_sample)
                    .collect()
            },
            Self::L2 => {
                predictions.iter()
                    .zip(target)
                    .map(|(p, y)| 2.0 * (p - y) / n_sample)
                    .collect()
            },
        }
    }


    fn best_coefficient(
        &self, 
        targets: &[f64],
        predictions: &[f64],
    ) -> f64
    {
        match self {
            Self::L1 => {
                let mut items = targets.iter()
                    .zip(predictions)
                    .filter_map(|(&r, &p)| 
                        if p == 0.0 { None } else { Some((p.abs(), r / p)) }
                    )
                    .collect::<Vec<_>>();

                weighted_median(&mut items[..])
            },
            Self::L2 => {
                let y_sum = targets.iter().sum::<f64>();
                let p_sum = predictions.iter().sum::<f64>();

                assert_ne!(p_sum, 0.0);

                y_sum / p_sum
            },
        }
    }
}


/// Returns a median of the given array
fn weighted_median(items: &mut [(f64, f64)]) -> f64 {
    let n_items = items.len();

    assert!(n_items > 0);

    if n_items == 1 {
        return items[0].1;
    }
    items.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let total_weight = items.iter()
        .map(|(w, _)| *w)
        .sum::<f64>();


    let mut partial_sum = 0.0_f64;
    for (w, x) in items {
        partial_sum += *w;
        if partial_sum >= 0.5 * total_weight {
            return *x;
        }
    }

    unreachable!()
}



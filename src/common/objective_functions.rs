use crate::{
    Sample,
    CombinedHypothesis,
    Classifier,
};

use super::checker;


/// A trait that logs objective value of boosting algorithms.
pub trait ObjectiveFunction<H> {
    /// Evaluates given combined hypothesis.
    fn eval(&self, sample: &Sample, hypothesis: &CombinedHypothesis<H>) -> f64;
}


/// Soft margin objective is the objective function 
/// for the following boosting algorithms:
/// - [`LPBoost`]
/// - [`ERLPBoost`]
/// - [`CERLPBoost`]
/// - [`SoftBoost`]
/// - [`SmoothBoost`]
/// - [`MLPBoost`]
/// 
/// [`LPBoost`]: crate::booster::LPBoost
/// [`ERLPBoost`]: crate::booster::ERLPBoost
/// [`CERLPBoost`]: crate::booster::CERLPBoost
/// [`SoftBoost`]: crate::booster::SoftBoost
/// [`SmoothBoost`]: crate::booster::SmoothBoost
/// [`MLPBoost`]: crate::booster::MLPBoost
pub struct SoftMarginObjective(f64);

impl SoftMarginObjective {
    /// Construct a new instance of `SoftMarginObjective`.
    pub fn new(nu: f64) -> Self {
        Self(nu)
    }
}

impl<H> ObjectiveFunction<H> for SoftMarginObjective
    where H: Classifier,
{
    fn eval(
        &self,
        sample: &Sample,
        hypothesis: &CombinedHypothesis<H>,
    ) -> f64
    {
        checker::check_sample(sample);
        let n_sample = sample.shape().0;
        checker::check_nu(self.0, n_sample);


        let target = sample.target();

        let mut margins = hypothesis.confidence_all(sample)
            .into_iter()
            .zip(target.into_iter())
            .map(|(hx, y)| y * hx)
            .collect::<Vec<f64>>();

        margins.sort_by(|a, b| a.partial_cmp(&b).unwrap());

        let unit_weight = 1.0 / self.0;
        let mut weight_left = 1.0;

        let mut objective_value = 0.0;
        for yh in margins {
            if weight_left > unit_weight {
                objective_value += unit_weight * yh;
                weight_left -= unit_weight;
            } else {
                objective_value += weight_left * yh;
                break;
            }
        }


        objective_value
    }
}


/// Hard margin objective is the objective function 
/// for the following boosting algorithms:
/// - [AdaBoostV]
/// - [TotalBoost]
/// 
/// [AdaBoostV]: crate::booster::AdaBoostV
/// [TotalBoost]: crate::booster::TotalBoost
pub struct HardMarginObjective(SoftMarginObjective);

impl HardMarginObjective {
    /// Construct a new instance of `HardMarginObjective`.
    pub fn new() -> Self {
        let soft_margin = SoftMarginObjective::new(1.0);
        Self(soft_margin)
    }
}

impl<H> ObjectiveFunction<H> for HardMarginObjective
    where H: Classifier,
{
    fn eval(
        &self,
        sample: &Sample,
        hypothesis: &CombinedHypothesis<H>
    ) -> f64
    {
        self.0.eval(sample, hypothesis)
    }
}

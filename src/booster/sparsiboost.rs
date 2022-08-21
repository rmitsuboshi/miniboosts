/// Provides the `SparsiBoost` by Grønlund et al.
/// See the [paper](https://arxiv.org/abs/1901.10789).

use crate::{Data, Sample};
use crate::{Classifier, CombinedClassifier};
use crate::{Booster, BaseLearner};
use crate::AdaBoostV;


/// Defines the behavor of `SparsiBoost`.
/// This struct is defined as a wrapper of some boosting algorithm.
pub struct SparsiBoost<B> {
    pub(self) booster: B,
}


impl SparsiBoost<B> {
    /// Returns an instance of `SparsiBoost`.
    pub fn init<D, L>(sample: &Sample<D, L>) -> Self {
        let booster = AdaBoostV::init(sample);
        Self { booster }
    }


    /// Set the tolerance parameter.
    pub fn set_tolerance(&mut self, tolerance: f64) -> Self {
        self.booster.set_tolerance(tolerance);
    }



    /// Returns the max iteration 
    /// for finding an approximate solution.
    #[inline]
    pub fn max_loop(&self) -> usize {
        self.booster.max_loop()
    }

}



impl<D, L, C> Booster<D, L, C> for SparsiBoost
    where D: Data,
          L: Clone + Into<f64>,
          C: Classifier<D, L> + Eq + PartialEq,
{
    fn run<B>(&mut self,
              base_learner: &B,
              sample:       &Sample<D, L>,
              eps:          f64)
        -> CombinedClassifier<D, L, C>
        where B: BaseLearner<D, L, Clf = C>,
    {
        let clfs = self.booster.run(
            base_learner,
            sample,
            eps
        );
    }
}




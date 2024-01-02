use rand::prelude::*;
use colored::Colorize;
use crate::Sample;

use std::iter::Iterator;

const WIDTH: usize = 9;

/// A struct that generates 
/// pairs of training/test sample for cross validation.
/// # Example
/// ```no_run
/// use miniboosts::prelude::*;
/// use miniboosts::CrossValidation;
///
/// fn main() {
///     let sample = SampleReader::new()
///         .file(path)
///         .has_header(true)
///         .target_feature("class")
///         .read()
///         .unwrap();
///     let cv = CrossValidation::new(&sample)
///         .n_folds(5)
///         .verbose(true)
///         .seed(777)
///         .shuffle();
///     for (train, test) in cv {
///         let nu = train.shape().0 as f64 * 0.01;
///         let mut booster = LPBoost::init(&train)
///             .tolerance(TOLERANCE)
///             .nu(nu);
///         let tree = DecisionTreeBuilder::new(&train)
///             .max_depth(3)
///             .criterion(Criterion::Entropy)
///             .build();
///         let f = booster.run(&tree);
/// 
///         let train_loss = zero_one_loss(&train, &f);
///         let test_loss = zero_one_loss(&test, &f);
///         println!("[train: {train_loss}] [test: {test_loss}]");
///     }
/// }
///
/// fn zero_one_loss<H>(sample: &Sample, f: &H) -> f64
///     where H: Classifier
/// {
///     let n_sample = sample.shape().0 as f64;
/// 
///     let target = sample.target();
/// 
///     f.predict_all(sample)
///         .into_iter()
///         .zip(target.into_iter())
///         .map(|(hx, &y)| if hx != y as i64 { 1.0 } else { 0.0 })
///         .sum::<f64>()
///         / n_sample
/// }
/// ```
pub struct CrossValidation<'a> {
    train_size: usize,
    current_fold: usize,
    n_folds: usize,
    seed: u64,
    sample: &'a Sample,
    ix: Vec<usize>,
    verbose: bool,
}


impl<'a> CrossValidation<'a> {
    /// Construct a new instance of `CrossValidation.`
    #[inline]
    pub fn new(sample: &'a Sample) -> Self {
        let n_sample = sample.shape().0;
        let train_size = (n_sample as f64 * 0.8) as usize;
        let ix = (0..n_sample).collect::<Vec<_>>();
        Self {
            current_fold: 0,
            n_folds: 5,
            seed: 1234,
            verbose: false,
            train_size,
            sample,
            ix,
        }
    }


    /// Set the ratio of training sample.
    /// Default value is `0.8`.
    #[inline]
    pub fn train_ratio(mut self, ratio: f64) -> Self {
        assert!(
            0f64 < ratio && ratio < 1f64,
            "Training ratio should be in `[0, 1)`."
        );
        let n_sample = self.sample.shape().0 as f64;
        self.train_size = (ratio * n_sample) as usize;
        self
    }


    /// Set the number of folds.
    /// Default value is `5.`
    #[inline]
    pub fn n_folds(mut self, n_folds: usize) -> Self {
        self.n_folds = n_folds;
        self
    }


    /// Set the seed of the randomness for shuffling.
    /// Default vaule is `1234.`
    #[inline]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }


    /// Set the verbose parameter.
    /// If `true`, `CrossValidation` prints some information
    /// when generating a train/test pair.
    /// Default vaule is `false.`
    #[inline]
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }


    /// Shuffle the training sample.
    /// By default, `CrossValidation` does not shuffle the sample.
    #[inline]
    pub fn shuffle(mut self) -> Self {
        let mut rng = StdRng::seed_from_u64(self.seed);
        self.ix.shuffle(&mut rng);
        self
    }



    /// Returns the training/test sample for `i`th fold.
    #[inline]
    fn fold_at(&self, i: usize) -> (Sample, Sample) {
        let sample_size = self.sample.shape().0;
        let test_size = sample_size - self.train_size;
        let (start, end) = (i*test_size, (i+1)*test_size);
        self.sample.split(&self.ix, start, end)
    }
}


impl<'a> Iterator for CrossValidation<'a> {
    type Item = (Sample, Sample);
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_fold >= self.n_folds { return None; }

        let output = self.fold_at(self.current_fold);
        self.current_fold += 1;

        if self.verbose {
            let train_size = output.0.shape().0;
            let test_size = output.1.shape().0;
            println!(
                "{}    {}    {}",
                format!("  [{: >3}'th fold]", self.current_fold).bold().red(),
                format!("[TRAIN {:>WIDTH$}]", train_size).bold().green(),
                format!("[TEST {:>WIDTH$}]", test_size).bold().yellow(),
            );
        }

        Some(output)
    }
}




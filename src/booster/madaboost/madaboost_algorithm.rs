//! Provides [`MadaBoost`] by Domingo and Watanabe, 2000.
use rayon::prelude::*;


use crate::{
    Booster,
    WeakLearner,
    Classifier,
    WeightedMajority,
    Sample,

    common::utils,
    research::Research,
};

use std::ops::ControlFlow;


/// The MadaBoost algorithm 
/// proposed by Carlos Domingo and Osamu Watanabe, 2000.
/// 
/// This algorithm is based on the paper: 
///
/// [
/// MadaBoost: A Modification of AdaBoost
/// ](https://www.learningtheory.org/colt2000/papers/DomingoWatanabe.pdf)
/// by Carlos Domingo and Osamu Watanabe.
/// 
/// MadaBoost is a boosting algorithm for binary classification 
/// that minimizes exponential loss over a set of training examples.
///
/// This struct provides the `MB:1/2` algorithm.
///
/// # Convergence rate
/// Given a set of training examples of size `m > 0`
/// and an accuracy parameter `ε > 0`,
/// `MadaBoost` finds an convex combination of hypotheses
/// whose empirical loss is less than `ε`
/// in `O( m / ε² )` iterations.
/// 
/// 
/// ```no_run
/// use miniboosts::prelude::*;
/// 
/// // Read the training sample from the CSV file.
/// // We use the column named `class` as the label.
/// let sample = SampleReader::new()
///     .file(path_to_file)
///     .has_header(true)
///     .target_feature("class")
///     .read()
///     .unwrap();
/// 
/// // Initialize `MadaBoost` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis whose training error is
/// // less than `0.01` if the traing examples are linearly separable.
/// let mut booster = MadaBoost::init(&sample)
///     .tolerance(0.01);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .criterion(Criterion::Entropy)
///     .build();
/// 
/// // Run `MadaBoost` and obtain the resulting hypothesis `f`.
/// let f = booster.run(&weak_learner);
/// 
/// // Get the predictions on the training set.
/// let predictions = f.predict_all(&sample);
/// 
/// // Get the number of training examples.
/// let n_sample = sample.shape().0 as f64;
/// 
/// // Calculate the training loss.
/// let target = sample.target();
/// let training_loss = target.into_iter()
///     .zip(predictions)
///     .map(|(&y, fx)| if y as i64 == fx { 0.0 } else { 1.0 })
///     .sum::<f64>()
///     / n_sample;
/// 
///
/// println!("Training Loss is: {training_loss}");
/// ```
pub struct MadaBoost<'a, F> {
    // Training sample
    sample: &'a Sample,

    // Weights for each instances in `sample.`
    // At the end of round `t,`
    // the `i`th element of `betas` holds
    // `ln Bt[i] = sum_{k=1}^{T} ( y[i] hk(x[i]) ln beta[k] ).`
    betas: Vec<f64>,

    // Tolerance parameter
    tolerance: f64,


    // Weights on hypotheses in `hypotheses`
    alphas: Vec<f64>,

    // Hypohteses obtained by the weak-learner.
    hypotheses: Vec<F>,


    // Max iteration until MadaBoost guarantees the optimality.
    max_iter: usize,


    // Optional. If this value is `Some(it)`,
    // the algorithm terminates after `it` iterations.
    force_quit_at: Option<usize>,

    // Terminated iteration.
    // MadaBoost terminates in eary step 
    // if the training set is linearly separable.
    terminated: usize,
}


impl<'a, F> MadaBoost<'a, F> {
    /// Constructs a new instance of `MadaBoostV`.
    /// 
    /// Time complexity: `O(1)`.
    #[inline]
    pub fn init(sample: &'a Sample) -> Self {
        let n_sample = sample.shape().0;

        Self {
            sample,

            betas: Vec::new(),
            tolerance: 1.0 / (n_sample as f64 + 1.0),

            alphas: Vec::new(),
            hypotheses: Vec::new(),

            max_iter: usize::MAX,
            force_quit_at: None,
            terminated: usize::MAX,
        }
    }


    /// Returns the maximum iteration
    /// of the `MadaBoost` to find a combined hypothesis
    /// that has error at most `tolerance`.
    /// After the `self.max_loop()` iterations,
    /// `MadaBoost` guarantees zero training error in terms of zero-one loss
    /// if the training examples are linearly separable.
    /// 
    /// Time complexity: `O(1)`.
    pub fn max_loop(&self) -> usize {
        let n_sample = self.sample.shape().0 as f64;

        ((n_sample - 1f64) / self.tolerance.powi(2)).ceil() as usize
    }


    /// Force quits after at most `it` iterations.
    /// Note that if `it` is smaller than the iteration bound
    /// for MadaBoost, the returned hypothesis has no guarantee.
    /// 
    /// Time complexity: `O(1)`.
    pub fn force_quit_at(mut self, it: usize) -> Self {
        self.force_quit_at = Some(it);
        self
    }


    /// Set the tolerance parameter.
    /// `MadaBoostV` terminates immediately
    /// after reaching the specified `tolerance`.
    /// 
    /// Time complexity: `O(1)`.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }


    /// Returns a alpha on the new hypothesis.
    /// `update_params` also updates `self.betas`.
    /// 
    /// `MadaBoost` uses exponential update,
    /// which is numerically unstable so that I adopt a logarithmic computation.
    /// 
    /// Time complexity: `O( m ln(m) )`,
    /// where `m` is the number of training examples.
    /// The additional `ln(m)` term comes from the numerical stabilization.
    #[inline]
    fn update_params(
        &mut self,
        margins: Vec<f64>,
        edge: f64
    ) -> f64
    {
        // Defines the `epsilon` by transforming `edge.`
        // We assume that `eps` is in `[0.0, 1.0).`
        let eps = 0.5f64 * (1f64 - edge);
        assert!((0f64..1f64).contains(&eps), "EPS: {}", eps);
        let eps2 = (0.5f64 * eps).sqrt();
        let beta = (eps2 / (1f64 - eps2)).sqrt();

        // Compute the weight on new hypothesis.
        // This is the returned value of this function.
        let alpha = (1f64 / beta).ln();


        // To prevent overflow, take the logarithm.
        self.betas.par_iter_mut()
            .zip(margins)
            .for_each(|(b, yh)| { *b += yh * beta; });


        alpha
    }


    fn beta2distribution(&self) -> Vec<f64> {
        let n_sample = self.sample.shape().0;

        let weights = {
            let mut weights = self.betas.iter()
                .copied()
                .map(|b| b.min(1f64))
                .collect::<Vec<_>>();
            weights.shrink_to_fit();
            weights
        };

        // Sort indices by ascending order
        let mut indices = (0..n_sample).into_par_iter()
            .collect::<Vec<usize>>();
        indices.sort_unstable_by(|&i, &j| {
            weights[i].partial_cmp(&weights[j]).unwrap()
        });


        let mut normalizer = weights[indices[0]];
        for i in indices.into_iter().skip(1) {
            let mut a = normalizer;
            let mut b = weights[i];
            if a < b {
                std::mem::swap(&mut a, &mut b);
            }

            normalizer = a + (1.0 + (b - a).exp()).ln();
        }



        weights.into_iter()
            .map(|b| (b - normalizer).exp())
            .collect()
    }
}


impl<F> Booster<F> for MadaBoost<'_, F>
    where F: Classifier + Clone,
{
    type Output = WeightedMajority<F>;


    fn name(&self) -> &str {
        "MadaBoost"
    }


    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_sample, n_feature) = self.sample.shape();
        let quit = if let Some(it) = self.force_quit_at {
            format!("At round {it}")
        } else {
            format!("-")
        };
        let info = Vec::from([
            ("# of examples", format!("{}", n_sample)),
            ("# of features", format!("{}", n_feature)),
            ("Tolerance", format!("{}", self.tolerance)),
            ("Max iteration", format!("{}", self.max_loop())),
            ("Force quit", quit),
        ]);
        Some(info)
    }


    fn preprocess<W>(
        &mut self,
        _weak_learner: &W,
    )
        where W: WeakLearner<Hypothesis = F>
    {
        self.sample.is_valid_binary_instance();
        // Initialize parameters
        let n_sample = self.sample.shape().0;
        let uni = 1.0 / n_sample as f64;
        self.betas = vec![uni; n_sample];

        self.alphas = Vec::new();
        self.hypotheses = Vec::new();


        self.max_iter = self.max_loop();

        if let Some(it) = self.force_quit_at {
            self.max_iter = it;
        }
    }


    fn boost<W>(
        &mut self,
        weak_learner: &W,
        iteration: usize,
    ) -> ControlFlow<usize>
        where W: WeakLearner<Hypothesis = F>,
    {
        if self.max_iter < iteration {
            return ControlFlow::Break(self.max_iter);
        }


        let dist = self.beta2distribution();
        // Get a new hypothesis
        let h = weak_learner.produce(self.sample, &dist[..]);


        // Each element in `margins` is the product of
        // the predicted vector and the correct vector
        let margins = utils::margins_of_hypothesis(self.sample, &h);


        let edge = utils::inner_product(&margins, &dist[..]);


        // If `h` predicted all the examples in `sample` correctly,
        // use it as the combined classifier.
        if edge.abs() >= 1.0 {
            self.terminated = iteration;
            self.alphas = vec![edge.signum()];
            self.hypotheses = vec![h];
            return ControlFlow::Break(iteration);
        }


        // Compute the alpha on the new hypothesis
        let alpha = self.update_params(margins, edge);
        self.alphas.push(alpha);
        self.hypotheses.push(h);

        ControlFlow::Continue(())
    }


    fn postprocess<W>(
        &mut self,
        _weak_learner: &W,
    ) -> Self::Output
        where W: WeakLearner<Hypothesis = F>
    {
        WeightedMajority::from_slices(&self.alphas[..], &self.hypotheses[..])
    }
}


impl<H> Research for MadaBoost<'_, H>
    where H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        WeightedMajority::from_slices(&self.alphas[..], &self.hypotheses[..])
    }
}

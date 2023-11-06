//! Provides [`GraphSepBoost`](Graph Separation Boosting)
//! by Noga Alon, Alon Gonen, Elad Hazan, and Shay Moran, 2023.
use crate::{
    Booster,
    WeakLearner,
    Classifier,
    NaiveAggregation,
    Sample,

    research::Research,
};

use std::ops::ControlFlow;
use std::collections::HashSet;


/// The Graph Separation Boosting algorithm proposed by Robert E. Schapire and Yoav Freund.
/// 
/// The algorithm is comes from the following paper: 
/// [Boosting Simple Learners](https://theoretics.episciences.org/10757/pdf)
/// by Noga Alon, Alon Gonen, Elad Hazan, and Shay Moran.
/// 
/// Given a `γ`-weak learner and a set `S` of training examples of size `m`,
/// `GraphSepBoost` terminates in `O( ln(m) / γ)` rounds.
///
/// To guarantee the generalization ability,
/// one needs to use a **simple** weak-learner.
/// 
/// # Example
/// The following code shows a small example 
/// for running [`Graph Separation Boosting`](Graph Separation Boosting).  
/// See also:
/// - [`DecisionTree`]
/// - [`DecisionTreeClassifier`]
/// - [`NaiveAggregation<F>`]
/// - [`Sample`]
/// 
/// [`DecisionTree`]: crate::weak_learner::DecisionTree
/// [`DecisionTreeClassifier`]: crate::weak_learner::DecisionTreeClassifier
/// [`NaiveAggregation<F>`]: crate::hypothesis::NaiveAggregation
/// 
/// 
/// ```no_run
/// use miniboosts::prelude::*;
/// 
/// // Read the training sample from the CSV file.
/// // We use the column named `class` as the label.
/// let has_header = true;
/// let sample = Sample::from_csv(path_to_csv_file, has_header)
///     .expect("Failed to read the training sample")
///     .set_target("class");
/// 
/// // Initialize `Graph Separation Boosting` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis whose training error is
/// // less than `0.01` if the traing examples are linearly separable.
/// let mut booster = GraphSepBoost::init(&sample)
///     .tolerance(0.01);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTreeBuilder::new(&sample)
///     .max_depth(1)
///     .criterion(Criterion::Entropy)
///     .build();
/// 
/// // Run `GraphSepBoost` and obtain the resulting hypothesis `f`.
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
///     .map(|(&y, fx) if y as i64 == fx { 0.0 } else { 1.0 })
///     .sum::<f64>()
///     / n_sample;
/// 
///
/// println!("Training Loss is: {training_loss}");
/// ```
pub struct GraphSepBoost<'a, F> {
    // Training sample
    sample: &'a Sample,


    // The number of edges of each vertex (which corresponds to some instance)
    edges: Vec<HashSet<usize>>,


    // Tolerance parameter
    tolerance: f64,


    // Hypohteses obtained by the weak-learner.
    hypotheses: Vec<F>,
}


impl<'a, F> GraphSepBoost<'a, F> {
    /// Constructs a new instance of `GraphSepBoost`.
    /// 
    /// Time complexity: `O(1)`.
    #[inline]
    pub fn init(sample: &'a Sample) -> Self {
        let n_sample = sample.shape().0;

        Self {
            sample,

            tolerance: 1.0 / (n_sample as f64 + 1.0),
            hypotheses: Vec::new(),

            edges: Vec::new(),
        }
    }
}

impl<'a, F> GraphSepBoost<'a, F>
    where F: Classifier
{
    /// Set the tolerance parameter.
    /// `GraphSepBoost` terminates immediately
    /// after reaching the specified `tolerance`.
    /// 
    /// Time complexity: `O(1)`.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }


    /// Returns a weight on the new hypothesis.
    /// `update_params` also updates `self.dist`.
    /// 
    /// `GraphSepBoost` uses exponential update,
    /// which is numerically unstable so that I adopt a logarithmic computation.
    /// 
    /// Time complexity: `O( m ln(m) )`,
    /// where `m` is the number of training examples.
    /// The additional `ln(m)` term comes from the numerical stabilization.
    #[inline]
    fn update_params(&mut self, h: &F) {
        let predictions = h.predict_all(self.sample);

        let (n_sample, _) = self.sample.shape();
        for i in 0..n_sample {
            for j in i+1..n_sample {
                if predictions[i] != predictions[j] {
                    self.edges[i].remove(&j);
                    self.edges[j].remove(&i);
                }
            }
        }
    }
}


impl<F> Booster<F> for GraphSepBoost<'_, F>
    where F: Classifier + Clone,
{
    type Output = NaiveAggregation<F>;


    fn name(&self) -> &str {
        "Graph Separation Boosting"
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

        let target = self.sample.target();

        self.edges = vec![HashSet::new(); n_sample];
        for i in 0..n_sample {
            for j in i+1..n_sample {
                if target[i] != target[j] {
                    self.edges[i].insert(j);
                    self.edges[j].insert(i);
                }
            }
        }

        self.hypotheses = Vec::new();
    }


    fn boost<W>(
        &mut self,
        weak_learner: &W,
        iteration: usize,
    ) -> ControlFlow<usize>
        where W: WeakLearner<Hypothesis = F>,
    {
        let n_edges_2 = self.edges.iter()
            .map(|edge| edge.len())
            .sum::<usize>();
        if n_edges_2 == 0 {
            return ControlFlow::Break(iteration);
        }
        let denom = n_edges_2 as f64;

        let dist = self.edges.iter()
            .map(|edge| edge.len() as f64 / denom)
            .collect::<Vec<_>>();

        // Get a new hypothesis
        let h = weak_learner.produce(self.sample, &dist);
        self.update_params(&h);
        self.hypotheses.push(h);

        ControlFlow::Continue(())
    }


    fn postprocess<W>(
        &mut self,
        _weak_learner: &W,
    ) -> Self::Output
        where W: WeakLearner<Hypothesis = F>
    {
        let hypotheses = std::mem::take(&mut self.hypotheses);
        NaiveAggregation::new(hypotheses, &self.sample)
    }
}


impl<H> Research for GraphSepBoost<'_, H>
    where H: Classifier + Clone,
{
    type Output = NaiveAggregation<H>;
    fn current_hypothesis(&self) -> Self::Output {
        NaiveAggregation::from_slice(&self.hypotheses, &self.sample)
    }
}

//! Provides [`GraphSeparationBoosting`](Graph Separation Boosting)
//! by Noga Alon, Alon Gonen, Elad Hazan, and Shay Moran, 2023.
use miniboosts_core::{
    Booster,
    WeakLearner,
    Classifier,
    Sample,
};
use logging::CurrentHypothesis;
use hypotheses::NaiveAggregation;

use std::ops::ControlFlow;
use std::collections::HashSet;

pub struct GraphSeparationBoosting<'a, H> {
    // Training sample
    sample: &'a Sample,

    // The number of edges of each vertex (which corresponds to some instance)
    edges: Vec<HashSet<usize>>,

    // Hypohteses obtained by the weak-learner.
    hypotheses: Vec<H>,

    // The number of edges at the end of the previous round.
    n_edges: usize,
}

impl<'a, H> GraphSeparationBoosting<'a, H> {
    /// Constructs a new instance of `GraphSeparationBoosting`.
    /// 
    /// Time complexity: `O(1)`.
    #[inline]
    pub fn init(sample: &'a Sample) -> Self {
        Self {
            sample,
            hypotheses: Vec::new(),
            edges: Vec::new(),
            n_edges: usize::MAX,
        }
    }
}

impl<H> GraphSeparationBoosting<'_, H>
    where H: Classifier
{
    /// Returns a weight on the new hypothesis.
    /// `update_params` also updates `self.dist`.
    /// 
    /// `GraphSeparationBoosting` uses exponential update,
    /// which is numerically unstable so that I adopt a logarithmic computation.
    /// 
    /// Time complexity: `O( m ln(m) )`,
    /// where `m` is the number of training examples.
    /// The additional `ln(m)` term comes from the numerical stabilization.
    #[inline]
    fn update_params(&mut self, h: &H) {
        let predictions = h.predict_all(self.sample);

        let (n_examples, _) = self.sample.shape();
        for i in 0..n_examples {
            for j in i+1..n_examples {
                if predictions[i] != predictions[j] {
                    self.edges[i].remove(&j);
                    self.edges[j].remove(&i);
                }
            }
        }
    }
}

impl<H> Booster<H> for GraphSeparationBoosting<'_, H>
    where H: Classifier + Clone,
{
    type Output = NaiveAggregation<H>;

    fn name(&self) -> &str {
        "Graph Separation Boosting"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_examples, n_feature) = self.sample.shape();
        let info = Vec::from([
            ("# of examples", format!("{n_examples}")),
            ("# of features", format!("{n_feature}")),
        ]);
        Some(info)
    }

    fn preprocess(&mut self) {
        self.sample.is_valid_binary_instance();
        // Initialize parameters
        let n_examples = self.sample.shape().0;

        let target = self.sample.target();

        self.edges = vec![HashSet::new(); n_examples];
        for i in 0..n_examples {
            for j in i+1..n_examples {
                if target[i] != target[j] {
                    self.edges[i].insert(j);
                    self.edges[j].insert(i);
                }
            }
        }

        self.n_edges = self.edges
            .iter()
            .map(|edges| edges.len())
            .sum();

        self.hypotheses = Vec::new();
    }

    fn boost<W>(&mut self, weak_learner: &W, iteration: usize)
        -> ControlFlow<usize>
        where W: WeakLearner<Hypothesis = H>,
    {
        if self.n_edges == 0 {
            return ControlFlow::Break(iteration);
        }

        let dist = self.edges.iter()
            .map(|edge| edge.len() as f64 / self.n_edges as f64)
            .collect::<Vec<_>>();

        // Get a new hypothesis
        let h = weak_learner.produce(self.sample, &dist);
        self.update_params(&h);
        self.hypotheses.push(h);

        let n_edges = self.edges
            .iter()
            .map(|edges| edges.len())
            .sum::<usize>();
        if self.n_edges == n_edges {
            eprintln!("[WARN] number of edges does not decrease.");
            return ControlFlow::Break(iteration+1);
        }
        self.n_edges = n_edges;

        ControlFlow::Continue(())
    }

    fn postprocess(&mut self) -> Self::Output {
        let hypotheses = std::mem::take(&mut self.hypotheses);
        NaiveAggregation::new(hypotheses, self.sample)
    }
}

impl<H> CurrentHypothesis for GraphSeparationBoosting<'_, H>
    where H: Classifier + Clone,
{
    type Output = NaiveAggregation<H>;
    fn current_hypothesis(&self) -> Self::Output {
        NaiveAggregation::from_slice(&self.hypotheses, self.sample)
    }
}


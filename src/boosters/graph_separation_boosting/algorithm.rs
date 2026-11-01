//! Implementation of [`GraphSeparationBoosting`] (Alon, Gonen, Hazan, and
//! Moran, "Boosting Simple Learners", TheoretiCS, 2023).
//!
//! See [`GraphSeparationBoosting`] for the algorithm, guarantees, and
//! limitations.
use crate::core::CurrentHypothesis;
use crate::core::{Booster, Classifier, Sample, WeakLearner};
use crate::hypotheses::NaiveAggregation;

use std::collections::HashMap;
use std::ops::ControlFlow;

/// A booster that separates every oppositely labeled pair of training
/// examples and aggregates the weak hypotheses into a consistent classifier.
///
/// # Algorithm
///
/// This follows Algorithm 1 (Graph Separation Boosting) of Alon, Gonen,
/// Hazan, and Moran. Consider the graph whose vertices are the `m` training
/// examples and whose edges join examples with opposite labels. Round `t`:
///
/// 1. Gives the weak learner the distribution proportional to vertex degree,
///    ```txt
///    d_i = deg_t(i) / Σ_j deg_t(j),
///    ```
///    and receives `h_t`.
/// 2. Removes every edge `{i, j}` with `h_t(x_i) ≠ h_t(x_j)`, where
///    predictions are signs of the confidence (`0` counts as `+1`).
/// 3. Stops when no edges remain.
///
/// The output is a [`NaiveAggregation`] of `h_1, ..., h_T`: it predicts `+1`
/// on `x` exactly when the prediction vector `(h_1(x), ..., h_T(x))` equals
/// that of some positive training example, and `-1` otherwise. This is the
/// aggregation rule in the proof of Lemma 4.2 of the paper; once all edges are
/// removed, it is consistent with the training sample.
///
/// Implementation choices:
///
/// - The graph is not stored explicitly. Examples with identical prediction
///   history form a group; a remaining edge joins exactly the oppositely
///   labeled pairs within a group. Each round refines the groups, using
///   expected `O(m)` time and memory in addition to evaluating `h_t`.
/// - The paper draws an i.i.d. subsample from the degree distribution and
///   runs the weak learner on it; here the distribution itself is passed to
///   the weak learner, as with the other boosters.
/// - If a round removes no edges, boosting stops with a warning on standard
///   error. This safeguard is not part of the paper; the output then need not
///   be consistent with the training sample.
///
/// # Theoretical guarantee
///
/// Assume the sample is `γ`-realizable with respect to the base class and the
/// weak learner's expected correlation is at least `γ/2` on every such
/// distribution (Eq. (3) of the paper). Then the number of rounds `T`
/// satisfies `Pr[T ≥ t] ≤ exp(2 ln(m) - tγ/2)`, so `E[T] = O(ln(m) / γ)`
/// (Theorem 2.3, restated as Theorem 4.3). The paper also gives
/// sample-complexity bounds, which depend on how the aggregation rule is
/// chosen.
///
/// # Limitations
///
/// - The weak-learning assumption is not checked at runtime. A weak learner
///   that separates no new pair triggers the stall stop above.
/// - Without a stall, boosting continues until the training sample is
///   separated, so there is no iteration budget, and the output can overfit
///   noisy labels (it is consistent with the training sample).
/// - The output depends on the stored positive prediction vectors, so it
///   keeps one bit set of length `T` per positive training example.
/// - Labels must be exactly `-1` and `+1`; [`Booster::preprocess`] panics
///   otherwise.
///
/// # Parameters
///
/// None besides the training sample.
///
/// # Reference
///
/// Noga Alon, Alon Gonen, Elad Hazan, and Shay Moran.
/// [Boosting Simple Learners](https://theoretics.episciences.org/10757).
/// TheoretiCS, Volume 2 (2023), Article 7.
///
/// # Examples
/// Train on a held-out split with decision stumps and evaluate on the test
/// part.
/// ```
/// use std::io::BufReader;
/// use miniboosts::{Sample, prelude::*};
///
/// // Labels follow the sign of `x + y`, so no single stump separates them.
/// let csv = b"x,y,class\n\
///     -2,1,-1\n-1,-2,-1\n1,-3,-1\n-3,2,-1\n0,-1,-1\n-2,-1,-1\n\
///     2,1,1\n1,2,1\n-1,3,1\n3,-1,1\n0,2,1\n2,2,1\n";
/// let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
///     .unwrap()
///     .set_target("class");
///
/// // A fixed permutation keeps the split reproducible:
/// // rows ix[0..4] form the test set, the other 8 rows the training set.
/// let ix = [0, 6, 4, 10, 1, 2, 3, 5, 7, 8, 9, 11];
/// let (train, test) = sample.split(&ix, 0, 4);
///
/// // The weak learner sees only the training split.
/// let learner = DecisionTreeBuilder::new(&train).max_depth(1).build();
///
/// // No parameters: boosting runs until every oppositely labeled pair of
/// // training examples is separated by some hypothesis (or a round stalls).
/// let mut booster = GraphSeparationBoosting::init(&train);
/// let f = booster.run(&learner);
///
/// let predictions = f.predict_all(&test);
/// assert!(predictions.iter().all(|&p| p == -1 || p == 1));
/// let correct = predictions
///     .iter()
///     .zip(test.target())
///     .filter(|&(&p, &y)| p as f64 == y)
///     .count();
/// let accuracy = correct as f64 / test.shape().0 as f64;
/// println!("GraphSeparationBoosting test accuracy: {accuracy}");
///
/// // Unless a stall warning was printed to standard error, `f` is
/// // consistent with the training sample, i.e. its training accuracy is 1.
/// // On unseen data it predicts +1 only for prediction vectors seen on a
/// // positive training example.
/// let train_predictions = f.predict_all(&train);
/// let train_correct = train_predictions
///     .iter()
///     .zip(train.target())
///     .filter(|&(&p, &y)| p as f64 == y)
///     .count();
/// println!("training accuracy: {}", train_correct as f64 / train.shape().0 as f64);
/// ```
pub struct GraphSeparationBoosting<'a, H> {
    // Training sample
    sample: &'a Sample,

    // Rows in one group have identical predictions under every collected hypothesis.
    groups: Vec<usize>,
    // [negative labels, positive labels] for each prediction-signature group.
    counts: Vec<[usize; 2]>,

    // Hypotheses obtained by the weak learner.
    hypotheses: Vec<H>,

    // Degree sum (twice the edge count) at the end of the previous round.
    n_edges: u128,
}

impl<'a, H> GraphSeparationBoosting<'a, H> {
    /// Constructs a new `GraphSeparationBoosting` for `sample`.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{Sample, prelude::*};
    /// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// let learner = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    /// let mut booster = GraphSeparationBoosting::init(&sample);
    /// // The output is a `NaiveAggregation`, not a weighted majority vote.
    /// let f = booster.run(&learner);
    /// let predictions = f.predict_all(&sample);
    /// assert!(predictions.iter().all(|&p| p == -1 || p == 1));
    /// ```
    #[inline]
    pub fn init(sample: &'a Sample) -> Self {
        Self {
            sample,
            hypotheses: Vec::new(),
            groups: Vec::new(),
            counts: Vec::new(),
            n_edges: 0,
        }
    }
}

impl<H> GraphSeparationBoosting<'_, H>
where
    H: Classifier,
{
    /// Refine prediction-signature groups instead of materializing graph edges.
    ///
    /// In Algorithm 1 of Alon et al., [Boosting Simple Learners (2023)](https://theoretics.episciences.org/10757/pdf),
    /// an edge survives exactly
    /// when its endpoints have opposite labels and identical prediction history.
    /// Thus each group induces a complete bipartite graph. A row's degree is
    /// the opposite-label count, and the degree sum is `2 * sum(n_neg * n_pos)`.
    /// Splitting by `(old group, prediction)` implements precisely the paper's
    /// edge deletion, with expected O(m) time and O(m) memory, excluding prediction.
    fn update_params(&mut self, h: &H) {
        let predictions = h.predict_all(self.sample);
        let mut children = HashMap::with_capacity(self.groups.len());
        let mut counts = Vec::<[usize; 2]>::new();
        for (row, group) in self.groups.iter_mut().enumerate() {
            let key = (*group, predictions[row]);
            *group = *children.entry(key).or_insert_with(|| {
                counts.push([0, 0]);
                counts.len() - 1
            });
            counts[*group][usize::from(self.sample.target()[row] > 0.0)] += 1;
        }
        self.counts = counts;
    }

    fn degree_sum(&self) -> u128 {
        // u128 also avoids overflowing usize when the implicit graph is dense.
        self.counts
            .iter()
            .map(|&[negative, positive]| 2 * negative as u128 * positive as u128)
            .sum()
    }

    fn distribution(&self) -> Vec<f64> {
        self.groups
            .iter()
            .zip(self.sample.target())
            .map(|(&group, &label)| {
                self.counts[group][usize::from(label < 0.0)] as f64 / self.n_edges as f64
            })
            .collect()
    }
}

impl<H> Booster<H> for GraphSeparationBoosting<'_, H>
where
    H: Classifier + Clone,
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
        // The two-count representation requires the paper's {-1, +1} labels.
        // Core validation currently only warns for other two-class encodings.
        assert!(
            self.sample.target().iter().all(|&y| y == -1.0 || y == 1.0),
            "GraphSeparationBoosting requires labels -1 and +1"
        );
        // Initialize parameters
        let n_examples = self.sample.shape().0;

        let positives = self.sample.target().iter().filter(|&&y| y > 0.0).count();
        self.groups = vec![0; n_examples];
        self.counts = vec![[n_examples - positives, positives]];
        self.n_edges = self.degree_sum();

        self.hypotheses = Vec::new();
    }

    fn boost<W>(&mut self, weak_learner: &W, iteration: usize) -> ControlFlow<usize>
    where
        W: WeakLearner<Hypothesis = H>,
    {
        if self.n_edges == 0 {
            return ControlFlow::Break(iteration);
        }

        let dist = self.distribution();

        // Get a new hypothesis
        let h = weak_learner.produce(self.sample, &dist);
        self.update_params(&h);
        self.hypotheses.push(h);

        let n_edges = self.degree_sum();
        // Preserve the existing one-round stall safeguard (an implementation
        // extension, not the paper's empty-graph stopping condition).
        if self.n_edges == n_edges {
            eprintln!("[WARN] number of edges does not decrease.");
            return ControlFlow::Break(iteration + 1);
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
where
    H: Classifier + Clone,
{
    type Output = NaiveAggregation<H>;
    fn current_hypothesis(&self) -> Self::Output {
        NaiveAggregation::from_slice(&self.hypotheses, self.sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::io::{BufReader, Cursor};

    #[derive(Clone)]
    struct Scores(Vec<f64>);

    impl Classifier for Scores {
        fn confidence(&self, _: &Sample, row: usize) -> f64 {
            self.0[row]
        }
    }

    struct Scripted {
        hypotheses: Vec<Scores>,
        next: Cell<usize>,
        distributions: RefCell<Vec<Vec<f64>>>,
    }

    impl WeakLearner for Scripted {
        type Hypothesis = Scores;
        fn produce(&self, _: &Sample, dist: &[f64]) -> Scores {
            self.distributions.borrow_mut().push(dist.to_vec());
            let next = self.next.get();
            self.next.set(next + 1);
            self.hypotheses[next].clone()
        }
    }

    // Independent reference: explicitly store every undirected edge as two
    // adjacency entries, as the original implementation did.
    fn compare_with_graph(labels: &[f64], hypotheses: Vec<Scores>) {
        let mut csv = "x,class\n".to_string();
        for (i, label) in labels.iter().enumerate() {
            csv.push_str(&format!("{i},{label}\n"));
        }
        let sample = Sample::from_reader(BufReader::new(Cursor::new(csv)), true)
            .unwrap()
            .set_target("class");
        let learner = Scripted {
            hypotheses,
            next: Cell::new(0),
            distributions: RefCell::new(Vec::new()),
        };
        let mut booster = GraphSeparationBoosting::init(&sample);
        // Reuse the booster after postprocess takes its hypotheses.
        for _ in 0..2 {
            booster.preprocess();
            learner.next.set(0);
            learner.distributions.borrow_mut().clear();
            let mut graph: Vec<Vec<bool>> = labels
                .iter()
                .map(|a| labels.iter().map(|b| a != b).collect())
                .collect();
            let mut retained = Vec::new();
            for iteration in 1..=learner.hypotheses.len() + 1 {
                let degrees: Vec<usize> = graph
                    .iter()
                    .map(|neighbors| neighbors.iter().filter(|&&edge| edge).count())
                    .collect();
                let before: usize = degrees.iter().sum();
                assert_eq!(booster.n_edges, before as u128);
                if before == 0 {
                    let calls = learner.next.get();
                    assert_eq!(
                        booster.boost(&learner, iteration),
                        ControlFlow::Break(iteration)
                    );
                    assert_eq!(learner.next.get(), calls);
                    break;
                }
                let expected_dist: Vec<_> = degrees
                    .iter()
                    .map(|&degree| degree as f64 / before as f64)
                    .collect();
                assert_eq!(booster.distribution(), expected_dist);
                let h = learner.hypotheses[iteration - 1].clone();
                let predictions = h.predict_all(&sample);
                for (i, neighbors) in graph.iter_mut().enumerate() {
                    for (j, edge) in neighbors.iter_mut().enumerate() {
                        *edge &= predictions[i] == predictions[j];
                    }
                }
                let after = graph.iter().flatten().filter(|&&edge| edge).count();
                retained.push(h);
                let expected_flow = if before == after {
                    ControlFlow::Break(iteration + 1)
                } else {
                    ControlFlow::Continue(())
                };
                assert_eq!(booster.boost(&learner, iteration), expected_flow);
                assert_eq!(learner.distributions.borrow().last(), Some(&expected_dist));
                assert_eq!(booster.degree_sum(), after as u128);
                // Check every pair, not only the degree sum: retained edges are
                // exactly opposite-label pairs sharing a prediction history.
                for i in 0..labels.len() {
                    for j in 0..labels.len() {
                        assert_eq!(
                            graph[i][j],
                            labels[i] != labels[j] && booster.groups[i] == booster.groups[j]
                        );
                    }
                }
                let reference = NaiveAggregation::from_slice(&retained, &sample);
                assert_eq!(
                    booster.current_hypothesis().predict_all(&sample),
                    reference.predict_all(&sample)
                );
                if expected_flow.is_break() {
                    break;
                }
            }
            let reference = NaiveAggregation::new(retained, &sample);
            assert_eq!(
                booster.postprocess().predict_all(&sample),
                reference.predict_all(&sample)
            );
        }
    }

    #[test]
    #[should_panic(expected = "GraphSeparationBoosting requires labels -1 and +1")]
    fn rejects_zero_one_labels_before_constructing_groups() {
        let sample = Sample::from_reader(BufReader::new(Cursor::new("x,class\n0,0\n1,1\n")), true)
            .unwrap()
            .set_target("class");
        GraphSeparationBoosting::<Scores>::init(&sample).preprocess();
    }

    #[test]
    fn signature_groups_match_explicit_graph_with_ties_stalls_and_reuse() {
        let labels = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
        // Two refinements, then full separation. Zero confidence must retain
        // Classifier's existing positive-label tie convention.
        compare_with_graph(
            &labels,
            vec![
                Scores(vec![-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
                Scores(vec![-1.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0]),
                Scores(labels.to_vec()),
            ],
        );
        // Immediate stall and a stall after nontrivial refinement must retain
        // the final hypothesis and the old Break(iteration + 1) convention.
        compare_with_graph(&labels, vec![Scores(vec![0.0; 8])]);
        compare_with_graph(
            &labels,
            vec![
                Scores(vec![-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]),
                Scores(vec![1.0; 8]),
            ],
        );
        // Exhaust every initial prediction and every nonconstant labeling on
        // four rows, including isolated vertices and unbalanced classes.
        for labels_mask in 1..15 {
            let labels: Vec<_> = (0..4)
                .map(|i| {
                    if labels_mask & (1 << i) == 0 {
                        -1.0
                    } else {
                        1.0
                    }
                })
                .collect();
            for prediction_mask in 0..16 {
                let scores = (0..4)
                    .map(|i| {
                        if prediction_mask & (1 << i) == 0 {
                            -1.0
                        } else {
                            0.0
                        }
                    })
                    .collect();
                compare_with_graph(&labels, vec![Scores(scores), Scores(labels.clone())]);
            }
        }
    }
}

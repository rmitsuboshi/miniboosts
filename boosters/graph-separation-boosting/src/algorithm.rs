//! Provides [`GraphSeparationBoosting`](Graph Separation Boosting)
//! by Noga Alon, Alon Gonen, Elad Hazan, and Shay Moran, 2023.
use hypotheses::NaiveAggregation;
use miniboosts_core::CurrentHypothesis;
use miniboosts_core::{Booster, Classifier, Sample, WeakLearner};

use std::collections::HashMap;
use std::ops::ControlFlow;

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
    /// Constructs a new instance of `GraphSeparationBoosting`.
    ///
    /// Time complexity: `O(1)`.
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

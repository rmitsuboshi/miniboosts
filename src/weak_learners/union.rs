//! A weak learner over the union of the hypothesis classes of other weak
//! learners.
//!
//! [`WeakLearnerUnion`] holds weak learners, possibly with different
//! hypothesis types, and returns the candidate with the largest edge in each
//! round. See its documentation for what this does and does not guarantee.

use crate::core::{Classifier, Sample, WeakLearner, helpers};
use rayon::prelude::*;
use std::sync::Arc;

/// The type-erased hypothesis returned by [`WeakLearnerUnion`].
pub type DynClassifier = Arc<dyn Classifier + Send + Sync>;

/// Runs the weak learners it holds and returns the candidate with the largest
/// edge.
///
/// Given a distribution `d`, each member `j` produces a candidate `h_j`, and
/// the learner returns the `h_j` that maximizes the signed edge
/// `γ_j = Σ_i d_i y_i h_j(x_i)`. Ties go to the member added first, so the
/// choice is deterministic if the members are.
///
/// Members run in parallel on rayon's thread pool, so they must be
/// `Send + Sync`, and a member's state must not depend on the order of the
/// members' calls. To measure sequential time, run in a pool with one thread.
///
/// # Guarantee
/// The returned edge is `max_j γ_j`. So if, for every distribution a booster
/// generates, *some* member has edge at least `γ`, a booster's weak-learning
/// assumption holds with `γ`; no single member has to satisfy it for every
/// distribution. If every member maximizes the edge exactly over its class,
/// the union does so over the union of the classes. For `k` classes whose
/// sign predictions have VC dimensions `v_1, ..., v_k`, the union has VC
/// dimension at most `Σ_j v_j + k - 1`.
///
/// # Limitations
/// - The returned hypothesis is the best of the candidates, not necessarily
///   the best in the union of the classes: a member that does not maximize the
///   edge (such as [`LogisticRegression`](crate::LogisticRegression) or
///   [`NaiveBayes`](crate::NaiveBayes)) can miss a better hypothesis of its
///   class. Then stopping criteria that assume an edge maximizer, such as
///   those of the LPBoost family, lose their guarantee.
/// - A larger edge in each round does not imply a lower final error. The union
///   can generalize worse than one of its members alone.
/// - Edges are comparable only if confidences lie in `[-1, 1]`. Like the
///   [`Classifier`] trait, this learner does not check the range. Among
///   otherwise equal hypotheses, it prefers those with larger confidence
///   magnitudes, such as `Sign` over `Tanh` outputs.
/// - Hypotheses are type-erased ([`DynClassifier`]), so the combined hypothesis
///   does not implement `Serialize`, `Deserialize`, `Debug`, or `PartialEq`.
/// - Every member runs in every round.
/// - It applies to classifiers only. Real-valued hypotheses, as used by
///   [`GradientBoosting`](crate::GradientBoosting) and
///   [`LogitBoost`](crate::LogitBoost), have unbounded outputs that the edge
///   cannot compare.
///
/// # Examples
/// ```
/// use std::io::BufReader;
/// use miniboosts::{Sample, prelude::*};
///
/// let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
/// let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
///     .unwrap()
///     .set_target("class");
/// // The members return different hypothesis types.
/// let learner = WeakLearnerUnion::new()
///     .with(DecisionTreeBuilder::new(&sample).max_depth(1).build())
///     .with(NaiveBayes::new());
/// let f = AdaBoost::init(&sample).force_quit_at(5).run(&learner);
/// assert_eq!(f.predict_all(&sample), vec![-1, -1, -1, 1, 1, 1]);
/// ```
#[derive(Default)]
pub struct WeakLearnerUnion<'a> {
    members: Vec<Box<dyn WeakLearner<Hypothesis = DynClassifier> + Send + Sync + 'a>>,
}

impl<'a> WeakLearnerUnion<'a> {
    /// Constructs a union without members. Add them with
    /// [`WeakLearnerUnion::with`].
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// See [`WeakLearnerUnion`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `learner` as a member.
    ///
    /// Time complexity: amortized `O(1)`.
    ///
    /// # Examples
    /// See [`WeakLearnerUnion`].
    pub fn with<W>(mut self, learner: W) -> Self
    where
        W: WeakLearner + Send + Sync + 'a,
        W::Hypothesis: Classifier + Send + Sync + 'static,
    {
        self.members.push(Box::new(Erased(learner)));
        self
    }
}

impl WeakLearner for WeakLearnerUnion<'_> {
    type Hypothesis = DynClassifier;

    fn name(&self) -> &str {
        "Weak learner union"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let names = self
            .members
            .iter()
            .map(|member| member.name())
            .collect::<Vec<_>>()
            .join(", ");
        Some(vec![("Members", names)])
    }

    /// Returns the candidate with the largest edge under `dist`; see
    /// [`WeakLearnerUnion`].
    ///
    /// Work: the members' total, plus `O(m k)` confidence evaluations for `m`
    /// examples and `k` members. With enough threads, the time is close to that
    /// of the slowest member.
    ///
    /// # Panics
    /// Panics if there are no members or a candidate's edge is not finite, as
    /// well as whenever a member panics on these inputs.
    fn produce(&self, sample: &Sample, dist: &[f64]) -> Self::Hypothesis {
        assert!(
            !self.members.is_empty(),
            "WeakLearnerUnion needs at least one member"
        );
        // `collect` keeps the members' order, so ties do not depend on the
        // thread schedule.
        let candidates = self
            .members
            .par_iter()
            .map(|member| {
                let candidate = member.produce(sample, dist);
                let edge = helpers::edge(sample, dist, &candidate);
                assert!(
                    edge.is_finite(),
                    "the edge of a candidate from {} is not finite",
                    member.name()
                );
                (edge, candidate)
            })
            .collect::<Vec<_>>();
        let mut best: Option<(f64, DynClassifier)> = None;
        for (edge, candidate) in candidates {
            if best.as_ref().is_none_or(|(largest, _)| edge > *largest) {
                best = Some((edge, candidate));
            }
        }
        let (_, hypothesis) = best.expect("there is at least one member");
        hypothesis
    }
}

// Erases a member's hypothesis type.
struct Erased<W>(W);

impl<W> WeakLearner for Erased<W>
where
    W: WeakLearner,
    W::Hypothesis: Classifier + Send + Sync + 'static,
{
    type Hypothesis = DynClassifier;

    fn name(&self) -> &str {
        self.0.name()
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        self.0.info()
    }

    fn produce(&self, sample: &Sample, dist: &[f64]) -> Self::Hypothesis {
        Arc::new(self.0.produce(sample, dist))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[derive(Clone)]
    struct Table(Vec<f64>);

    impl Classifier for Table {
        fn confidence(&self, _: &Sample, row: usize) -> f64 {
            self.0[row]
        }
    }

    struct Fixed(Vec<f64>);

    impl WeakLearner for Fixed {
        type Hypothesis = Table;

        fn produce(&self, _: &Sample, _: &[f64]) -> Table {
            Table(self.0.clone())
        }
    }

    fn four_examples() -> Sample {
        let rows = vec![vec![-2.0], vec![-1.0], vec![1.0], vec![2.0]];
        Sample::from_dense(&rows, Some(&[-1.0, -1.0, 1.0, 1.0])).unwrap()
    }

    #[test]
    fn the_choice_follows_the_distribution() {
        let sample = four_examples();
        let (constant, half) = (vec![1.0; 4], vec![-0.5, -0.5, 0.5, 0.5]);
        let union = WeakLearnerUnion::new()
            .with(Fixed(constant.clone()))
            .with(Fixed(half.clone()));
        // Uniform: the constant has edge 0 and the other 0.5.
        let h = union.produce(&sample, &[0.25; 4]);
        assert_eq!(h.confidence_all(&sample), half);
        // Mass on the positive examples: the constant has edge 1, the other 0.5.
        let h = union.produce(&sample, &[0.0, 0.0, 0.5, 0.5]);
        assert_eq!(h.confidence_all(&sample), constant);
    }

    #[test]
    fn ties_go_to_the_first_member() {
        let sample = four_examples();
        // Both have edge exactly 1/2 under the uniform distribution.
        let (first, second) = (vec![-1.0, 1.0, 1.0, 1.0], vec![-1.0, -1.0, 1.0, -1.0]);
        let union = WeakLearnerUnion::new()
            .with(Fixed(first.clone()))
            .with(Fixed(second));
        let h = union.produce(&sample, &[0.25; 4]);
        assert_eq!(h.confidence_all(&sample), first);
    }

    #[test]
    fn the_choice_does_not_depend_on_the_thread_schedule() {
        let n = 16;
        let rows: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
        let target: Vec<f64> = (0..n).map(|i| if i < 8 { -1.0 } else { 1.0 }).collect();
        let sample = Sample::from_dense(&rows, Some(&target)).unwrap();
        let dist = vec![1.0 / n as f64; n];
        // Member k misclassifies only example k: all edges equal 7/8 exactly.
        let wrong_at = |k: usize| {
            let mut values = target.clone();
            values[k] = -values[k];
            values
        };
        let mut union = WeakLearnerUnion::new();
        for k in 0..n {
            union = union.with(Fixed(wrong_at(k)));
        }
        for _ in 0..50 {
            let h = union.produce(&sample, &dist);
            assert_eq!(h.confidence_all(&sample), wrong_at(0));
        }
    }

    #[test]
    fn the_edge_is_signed() {
        let sample = four_examples();
        // Edges -1 and 1/2: a large negative edge must not win.
        let (wrong, small) = (vec![1.0, 1.0, -1.0, -1.0], vec![-0.5, -0.5, 0.5, 0.5]);
        let union = WeakLearnerUnion::new()
            .with(Fixed(wrong))
            .with(Fixed(small.clone()));
        let h = union.produce(&sample, &[0.25; 4]);
        assert_eq!(h.confidence_all(&sample), small);
    }

    #[test]
    fn a_single_member_passes_through() {
        let sample = four_examples();
        let values = vec![1.0, -1.0, 1.0, -1.0];
        let union = WeakLearnerUnion::new().with(Fixed(values.clone()));
        let h = union.produce(&sample, &[0.25; 4]);
        assert_eq!(h.confidence_all(&sample), values);
    }

    #[test]
    fn the_edge_is_the_maximum_over_different_hypothesis_types() {
        let rows = vec![
            vec![-2.0, 1.0],
            vec![-1.0, -1.0],
            vec![0.5, 2.0],
            vec![1.0, -2.0],
            vec![2.0, 0.5],
            vec![-0.5, 0.0],
        ];
        let target = [-1.0, -1.0, 1.0, 1.0, -1.0, 1.0];
        let sample = Sample::from_dense(&rows, Some(&target)).unwrap();
        let tree = || DecisionTreeBuilder::new(&sample).max_depth(1).build();
        let union = WeakLearnerUnion::new().with(tree()).with(NaiveBayes::new());
        for dist in [
            [1.0 / 6.0; 6],
            [0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.05, 0.05, 0.4, 0.05, 0.4, 0.05],
        ] {
            let by_tree = helpers::edge(&sample, &dist, &tree().produce(&sample, &dist));
            let by_bayes =
                helpers::edge(&sample, &dist, &NaiveBayes::new().produce(&sample, &dist));
            let edge = helpers::edge(&sample, &dist, &union.produce(&sample, &dist));
            // The same computation on the same inputs: no rounding difference.
            assert_eq!(edge, by_tree.max(by_bayes));
        }
    }

    #[test]
    fn boosters_accept_the_union() {
        let sample = four_examples();
        let learner = || {
            WeakLearnerUnion::new()
                .with(DecisionTreeBuilder::new(&sample).max_depth(1).build())
                .with(LogisticRegression::new())
        };
        let labels: Vec<i64> = sample.target().iter().map(|y| *y as i64).collect();

        let f = AdaBoost::init(&sample).force_quit_at(5).run(&learner());
        assert_eq!(f.predict_all(&sample), labels);

        let f = LpBoost::init(&sample).nu(1.0).run(&learner());
        assert_eq!(f.predict_all(&sample), labels);
    }

    #[test]
    fn info_names_the_members() {
        let union = WeakLearnerUnion::new()
            .with(NaiveBayes::new())
            .with(LogisticRegression::new());
        assert_eq!(
            union.info().unwrap(),
            vec![(
                "Members",
                "Gaussian naive Bayes, Logistic regression".to_string()
            )]
        );
    }

    #[test]
    #[should_panic(expected = "at least one member")]
    fn an_empty_union_panics() {
        let sample = four_examples();
        WeakLearnerUnion::new().produce(&sample, &[0.25; 4]);
    }

    #[test]
    #[should_panic(expected = "not finite")]
    fn a_non_finite_edge_panics() {
        let sample = four_examples();
        WeakLearnerUnion::new()
            .with(Fixed(vec![f64::NAN; 4]))
            .produce(&sample, &[0.25; 4]);
    }
}

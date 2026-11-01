//! A weak learner that returns an edge-maximizing member of a finite
//! hypothesis class.
//!
//! [`FiniteClass`] holds hypotheses `h_1, ..., h_k` and, given a
//! distribution `d` from a booster, returns the `h_j` (or, optionally, its
//! negation) with the largest edge `Σ_i d_i y_i h_j(x_i)`. Because the class
//! is finite, the maximization is exact, so experiments can
//! - compute the best achievable margin `ρ*` of the class with an LP
//!   (e.g. [`soft_margin_optimization()`](crate::soft_margin_optimization())
//!   with `nu = 1`) and measure a booster's gap `ρ* - ρ` directly;
//! - reproduce constructions stated as a matrix `A = (y_i h_j(x_i))`, such
//!   as lower-bound instances, with [`FiniteClass::from_table`];
//! - run deterministically, since ties are broken by index.
//!
//! [`RbfClassBuilder`](crate::weak_learners::rbf::RbfClassBuilder) builds
//! a finite class of Gaussian radial basis functions.
//!
//! # Examples
//! Three fixed hypotheses on four examples. AdaBoost with this learner can
//! only combine them; the LP value `1/3` is the best minimal margin of any
//! convex combination.
//! ```
//! use miniboosts::{Sample, prelude::*, soft_margin_optimization};
//! use miniboosts::weak_learners::finite_class::FiniteClass;
//!
//! let sample = Sample::dummy(4);
//! let y = sample.target().to_vec();
//! // Margins y_i h_j(x_i): hypothesis j misclassifies example j + 1.
//! let margins = [[1.0, -1.0, 1.0, 1.0], [1.0, 1.0, -1.0, 1.0], [1.0, 1.0, 1.0, -1.0]];
//! let table: Vec<Vec<f64>> = margins
//!     .iter()
//!     .map(|m| m.iter().zip(&y).map(|(u, yi)| u * yi).collect())
//!     .collect();
//! let learner = FiniteClass::from_table(&sample, table);
//!
//! // ρ* of the class (an LP over convex combinations).
//! let (rho_star, _) = soft_margin_optimization(1.0, &sample, learner.hypotheses());
//! assert!((rho_star - 1.0 / 3.0).abs() < 1e-6);
//!
//! let f = AdaBoostV::init(&sample).tolerance(0.1).force_quit_at(200).run(&learner);
//! let rho = (0..4)
//!     .map(|i| y[i] * f.confidence(&sample, i))
//!     .fold(f64::INFINITY, f64::min);
//! // The measured gap; AdaBoostV's guarantee is ρ ≥ ρ* - ν = ρ* - 0.1 after
//! // ⌊2 ln 4 / 0.1²⌋ = 277 rounds, so 200 rounds carry no guarantee.
//! println!("gap {}", rho_star - rho);
//! ```
use crate::core::{Classifier, Sample, WeakLearner};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A weak learner over a finite hypothesis class `{h_1, ..., h_k}`: each
/// call returns a member with the largest edge under the given
/// distribution.
///
/// # Algorithm
/// At construction, the margins `u_ij = y_i h_j(x_i)` on the training sample
/// are computed once. Given a distribution `d`,
/// [`produce`](WeakLearner::produce) computes every edge
/// `γ_j = Σ_i d_i u_ij` and returns the [`ClassMember`] with the largest
/// one. With [`FiniteClass::with_negations`], the class also contains
/// `-h_j`, so the member with the largest `|γ_j|` is returned with the sign
/// that makes its edge nonnegative. Ties go to the smaller index, and to
/// `+h_j` over `-h_j`.
///
/// # Guarantee
/// The returned member maximizes the edge over the class exactly, up to
/// floating-point rounding in the sums (ties within rounding may resolve
/// either way).
///
/// # Limitations
/// - [`produce`](WeakLearner::produce) uses the margins cached at
///   construction. It must be called with the same training sample; this
///   is checked only through its size and labels.
/// - Each call takes `O(m k)` time, and the cache holds `m k` values
///   (`8 m k` bytes) for `m` examples and `k` hypotheses.
/// - Confidences must be finite and lie in `[-1, 1]`, the range the
///   boosters' analyses assume.
///
/// # Examples
/// Wrap any [`Classifier`]s; here, two decision stumps trained beforehand.
/// ```
/// # use std::io::BufReader;
/// use miniboosts::{Classifier, Sample, WeakLearner, prelude::*};
/// use miniboosts::weak_learners::finite_class::FiniteClass;
///
/// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
/// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
/// #     .unwrap()
/// #     .set_target("class");
/// let tree = DecisionTreeBuilder::new(&sample).max_depth(1).build();
/// let on_x = tree.produce(&sample, &[1.0 / 6.0; 6]);
/// let skewed = [0.0, 0.5, 0.0, 0.0, 0.5, 0.0];
/// let on_y = tree.produce(&sample, &skewed);
///
/// let learner = FiniteClass::new(&sample, vec![on_x, on_y]).with_negations();
/// let h = learner.produce(&sample, &[1.0 / 6.0; 6]);
/// // The stump on x separates the sample: edge 1.
/// assert_eq!((h.index, h.sign), (0, 1.0));
/// // Six terms of 1/6: the sum is 1 up to rounding.
/// assert!((learner.edges(&[1.0 / 6.0; 6])[0] - 1.0).abs() < 1e-15);
/// ```
pub struct FiniteClass<H> {
    hypotheses: Vec<H>,
    // margins[j][i] = y_i h_j(x_i)
    margins: Vec<Vec<f64>>,
    // Training labels, to check that `produce` gets the same sample.
    target: Vec<f64>,
    with_negations: bool,
}

impl<H: Classifier> FiniteClass<H> {
    /// Constructs the learner over `hypotheses` for the training `sample`,
    /// computing and caching their margins on it. Negations are not
    /// included; see [`FiniteClass::with_negations`].
    ///
    /// Time complexity: `O(m k)` confidence evaluations.
    ///
    /// # Panics
    /// Panics if `hypotheses` is empty, the sample has no labels, or a
    /// confidence is not finite or lies outside `[-1, 1]`.
    ///
    /// # Examples
    /// See [`FiniteClass`].
    pub fn new(sample: &Sample, hypotheses: Vec<H>) -> Self {
        assert!(
            !hypotheses.is_empty(),
            "a finite class needs at least one hypothesis"
        );
        let target = sample.target().to_vec();
        assert_eq!(
            target.len(),
            sample.shape().0,
            "the sample needs labels; use `Sample::set_target`"
        );
        let margins: Vec<Vec<f64>> = hypotheses
            .iter()
            .map(|h| {
                target
                    .iter()
                    .enumerate()
                    .map(|(i, y)| {
                        let c = h.confidence(sample, i);
                        assert!(
                            (-1.0..=1.0).contains(&c),
                            "hypothesis confidences must be finite and in [-1, 1], got {c}"
                        );
                        y * c
                    })
                    .collect()
            })
            .collect();
        Self {
            hypotheses,
            margins,
            target,
            with_negations: false,
        }
    }
}

impl FiniteClass<TableHypothesis> {
    /// Constructs the learner from a table of confidences:
    /// `columns[j][i] = h_j(x_i)`, the confidence of hypothesis `j` on
    /// example `i` (multiply by `y_i` for the margin). Each column becomes a
    /// [`TableHypothesis`], which is defined only on this sample's rows.
    ///
    /// Use it for matrix-game instances, such as lower-bound constructions
    /// stated through `A = (y_i h_j(x_i))`.
    ///
    /// Time complexity: `O(m k)`.
    ///
    /// # Panics
    /// Panics as [`FiniteClass::new`] does, or if a column's length differs
    /// from the number of examples.
    ///
    /// # Examples
    /// See the [module documentation](self).
    pub fn from_table(sample: &Sample, columns: Vec<Vec<f64>>) -> Self {
        let n_examples = sample.shape().0;
        assert!(
            columns.iter().all(|c| c.len() == n_examples),
            "every column needs one confidence per example"
        );
        let hypotheses = columns
            .into_iter()
            .map(|confidences| TableHypothesis { confidences })
            .collect();
        Self::new(sample, hypotheses)
    }
}

impl<H> FiniteClass<H> {
    /// Adds the negation `-h_j` of every hypothesis to the class, making it
    /// closed under negation (symmetric). Every returned edge is then
    /// nonnegative.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{Sample, WeakLearner};
    /// use miniboosts::weak_learners::finite_class::FiniteClass;
    ///
    /// let sample = Sample::dummy(2); // labels +1, -1
    /// // A single hypothesis that is wrong on both examples.
    /// let learner = FiniteClass::from_table(&sample, vec![vec![-1.0, 1.0]]);
    /// assert_eq!(learner.produce(&sample, &[0.5, 0.5]).sign, 1.0);
    /// // With negations, -h is returned and has edge 1.
    /// let learner = learner.with_negations();
    /// assert_eq!(learner.produce(&sample, &[0.5, 0.5]).sign, -1.0);
    /// ```
    pub fn with_negations(mut self) -> Self {
        self.with_negations = true;
        self
    }

    /// Returns the hypotheses of the class, `h_1, ..., h_k`, without
    /// negations.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// See the [module documentation](self).
    pub fn hypotheses(&self) -> &[H] {
        &self.hypotheses
    }

    /// Returns the edges `γ_j = Σ_i d_i y_i h_j(x_i)` of `h_1, ..., h_k`
    /// under `dist` (without negations), from the cached margins.
    ///
    /// Time complexity: `O(m k)`.
    ///
    /// # Panics
    /// Panics if `dist` does not have one entry per training example.
    ///
    /// # Examples
    /// See [`FiniteClass`].
    pub fn edges(&self, dist: &[f64]) -> Vec<f64> {
        assert_eq!(
            dist.len(),
            self.target.len(),
            "the distribution needs one entry per training example"
        );
        self.margins
            .par_iter()
            .map(|u| u.iter().zip(dist).map(|(u, d)| u * d).sum())
            .collect()
    }
}

impl<H> WeakLearner for FiniteClass<H>
where
    H: Classifier + Clone,
{
    type Hypothesis = ClassMember<H>;

    fn name(&self) -> &str {
        "Finite class"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        Some(vec![
            ("# of hypotheses", self.hypotheses.len().to_string()),
            ("Negations", self.with_negations.to_string()),
        ])
    }

    /// Returns a member with the largest edge under `dist`; see
    /// [`FiniteClass`].
    ///
    /// # Panics
    /// Panics if `sample` differs from the training sample in size or
    /// labels, or `dist` has the wrong length.
    fn produce(&self, sample: &Sample, dist: &[f64]) -> Self::Hypothesis {
        assert!(
            sample.target() == self.target.as_slice(),
            "FiniteClass must be called with its training sample"
        );
        let edges = self.edges(dist);
        let mut best = (0, 1.0, f64::NEG_INFINITY);
        for (j, &edge) in edges.iter().enumerate() {
            if edge > best.2 {
                best = (j, 1.0, edge);
            }
            if self.with_negations && -edge > best.2 {
                best = (j, -1.0, -edge);
            }
        }
        let (index, sign, _) = best;
        ClassMember {
            index,
            sign,
            hypothesis: self.hypotheses[index].clone(),
        }
    }
}

/// The member `sign · h_index` of a [`FiniteClass`] returned by
/// [`WeakLearner::produce`].
///
/// # Examples
/// ```
/// use miniboosts::{Classifier, Sample};
/// use miniboosts::weak_learners::finite_class::{ClassMember, TableHypothesis};
///
/// let sample = Sample::dummy(2);
/// let h = TableHypothesis::new(vec![0.5, -0.25]);
/// let negated = ClassMember { index: 0, sign: -1.0, hypothesis: h };
/// assert_eq!(negated.confidence_all(&sample), vec![-0.5, 0.25]);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassMember<H> {
    /// Index `j` of the hypothesis in the class.
    pub index: usize,
    /// `1.0` for `h_j`, `-1.0` for its negation.
    pub sign: f64,
    /// The hypothesis `h_j`.
    pub hypothesis: H,
}

impl<H: Classifier> Classifier for ClassMember<H> {
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        self.sign * self.hypothesis.confidence(sample, row)
    }
}

/// A hypothesis given by a table of confidences, one per row: its
/// confidence on row `i` of any sample is `confidences[i]`.
///
/// It is meant for matrix-game experiments on a fixed sample (see
/// [`FiniteClass::from_table`]); it ignores the features, so its
/// predictions on another sample are meaningless.
///
/// # Examples
/// ```
/// use miniboosts::{Classifier, Sample};
/// use miniboosts::weak_learners::finite_class::TableHypothesis;
///
/// let h = TableHypothesis::new(vec![1.0, -1.0, 0.5]);
/// assert_eq!(h.confidence_all(&Sample::dummy(3)), vec![1.0, -1.0, 0.5]);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableHypothesis {
    confidences: Vec<f64>,
}

impl TableHypothesis {
    /// Constructs the hypothesis with `confidences[i]` on row `i`.
    ///
    /// # Examples
    /// See [`TableHypothesis`].
    pub fn new(confidences: Vec<f64>) -> Self {
        Self { confidences }
    }
}

impl Classifier for TableHypothesis {
    /// # Panics
    /// Panics if `row` is out of the table's range.
    fn confidence(&self, _: &Sample, row: usize) -> f64 {
        self.confidences[row]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table_learner() -> (Sample, FiniteClass<TableHypothesis>) {
        // Sample::dummy(4) has labels +1, +1, -1, -1; the table stores
        // confidences h_j(x_i) = y_i u_ij for the margins u_j below.
        let sample = Sample::dummy(4);
        let y = sample.target().to_vec();
        let margins = [
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0],
            [0.5, 0.5, 0.5, 0.5],
        ];
        let columns = margins
            .iter()
            .map(|u| u.iter().zip(&y).map(|(u, y)| u * y).collect())
            .collect();
        let learner = FiniteClass::from_table(&sample, columns);
        (sample, learner)
    }

    #[test]
    fn edges_are_the_weighted_margins() {
        let (_, learner) = table_learner();
        let d = [0.1, 0.2, 0.3, 0.4];
        let edges = learner.edges(&d);
        let expected = [0.1 + 0.2 - 0.3 - 0.4, 0.1 - 0.2 + 0.3 + 0.4, -1.0, 0.5];
        for (e, x) in edges.iter().zip(expected) {
            assert!((e - x).abs() < 1e-15);
        }
    }

    #[test]
    fn produce_returns_an_edge_maximizer() {
        let (sample, learner) = table_learner();
        // Uniform: edges 0, 0.5, -1, 0.5. The tie at 0.5 goes to index 1.
        let h = learner.produce(&sample, &[0.25; 4]);
        assert_eq!((h.index, h.sign), (1, 1.0));
        // With negations, -h_2 has edge 1.
        let learner = learner.with_negations();
        let h = learner.produce(&sample, &[0.25; 4]);
        assert_eq!((h.index, h.sign), (2, -1.0));
        assert_eq!(h.confidence_all(&sample), sample.target().to_vec());
    }

    #[test]
    fn positive_member_wins_an_exact_tie_with_a_negation() {
        let sample = Sample::dummy(2);
        // margins (1, -1): edge 0 under the uniform distribution.
        let learner = FiniteClass::from_table(&sample, vec![vec![1.0, 1.0]]).with_negations();
        assert_eq!(learner.produce(&sample, &[0.5, 0.5]).sign, 1.0);
    }

    #[test]
    fn invalid_inputs_panic() {
        let sample = Sample::dummy(2);
        for columns in [
            vec![],
            vec![vec![1.5, 0.0]],
            vec![vec![f64::NAN, 0.0]],
            vec![vec![1.0]],
        ] {
            let result = std::panic::catch_unwind(|| FiniteClass::from_table(&sample, columns));
            assert!(result.is_err());
        }
        let learner = FiniteClass::from_table(&sample, vec![vec![1.0, 1.0]]);
        let other = Sample::dummy(3);
        let result = std::panic::catch_unwind(|| learner.produce(&other, &[0.5, 0.5]));
        assert!(result.is_err());
    }
}

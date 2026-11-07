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
//! - run deterministically, since ties are broken by index;
//! - play an adversarial weak learner that returns barely enough edge, with
//!   [`FiniteClass::weakest_with_edge`] or [`FiniteClass::edge_exactly`], to
//!   test a booster's guarantee under the weak-learning assumption with a
//!   given `γ`.
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
use std::sync::atomic::{AtomicUsize, Ordering};

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
/// Two other rules model a weak learner that gives the booster no more than
/// the weak-learning assumption promises:
/// - [`FiniteClass::weakest_with_edge`]`(γ)` returns the member with the
///   *smallest* edge among those with edge at least `γ`;
/// - [`FiniteClass::edge_exactly`]`(γ)` returns that member scaled by
///   `γ / γ_j`, so that its edge is exactly `γ`. The scaled member is
///   confidence-rated (its confidences shrink toward `0`), even if the
///   class's hypotheses predict `±1`.
///
/// If no member has edge at least `γ`, the assumption fails for that
/// distribution: both rules then return a member with the largest edge,
/// unscaled, and count the call in
/// [`FiniteClass::assumption_violations`].
///
/// # Guarantee
/// The default rule maximizes the edge over the class exactly, up to
/// floating-point rounding in the sums (ties within rounding may resolve
/// either way). The other two rules return an edge of at least `γ` (exactly
/// `γ` up to rounding for [`FiniteClass::edge_exactly`]) whenever some member
/// has one.
///
/// # Limitations
/// - [`produce`](WeakLearner::produce) uses the margins cached at
///   construction. It must be called with the same training sample; this
///   is checked only through its size and labels.
/// - Each call takes `O(m k)` time, and the cache holds `m k` values
///   (`8 m k` bytes) for `m` examples and `k` hypotheses.
/// - Confidences must be finite and lie in `[-1, 1]`, the range the
///   boosters' analyses assume.
/// - The adversarial rules choose greedily, one round at a time. They do
///   not maximize a booster's number of rounds over all sequences of
///   answers, which an adversary in a lower-bound proof may do.
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
    rule: Rule,
    // Calls in which no member had edge at least `γ` under an adversarial
    // rule.
    violations: AtomicUsize,
}

// How `produce` chooses a member.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Rule {
    Strongest,
    Weakest(f64),
    Exactly(f64),
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
            rule: Rule::Strongest,
            violations: AtomicUsize::new(0),
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

    /// Makes [`produce`](WeakLearner::produce) return, among the members with
    /// edge at least `gamma`, one with the smallest edge, instead of the
    /// largest. Ties go to the smaller index, and to `+h_j` over `-h_j`.
    /// See [`FiniteClass`] for the case without such a member.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Panics
    /// Panics unless `0 < gamma <= 1`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{Sample, WeakLearner};
    /// use miniboosts::weak_learners::finite_class::FiniteClass;
    ///
    /// let sample = Sample::dummy(4); // labels +1, +1, -1, -1
    /// // Under the uniform distribution, the edges are 1, 0.5, and 0.
    /// let columns = vec![
    ///     vec![1.0, 1.0, -1.0, -1.0],
    ///     vec![1.0, 1.0, -1.0, 1.0],
    ///     vec![1.0, 1.0, 1.0, 1.0],
    /// ];
    /// let learner = FiniteClass::from_table(&sample, columns).weakest_with_edge(0.25);
    /// assert_eq!(learner.produce(&sample, &[0.25; 4]).index, 1);
    /// assert_eq!(learner.assumption_violations(), 0);
    /// ```
    pub fn weakest_with_edge(mut self, gamma: f64) -> Self {
        check_gamma(gamma);
        self.rule = Rule::Weakest(gamma);
        self
    }

    /// Makes [`produce`](WeakLearner::produce) return the member that
    /// [`FiniteClass::weakest_with_edge`]`(gamma)` would, scaled by
    /// `gamma / γ_j` so that its edge is exactly `gamma`. The scale is
    /// stored in [`ClassMember::sign`]. See [`FiniteClass`] for the case
    /// without a member with edge at least `gamma`.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Panics
    /// Panics unless `0 < gamma <= 1`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{Classifier, Sample, WeakLearner};
    /// use miniboosts::weak_learners::finite_class::FiniteClass;
    ///
    /// let sample = Sample::dummy(4); // labels +1, +1, -1, -1
    /// // The only member is correct everywhere: edge 1.
    /// let learner = FiniteClass::from_table(&sample, vec![vec![1.0, 1.0, -1.0, -1.0]])
    ///     .edge_exactly(0.25);
    /// let h = learner.produce(&sample, &[0.25; 4]);
    /// assert_eq!(h.sign, 0.25);
    /// assert_eq!(h.confidence_all(&sample), vec![0.25, 0.25, -0.25, -0.25]);
    /// ```
    pub fn edge_exactly(mut self, gamma: f64) -> Self {
        check_gamma(gamma);
        self.rule = Rule::Exactly(gamma);
        self
    }

    /// Returns the number of calls to [`produce`](WeakLearner::produce) in
    /// which no member had edge at least `γ` under
    /// [`FiniteClass::weakest_with_edge`] or [`FiniteClass::edge_exactly`].
    /// It is always `0` under the default rule.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{Sample, WeakLearner};
    /// use miniboosts::weak_learners::finite_class::FiniteClass;
    ///
    /// let sample = Sample::dummy(2); // labels +1, -1
    /// // Edge 0.5 under the uniform distribution: below γ = 0.75.
    /// let learner = FiniteClass::from_table(&sample, vec![vec![1.0, 0.0]]).weakest_with_edge(0.75);
    /// let h = learner.produce(&sample, &[0.5, 0.5]);
    /// // The best member is returned, and the call is counted.
    /// assert_eq!(h.index, 0);
    /// assert_eq!(learner.assumption_violations(), 1);
    /// ```
    pub fn assumption_violations(&self) -> usize {
        self.violations.load(Ordering::Relaxed)
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
        let rule = match self.rule {
            Rule::Strongest => "largest edge".to_string(),
            Rule::Weakest(gamma) => format!("smallest edge at least {gamma}"),
            Rule::Exactly(gamma) => format!("edge exactly {gamma}"),
        };
        Some(vec![
            ("# of hypotheses", self.hypotheses.len().to_string()),
            ("Negations", self.with_negations.to_string()),
            ("Rule", rule),
        ])
    }

    /// Returns a member chosen by the learner's rule under `dist`: by
    /// default one with the largest edge; see [`FiniteClass`].
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
        // Members in tie-breaking order: by index, `+h_j` before `-h_j`.
        let members = edges.iter().enumerate().flat_map(|(j, &edge)| {
            let negation = self.with_negations.then_some((j, -1.0, -edge));
            [Some((j, 1.0, edge)), negation].into_iter().flatten()
        });
        let strongest = |members: &mut dyn Iterator<Item = (usize, f64, f64)>| {
            members.fold((0, 1.0, f64::NEG_INFINITY), |best, member| {
                if member.2 > best.2 { member } else { best }
            })
        };
        let (index, sign, _) = match self.rule {
            Rule::Strongest => strongest(&mut members.into_iter()),
            Rule::Weakest(gamma) | Rule::Exactly(gamma) => {
                let weakest = members.clone().filter(|member| member.2 >= gamma).fold(
                    None,
                    |best: Option<(usize, f64, f64)>, member| match best {
                        Some(b) if b.2 <= member.2 => Some(b),
                        _ => Some(member),
                    },
                );
                match (weakest, self.rule) {
                    (Some((j, sign, edge)), Rule::Exactly(_)) => (j, sign * gamma / edge, gamma),
                    (Some(member), _) => member,
                    (None, _) => {
                        self.violations.fetch_add(1, Ordering::Relaxed);
                        strongest(&mut members.into_iter())
                    }
                }
            }
        };
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
    /// `1.0` for `h_j`, `-1.0` for its negation. Under
    /// [`FiniteClass::edge_exactly`]`(γ)` it is instead `±γ / γ_j`, the
    /// signed scale that gives the member edge `γ`; its magnitude is at most
    /// `1`.
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

fn check_gamma(gamma: f64) {
    assert!(
        gamma > 0.0 && gamma <= 1.0,
        "the edge γ must satisfy 0 < γ <= 1, got {gamma}"
    );
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
    fn weakest_with_edge_returns_the_smallest_sufficient_edge() {
        let (sample, learner) = table_learner();
        // Uniform: edges 0, 0.5, -1, 0.5; with negations also 0, -0.5, 1,
        // -0.5. Ties at 0.5 go to the smaller index.
        let learner = learner.with_negations().weakest_with_edge(0.25);
        let h = learner.produce(&sample, &[0.25; 4]);
        assert_eq!((h.index, h.sign), (1, 1.0));
        // Only -h_2 reaches 0.75.
        let (sample, learner) = table_learner();
        let learner = learner.with_negations().weakest_with_edge(0.75);
        let h = learner.produce(&sample, &[0.25; 4]);
        assert_eq!((h.index, h.sign), (2, -1.0));
        assert_eq!(learner.assumption_violations(), 0);
    }

    #[test]
    fn edge_exactly_scales_the_member_to_edge_gamma() {
        let (sample, learner) = table_learner();
        let learner = learner.with_negations().edge_exactly(0.25);
        for d in [[0.25; 4], [0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]] {
            let h = learner.produce(&sample, &d);
            let edge: f64 = (0..4)
                .map(|i| d[i] * sample.target()[i] * h.confidence(&sample, i))
                .sum();
            // A sum of four products of values with few significant bits.
            assert!((edge - 0.25).abs() < 1e-15, "{edge}");
            assert!(h.sign.abs() <= 1.0);
        }
        assert_eq!(learner.assumption_violations(), 0);
    }

    #[test]
    fn without_a_sufficient_edge_the_best_member_is_returned_and_counted() {
        let (sample, learner) = table_learner();
        // Without negations the largest edge is 0.5 < 0.75.
        let learner = learner.edge_exactly(0.75);
        let h = learner.produce(&sample, &[0.25; 4]);
        assert_eq!((h.index, h.sign), (1, 1.0));
        learner.produce(&sample, &[0.25; 4]);
        assert_eq!(learner.assumption_violations(), 2);
    }

    #[test]
    fn adaboost_meets_its_bound_against_edge_exactly() {
        use crate::{AdaBoost, Booster};
        // Margins u_j: hypothesis j misclassifies example j only.
        let m = 8;
        let sample = Sample::dummy(m);
        let y = sample.target().to_vec();
        let columns = (0..m)
            .map(|j| (0..m).map(|i| if i == j { -y[i] } else { y[i] }).collect())
            .collect();
        let gamma = 0.1;
        let learner = FiniteClass::from_table(&sample, columns).edge_exactly(gamma);
        let rounds = 60;
        let f = AdaBoost::init(&sample).force_quit_at(rounds).run(&learner);
        assert_eq!(learner.assumption_violations(), 0);
        // Training error <= Π_t sqrt(1 - γ_t²) = (1 - γ²)^{T/2} with γ_t = γ,
        // which also holds for confidence-rated hypotheses (Schapire and
        // Singer, 1999, Sec. 3.1); the error is a multiple of 1/m.
        let error = (0..m)
            .filter(|&i| y[i] * f.confidence(&sample, i) <= 0.0)
            .count() as f64
            / m as f64;
        assert!(error <= (1.0 - gamma * gamma).powf(rounds as f64 / 2.0));
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
        for gamma in [0.0, -0.5, 1.5, f64::NAN] {
            let learner = FiniteClass::from_table(&sample, vec![vec![1.0, 1.0]]);
            let result = std::panic::catch_unwind(|| learner.weakest_with_edge(gamma));
            assert!(result.is_err());
        }
    }
}

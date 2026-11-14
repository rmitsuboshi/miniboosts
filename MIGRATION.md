# Unreleased

Decision trees train and predict faster; the trees themselves are unchanged.
`DecisionTree` stores the bin of every row of the builder's sample for each
dense feature (1 byte per value with at most 256 bins) and uses it when `produce` is called on that
sample. Tree hypotheses and `WeightedMajority` override
`Classifier::confidence_all` and `predict_all` to evaluate all rows at once, and
boosters now compute training margins with `confidence_all`; a custom
`Classifier` that overrides it must return the same values as `confidence`.

`helpers::inner_product` and `helpers::normalize` no longer run in parallel, so
their results do not depend on the number of threads. Sums that used
`inner_product` (the Frank–Wolfe solvers, `CorrectiveErlpBoost`, and
`Feature::weighted_mean` on dense features) can differ from earlier results in
the last bits.

`LpBoost::tolerance` now panics for values below `1e-6`
(`core::constants::LPBOOST_MIN_TOLERANCE`), and the Python `LPBoostClassifier`
raises `ValueError`. The LP solver cannot certify a smaller gap, so such a run
might not have stopped. The default `1e-3` is unchanged.

`Sample::split` is much faster and returns the same samples. The SVMlight
reader skips blank lines, text after `#`, and `qid:` fields, which used to
panic. `LoggerBuilder` and `Entropy` implement `Default`.

`MajorityOfFive` (Høgsgaard, Larsen, and Mathiasen, 2024) and `LarsenRitzert`
(Larsen and Ritzert, 2022) are new, in `boosters::subsample_majority` and the
prelude. They run a booster, given as a closure, on sub-samples and return the
`MajorityVote` of the results. `Sample::select_rows` returns a sample made of
given rows, with repetitions allowed.

`SparsiBoost::first_stage(booster)` replaces AdaBoostV by another booster whose
`WeightedMajority` is then sparsified, e.g. TotalBoost. `SparsiBoost` gained a
third type parameter for it, defaulting to `AdaBoostV`, so existing code still
compiles; `max_loop` and `adaboostv_tolerance` exist only with the default.
The Python `SparsiBoostClassifier` has matching `first_stage` and
`first_stage_tolerance` parameters.

`FiniteClass::weakest_with_edge(γ)` and `FiniteClass::edge_exactly(γ)` make a
finite class return a member with barely edge `γ` instead of the largest edge,
and `FiniteClass::assumption_violations` counts the calls in which no member
reached `γ`. Under `edge_exactly`, `ClassMember::sign` holds the scale `±γ/γ_j`
instead of `±1`.

`DecisionTreeBuilder::exact_splits` uses every midpoint between distinct
feature values as a threshold (`Bins::per_value` is now public), and
`DecisionTreeBuilder::leaf_confidence(LeafConfidence::Sign)` makes nodes
predict `±1`. With `max_depth(1)` and `SplitBy::Edge`, they give an exact
edge-maximizing decision stump. `LeafConfidence` is in the prelude. The
defaults, and thus existing trees, are unchanged; `DecisionTree`'s `Display`
and `info` now also report the leaf confidence.

`WeakLearnerUnion` is a new weak learner, also in the prelude. It holds other
weak learners, possibly with different hypothesis types, and returns the
candidate with the largest edge, as a `DynClassifier`. Members run in parallel
and must be `Send + Sync`. `Classifier` is now
implemented for `Arc<H>`. No existing API changed.

`NaiveAggregation` now implements `Serialize` and `Deserialize`. A
`NaiveBayesClassifier` whose training distribution gave one label no weight now
loads from JSON, where saving its infinite log odds used to produce a file that
could not be read. Its serialized form has a new `only_label` field: JSON files
from 0.5.2 still load, but files in formats that are not self-describing, such
as postcard, do not.

Every public type now implements `Debug`. The output of `Logger` and
`LoggerBuilder` omits the objective and loss functions, and `WeakLearnerUnion`
shows only the number of its members. `Prediction<T>`, `Confidence<T>`, and
`MajorityVote<H>` implement `Eq` when their type parameter does.

Builder methods that take `self` and return it are marked `#[must_use]`:
`booster.tolerance(0.1);` as a statement, which never had an effect, now gives
an `unused_must_use` warning. Write `let booster = booster.tolerance(0.1);`.

`#[inline(always)]` was removed from the functions that had it. Results are
unchanged.

Hypotheses are evaluated on all rows at once in more places, with unchanged
results. `LinearClassifier`, `RbfHypothesis`, `NaiveBayesClassifier`,
`ClassMember`, and `MajorityVote` override `Classifier::confidence_all` so that
each feature name is looked up once per call instead of once per row; with
`NaiveBayes` as the weak learner, 300 rounds of AdaBoost on 1000 examples with
20 features took 99 ms instead of 251 ms. `SmoothBoost`, `SoftBoost`,
`TotalBoost`, `ErlpBoost`, `SparsiBoost`, `FiniteClass::new`, and
`edge_minimization` now call `confidence_all` rather than `confidence` row by
row, and the tree learners look the split feature up once per node. AdaBoost,
AdaBoostV, MadaBoost, and SmoothBoost update their weights on the calling
thread: handing an `O(m)` loop to the thread pool cost more than the loop.

Decision trees train and predict faster on sparse features (SVMlight input),
with unchanged trees. `DecisionTree` now also stores the bin of every stored
entry of a sparse feature (1 byte per entry), and the tree learners and tree
hypotheses read sparse features at increasing rows without a search per row.
AdaBoost with depth-3 trees on 20000 examples with 300 features, 5% of the
entries nonzero: 88 ms per round became 13 ms.

On a sparse feature with at least 4 examples per stored entry in a node, the
tree learner adds the weights of the stored entries only and takes the weight
of the implicit zeros of each label as the label's total minus those, which
costs time proportional to the stored entries. This is a numerical shortcut,
not the original sum: the weights of the bin containing `0.0` are rounded
differently (relative error below `1e-6` up to `10⁷` examples), so a split
between near-tied candidates can change. When the zeros of a label hold less
than `1e-3` of its weight, the examples are added one by one as before. On the
data above the round takes 7 ms; with 1% nonzero entries among 2000 features,
training is 4 times faster. Dense features are unaffected.

The linear learners no longer expand a sample into dense rows.
`LinearClassifier::confidence_all` and `BoundedLinear` add up the features
column by column, and `LogisticRegression` uses the stored entries of each
example when the sample has a sparse feature. Every sum has the same terms in
the same order as before, except that products with implicit zeros are left
out, which can only turn a `0.0` into `-0.0` or back. On 20000 examples with
300 sparse features (5% nonzero), 30 rounds of AdaBoost with `BoundedLinear`
took 35 ms instead of 6.4 s; on dense samples it is about twice as fast.

`Sample::from_csc` builds a sample with sparse features from a matrix in
compressed sparse column form (`n_rows`, `indptr`, `indices`, `data`), with the
feature names of `Sample::from_dense`. It rejects unsorted or duplicate row
indices and non-finite values, and does not store entries equal to `0.0`.
`Sample::from_csr` does the same for compressed sparse row form
(`n_features`, `indptr`, `indices`, `data`) and gives the same sample.
The Python estimators accept `scipy.sparse` input through it: the native weak
learners and `RegressionTreeRegressor` keep the features sparse in Rust, and
boosters pass the sparse matrix on to their weak learner. `scipy>=1.8` is now
a declared dependency of the Python package (scikit-learn already required
scipy). Dense input behaves as before.

`GradientBoosting` and `LogitBoost` with `RegressionTree` are 10 to 27 times
faster and return the same models. `Sample` now shares its features between
clones and the copies made by `Sample::with_target`, which no longer copies
them (a method that changes the features of one copy, such as `set_target` or
`replace_names`, copies them first). `RegressionTree` stores the bin of every
value of its builder's sample (1 byte per value, or per stored entry of a
sparse feature), uses it when `produce` gets that sample or a `with_target`
copy of it, and searches the features in parallel. `RegressionTreeRegressor`
and `AdditiveModel` override `predict_all` (and `AdditiveModel`
`confidence_all`), and both boosters call it. On 20000 examples with depth-3
trees, a round of gradient boosting went from 83 ms to 7.8 ms with 50 dense
features, and from 265 ms to 20 ms with 300 sparse features (5% nonzero); a
round of LogitBoost went from 78 ms to 2.9 ms on the dense data.

`NaiveBayesClassifier` and `RbfHypothesis` predict on sparse features in time
proportional to the stored entries. Each adds its terms at `x_k = 0` once for
all rows and corrects them where a feature stores a value, so on a sample
with a sparse feature the confidences differ from earlier results in the last
bits (`confidence` and `confidence_all` still agree exactly). For
`RbfHypothesis` the squared distance then has an absolute error of order
`2⁻⁵³ ‖x_c‖²` and is clamped at 0. Samples without sparse features give the
same values as before.

The Python estimators pass feature matrices to Rust as the bytes of their
arrays instead of nested lists. On 20000 examples with 300 dense features,
`decision_function` of a native weak learner takes 32 ms instead of 187 ms,
and 20 rounds of `AdaBoostClassifier` with depth-3 trees 5.3 s instead of
12.3 s. Results are unchanged.

Memory use is lower, with unchanged results. The row-generation solvers
behind `ErlpBoost`, `TotalBoost`, and `SoftBoost` compute the margins of all
hypotheses once per solve and write them into the one constraint matrix the
solver gets; they used to hold a table of confidences, a matrix of the linear
constraints, a dense copy of the margins, and the final matrix (about 40
bytes per example and hypothesis at the peak, now about 24 for `ErlpBoost`
and 16 for `SoftBoost` and `TotalBoost`, before the solver's own copies).
`ColumnGeneration` (`LpBoost`) hands its stored columns to the solver
without copying them every round. The tree learners store bin numbers in one
byte instead of four when a feature has at most 256 bins (the default is 255).
The Python boosters no longer copy the feature matrix into Rust, which never
read it: the weak learners are fitted in Python.

# Changes in 0.5.2

Version 0.5.2 only adds API: code that compiled against 0.5.1 compiles
unchanged, except that a glob import of `miniboosts::prelude::*` brings the new
names below into scope and may clash with local names.

New boosters: `ArcGv` (Breiman, 1999), `BoostByMajority` (Freund, 1995; returns
the new `MajorityVote`), `RobustBoost` (Freund, 2009), `GradientBoosting`
(Friedman, 2001; with `GbmLoss`), and `LogitBoost` (Friedman, Hastie, and
Tibshirani, 2000). The last two fit real-valued responses with a regression weak
learner, such as the new `RegressionTreeBuilder`, and return the new
`AdditiveModel`. `Sample::with_target` returns a copy with replaced targets.
`ArcGv`, `GradientBoosting`, and `LogitBoost` require `force_quit_at`, and
`BoostByMajority` requires `gamma`.

Two weak learners are new: `FiniteClass` (with `ClassMember` and
`TableHypothesis`) returns an edge-maximizing member of a finite hypothesis
class, and `RbfClassBuilder` builds a `FiniteClass` of Gaussian radial basis
functions (`RbfHypothesis`). `BoundedLinear` and `LogisticRegression` return a
`LinearClassifier` (with `LinearOutput`), and `NaiveBayes` returns a
`NaiveBayesClassifier`. The learners are in the prelude.

`BrownBoost` (Freund, 2001) is new. It adds the dependency
[`libm`](https://crates.io/crates/libm) (rust-lang's pure-Rust port of musl's
math library) for `erf` and `erfc`, which the standard library does not
provide on stable Rust.

`SparsiBoost` (Grønlund, Larsen, and Mathiasen, ICML 2019) is new, together
with the `optimization::sparsification` module (`sparsify`, `sparsify_within`,
`sparsify_ensemble`, `discrepancy_coloring`, `max_margin_change`). Configure it
with `n_hypotheses(T)`, `tolerance(v)`, or both. It is also
exported from the prelude; a glob import of `miniboosts::prelude::*` now brings
`SparsiBoost` into scope. No existing API changed.

`SmoothBoost::reached_round_budget` reports whether training stopped at the
round budget without the paper's stopping test firing; the margin guarantee
does not apply in that case. SmoothBoost now panics if a weak hypothesis
returns a non-finite confidence instead of continuing with it.

# Changes in 0.5.1

Version 0.5.1 adds rustdoc with runnable examples for every public item and
fixes the following. Code that compiled against 0.5.0 compiles unchanged unless
it used the removed deformed-entropy items below.

`WeightedMajority::from_slices` now drops only exactly-zero weights and keeps
negative ones. Previously it also dropped negative weights, so an AdaBoost
hypothesis with a negative edge was used in the distribution update but omitted
from the final classifier, and a perfectly wrong hypothesis made training panic.
With symmetric hypothesis classes (such as decision trees), weak learners return
nonnegative edges and results are unchanged.

The unused deformed-entropy (Tsallis) ERLPBoost components were removed:
`DeformedEntropyRegularizedMaxEdge`, `objective_function::DeformedErlpSoftMarginObjective`,
the `deformed_*` and `gradient_of_conjugate_deformed_entropy` helpers, and
`checkers::deformation_parameter`. No booster used them, and their
normalization bisection could fail to terminate.

`WeightedMajority::from_slices` returns the empty ensemble (confidence `0.0`,
so every prediction is `+1`) when no nonzero weight remains, instead of
panicking. As a result, `CurrentHypothesis::current_hypothesis` no longer panics
before the first round (SoftBoost and TotalBoost skip their LP when there are no
hypotheses), and `AdaBoost`/`AdaBoostV`/`MadaBoost` with `force_quit_at(0)` return
the empty ensemble.

`SampleReader::default()` no longer requires the path and column-name types to
implement `Default`, so borrowed paths such as `.file(&path_buf)` compile.

`CorrectiveErlpBoost` and `MlpBoost` no longer require `PartialEq` (and
`CorrectiveErlpBoost` no longer requires `Debug`) on the hypothesis type; the
bounds were unused. `ErlpBoost::max_loop` is now public. `SoftBoost::max_loop`
and `CorrectiveErlpBoost::max_loop` take `&self`, and
`CorrectiveErlpBoost::max_loop` returns the actual budget before training
instead of `1`. `DEFAULT_MAX_DEPTH` and `DEFAULT_NBIN` are exported from
`miniboosts::weak_learners::decision_tree`.

# Migrating to 0.5.0

Version 0.5.0 ships as a single `miniboosts` crate. Set your dependency to
`miniboosts = "0.5.0"` and use the documentation for that version.
See [RELEASING.md](RELEASING.md) for the publishing procedure.

The development workspace's internal crates have become modules in `src/`.
Root exports and `miniboosts::prelude` are preserved. Replace direct imports of
`miniboosts_core` with `miniboosts::core`, and imports of internal algorithm
crates with `miniboosts::boosters::<algorithm>` (or the root type re-exports).
Use `miniboosts::weak_learners::decision_tree`, `miniboosts::hypotheses`,
`miniboosts::optimization`, and `miniboosts::logging` for their corresponding
support APIs. Only `miniboosts` is published; `miniboosts-example` stays private.

The following notes describe the API changes from the earlier published releases
and the numerical changes developed in the former workspace layout.

| Previous usage | Workspace usage |
| --- | --- |
| `LPBoost`, `ERLPBoost`, `MLPBoost`, `CERLPBoost` | `LpBoost`, `ErlpBoost`, `MlpBoost`, `CorrectiveErlpBoost` |
| `GraphSepBoost` | `GraphSeparationBoosting` |
| `SampleReader::new()` | `SampleReader::default()` |
| `.criterion(Criterion::Entropy)` | `.split_by(SplitBy::Entropy)` |
| Logger's `SoftMarginObjective` | `LoggingSoftMarginObjective` |
| `Research` for snapshots | `CurrentHypothesis` |
| `miniboosts::booster` / `weak_learner` / `hypothesis` modules | Root re-exports or the member crates |
| `gurobi` feature | Removed; current solvers use Clarabel |

The prelude exports common boosters and traits, `SampleReader`,
`DecisionTreeBuilder`, and `SplitBy`. Import `Sample`, logger types, and objective
adapters explicitly from `miniboosts`.

The uncompiled GBM and regression-tree sources were removed from the repository;
they remain available in the git history (commit `4b8be8c`). Gaussian naive
Bayes, neural networks, the worst-case LPBoost learner, and the previous
cross-validation helper are not exposed by this workspace. There is no automatic
replacement: retain a compatible older checkout if required, or implement the
needed learner against the shared traits. For manual cross-validation,
`Sample::split(indices, start, end)` takes a full row permutation and a half-open
test interval, returning `(train, test)`.

Algorithm tolerances are not interchangeable classification-error guarantees.
Compare documented objectives and stopping conditions when migrating experiments.
Solver and numerical changes require revalidation; this refactor makes no claim
of identical models or improved runtime against the older implementation.

`CurrentHypothesis` lives in `miniboosts::core`; the logging and root
re-exports remain available. `BlendedPairwise` is rejected by the MLPBoost and
CorrectiveERLPBoost setters until their coefficient-space implementation is available.
SoftBoost now solves the entropy projection using exponential cones instead of SQP.
SmoothBoost defaults to a weak-learning advantage of 0.25 and starts cumulative
margins at zero. These corrections can change results relative to this branch's
earlier implementation.

Normalizing example weights now uses shared, max-shifted softmax with compensated
summation. AdaBoost and AdaBoostV retain log weights between rounds; probabilities
that underflow to zero can recover later. Coefficients and perfect-classification
checks use log-domain mass near extreme edges, so rounding an edge to one no longer
ends training early. These changes can affect long or numerically extreme runs.
The capped-simplex projection retains its sorted active-set algorithm.

External dependencies are declared once in the root package. The checked-in
Cargo.lock records tested versions; use `--locked` when reproducing experiments.
Clarabel 0.11 constructors are validated before solving. The numerical regression
tests retain their existing tolerances across this solver upgrade.

CorrectiveERLPBoost now applies its Frank-Wolfe update from zero coefficients on
round one, following Shalev-Shwartz and Singer (2008),
[Figure 1](https://home.ttic.edu/~shai/papers/ShalevSi08.pdf). Previously it forced
the first coefficient to one and left the Classic step counter at zero. Returned
models and logger snapshots now preserve coefficients with total mass at most one,
instead of rescaling them to unit mass; confidence values and margin objectives
therefore match training. A zero-edge initial oracle returns a zero model.
The experiment runner uses the booster's default ShortStep rule; Classic remains
available explicitly, but its fixed schedule need not improve the objective each
round and can converge slowly for small tolerances. Regenerate earlier CERLPBoost
CSVs before comparing runs.

Booster refinements after the main-branch review:

- AdaBoost, MadaBoost, LPBoost, ERLPBoost, CorrectiveERLPBoost and MLPBoost
  reject non-finite or nonpositive tolerances. The entropy-regularized methods
  also reject tolerances whose half underflows or whose derived regularization
  parameter is not representable. Parameter meanings are unchanged.
- AdaBoost's calculated round budget now uses the documented ceiling, with
  at least one round for a nonempty sample. An explicit `force_quit_at` still
  overrides that budget. The existing training-data validator still requires
  both label classes; this does not enable singleton training.
- MLPBoost's Classic schedule counts the initial atom as `w_1`, so the next
  FW candidate uses `2/3`, then `1/2`, following
  [Algorithm 1 and Theorem 3](https://arxiv.org/abs/2209.10831).
  Previously that first candidate used a full step. Classic remains available explicitly.
- CorrectiveERLPBoost caches the current training margin vector in O(m) space.
  MLPBoost caches one margin column per accepted hypothesis in O(mT) additional
  space. Weak hypotheses must return stable predictions for a fixed sample;
  historical hypotheses are no longer repeatedly evaluated during these updates.
  Summation order changes can affect rounding near stopping thresholds.
- GraphSeparationBoosting stores prediction-history groups instead of all
  opposite-label pairs. Its graph maintenance takes expected O(m) time per
  round and O(m) space, excluding hypothesis prediction and final aggregation.
  Its degree distribution and existing one-round stall rule are unchanged for
  valid binary labels. Labels other than -1 and +1 now fail before training.
- ERLPBoost's Shannon-entropy subproblem now uses an exponential-cone
  formulation of the same KL-regularized problem, following
  [Warmuth et al., Algorithm 1](https://mwarmuth.bitbucket.io/pubs/C83.pdf).
  The earlier SQP descent safeguard could panic on the ordinary breast-cancer
  example; quadratic-model accuracy did not certify entropy-objective descent.
  The new path checks the repaired primal objective against an independent
  capped-entropy dual lower bound (gap at most `1e-7`) and uses that lower bound
  for the outer stopping test. This is a floating-point numerical certificate,
  not an interval-arithmetic proof. Extremely small outer tolerances may require
  a tighter inner solver policy. Solver failures still panic and clear old
  results; they are not accepted as convergence. Generic non-Shannon objectives
  keep their SQP path. `RowGenerationObjective::entropy_parameter()` defaults
  to `None`, so existing custom objective implementations remain compatible.


These are computational and numerical changes, not a measured end-to-end
speedup claim. Regenerate experiment results when comparing with earlier commits.

MLPBoost now defaults to `FwUpdateRule::LineSearch` instead of `ShortStep`.
The example runner also uses this default instead of explicitly selecting Classic.
Classic and ShortStep remain opt-in. Line search numerically minimizes the same
regularized objective along the FW segment; see the discussion following Theorem 3
in [Boosting as Frank-Wolfe](https://arxiv.org/abs/2209.10831).
This changes learned trajectories and stopping times, so regenerate MLPBoost
experiment outputs. CorrectiveERLPBoost's default remains ShortStep.

Final soft-margin LP postprocessing now accepts `AlmostSolved` only after explicit
finite-value, primal/dual residual, absolute duality-gap, and simplex checks
(at `1e-7`). `Solved` receives the same checks. Tiny negative weights are
clamped and the weights normalized; failed or insufficiently accurate solves
still raise an error. This fixes a postprocessing panic exposed by the
MLPBoost line-search run.

Generic SQP treats a small objective increase as numerical stationarity only
when the current point's quadratic-model gap is within the measured QP gap
(with a roundoff floor). This replaces a fixed coordinate-distance cutoff that
could panic near a simplex boundary despite an accurate deformed-entropy solve.
It checks model accuracy, not an independent original-objective certificate;
the Shannon-entropy path retains its separate primal/dual certificate.

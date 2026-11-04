# Unreleased

`WeakLearnerUnion` is a new weak learner, also in the prelude. It holds other
weak learners, possibly with different hypothesis types, and returns the
candidate with the largest edge, as a `DynClassifier`. Members run in parallel
and must be `Send + Sync`. `Classifier` is now
implemented for `Arc<H>`. No existing API changed.

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

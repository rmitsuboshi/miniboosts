# Migrating to this workspace branch

This is an unpublished development layout, not a drop-in crates.io release.
Use a local `path` dependency and generate documentation from this checkout.
Package names, versions, metadata, and publishing order still need a release plan.

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

GBM and regression-tree sources remain disabled in Cargo.toml. Gaussian naive
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

`CurrentHypothesis` now lives in `miniboosts_core`; the `logging` and root
re-exports remain available. Simple booster crates no longer require the logger
or its optimizer dependencies. `BlendedPairwise` is rejected by the MLPBoost and
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

External dependencies are declared once in the root workspace. The checked-in
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

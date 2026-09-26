# Run and plot experiments

The experiment runner uses the MiniBoosts library in this checkout.
Input CSVs must have a header, numeric features, and a `class` column containing
`-1` and `1`. Training and test files must have the same feature columns in the same
order. Prepare a fixed train/test split before running; neither script splits
or shuffles data.

`train.csv` and `test.csv` below are placeholders for existing input files;
paths are relative to your current directory. If the breast-cancer CSVs are
present locally, run from the repository root:

```sh
bash img/run_all.sh img/csv/breast-cancer-train.csv img/csv/breast-cancer-test.csv img/csv/breast-cancer-results
python3 img/DepictExample.py --input-dir img/csv/breast-cancer-results --output-dir img/plots/breast-cancer
```

These dataset files are ignored by Git and are not bundled with a fresh clone.

From the repository root, run one algorithm:

```sh
cargo run --release --locked -p miniboosts-example -- lpboost train.csv test.csv img/csv/lpboost.csv
cargo run --release --locked -p miniboosts-example -- --list
cargo run --release --locked -p miniboosts-example -- --help
```

Run all 11 algorithms sequentially on the same inputs:

```sh
bash img/run_all.sh train.csv test.csv
# Optional output directory and settings:
bash img/run_all.sh train.csv test.csv img/csv/experiment-1 --max-iterations 200 --time-limit-ms 60000 --tolerance 0.001 --nu 1
```

The batch script stops on failure and saves each algorithm's CSV and console
log, plus `environment.txt`, `data-sha256.txt`, and a copy of `Cargo.lock`.
It records the command, Git revision and working-tree status, Rust version, and OS and architecture.
Python 3 is needed for input hashes. Use a fresh output directory per experiment:
files with matching names are overwritten. For published timings, also record
the CPU model, available cores and memory, and load on the machine; retain the patch
if the working tree has uncommitted changes.

Runs stop at 500 boosting steps by default. Set `--max-iterations 200` to use
a smaller budget. Training stops earlier if the algorithm converges or reaches
the time limit. The last step is written to CSV before finalization.

All runs use decision stumps (depth 1) with entropy-based splits, a default 60,000 ms time limit,
and a default `nu = max(1, 0.01 * training rows)`. The default tolerance is 0.001;
for SmoothBoost it sets `kappa`, with `gamma = 0.006` (override with `--gamma`).
That gamma is an assumed weak-learner advantage, not a guarantee inferred from
the data. GraphSeparationBoosting has no tolerance parameter; TotalBoost uses a hard
margin. `nu` configures the algorithms that support capping and the common
logged soft-margin metric. MLPBoost uses classic Frank–Wolfe updates.
Corrective ERLPBoost uses its default short-step updates. Its coefficient sum
can be below one; logged margins retain that scale rather than renormalizing
the model. Equal tolerances do not imply equal accuracy or stopping
criteria across algorithms.

CSV columns are `ObjectiveValue`, `TrainLoss`, `TestLoss`, and `Time`.
The objective is a common soft-margin evaluation metric, not necessarily the
algorithm's training objective. Losses are 0/1 error rates. Time is cumulative
boosting time in milliseconds, excluding setup, finalization, evaluation, and
I/O. Limits are checked between steps, so a single step can exceed the budget.
Rows describe the logger's intermediate models; final postprocessing may
produce a different model.

## Plot CSVs

Install Matplotlib in your Python environment if needed:

```sh
python3 -m pip install matplotlib
```

Generate the figures:

```sh
python3 img/DepictExample.py --input-dir img/csv --output-dir img/plots
```

This replaces `DepictExample.ipynb`. It writes `training-loss.png`,
`test-loss.png`, and (when relevant runs exist) `soft-margin.png`. Each figure uses solid lines and vertically stacked panels for iteration
count and elapsed boosting time, so it remains readable at README widths. Missing
algorithms are skipped; malformed or empty CSVs are rejected. Zero-millisecond measurements are retained on a symmetric-log time axis
(linear below 0.01 seconds, logarithmic above). Use `--title "Dataset name"` and `--every 3` for optional labeling and
subsampling. The soft-margin plot compares SoftBoost, LPBoost, ERLPBoost,
MLPBoost, and Corrective ERLPBoost; compare only runs with matching `nu`.
Each metric also has a `*-by-algorithm.png` figure with vertically stacked panels and
independent axis scales to reveal small changes. Use the overview figures for
comparisons on common scales. Curves are not smoothed; all rows are plotted by
default. Generated CSVs and default plot outputs are ignored by Git.

SoftBoost and TotalBoost stop early with a warning if the entropy solver cannot
meet its requested accuracy (including `AlmostSolved`). They return a model
fitted to the collected hypotheses; this does not certify convergence or
linear separability. Library users can inspect `numerical_stop_reason()` after
training. Final model fitting still requires a successful LP solve.

# MiniBoosts

MiniBoosts is a Rust library for implementing and comparing boosting algorithms.
Its algorithms share the `Booster`, `WeakLearner`, and `Classifier` traits, so you
can use the same data and weak learner across experiments.

This workspace version has not been published to crates.io. Use a local checkout;
the documentation on docs.rs may refer to an older API. Rust 1.85 or newer is
required (edition 2024).

## Getting started

Clone this repository, then point your application's dependency at the checkout:

```toml
[dependencies]
miniboosts = { path = "/path/to/miniboosts" }
```

The optimization routines use Clarabel for linear, quadratic, and conic
programming. No Gurobi installation or license is required.

Add the following to `src/main.rs` to train a binary classifier on a small
in-memory dataset:

```rust
use std::io::BufReader;
use miniboosts::{Sample, prelude::*};

fn main() {
    let csv = b"x,class\n-2,-1\n-1,-1\n1,1\n2,1\n";
    let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
        .unwrap()
        .set_target("class");
    let learner = DecisionTreeBuilder::new(&sample)
        .max_depth(2)
        .split_by(SplitBy::Entropy)
        .build();
    let mut booster = AdaBoost::init(&sample)
        .tolerance(0.1)
        .force_quit_at(20);
    let classifier = booster.run(&learner);
    assert_eq!(classifier.predict_all(&sample), vec![-1, -1, 1, 1]);
}
```

For CSV or SVMlight files, use `SampleReader::default()`. Binary classification
labels must be `-1` and `1`. The assertion checks predictions on this training
sample; use a separate test set to evaluate performance on unseen data.

## Algorithms

Import these types from `miniboosts` or `miniboosts::prelude`:

| Type | Source |
| --- | --- |
| `AdaBoost` | [Freund and Schapire][adaboost] |
| `AdaBoostV` | [Rätsch and Warmuth][adaboostv] |
| `MadaBoost` | [Domingo and Watanabe][madaboost] |
| `LpBoost` | [Demiriz, Bennett, and Shawe-Taylor][lpboost] |
| `ErlpBoost` | [Warmuth, Glocer, and Vishwanathan][erlpboost] |
| `CorrectiveErlpBoost` | [Shalev-Shwartz and Singer][cerlpboost] |
| `MlpBoost` | [Mitsuboshi, Hatano, and Takimoto][mlpboost] |
| `SmoothBoost` | [Servedio][smoothboost] |
| `SoftBoost` | [Warmuth, Glocer, and Rätsch][softboost] |
| `TotalBoost` | [Warmuth, Liao, and Rätsch][totalboost] |
| `GraphSeparationBoosting` | [Alon, Gonen, Hazan, and Moran][graphsepboost] |

The built-in weak learner is a [decision tree][decisiontree]. Implement
`WeakLearner` to use your own learner. The gradient boosting machine (GBM) and
regression tree implementations are not included in the active workspace.
See the [migration notes](MIGRATION.md) for removed APIs.

## Parameters and comparisons

`tolerance` is algorithm-specific. AdaBoost and MadaBoost use it to set iteration
budgets; it does not directly test measured classification error. AdaBoostV uses
it in its margin update and iteration budget. LPBoost, ERLPBoost, corrective
ERLPBoost, MLPBoost, and SoftBoost use optimization stopping criteria; TotalBoost
wraps SoftBoost with `nu = 1`. An optimization gap is not a classification-error
bound. Theoretical guarantees require the corresponding paper's weak-learning
or optimization-oracle assumptions and accurate subproblem solutions; using a
decision tree does not by itself ensure that these assumptions hold.

For soft-margin methods, `nu` caps example weights at `1 / nu`, with
`1 <= nu <= number_of_examples`. It does not guarantee a particular number of
misclassified examples. SmoothBoost requires a weak-learning advantage `gamma` and a target
error parameter `kappa`; its guarantee depends on that advantage holding on each
training distribution.

For fair comparisons, use the same data splits, preprocessing, weak learner,
and evaluation metrics. State each algorithm's stopping rule and use a common
time budget when comparing learning curves over time. Record seeds, parameters, commit,
dependencies, hardware, and thread count. Use release builds for timing and
report what was measured. Equal numeric tolerances need not imply equal accuracy.
Keep test data out of training and parameter selection.

## Run experiments and plot results

The example runner can run one algorithm or all 11 and save learning curves as
CSV files. The plotting script generates training-loss, test-loss, and
soft-margin figures. See the [experiment guide](img/README.md) for input formats,
commands, parameters, and timing details.

To list the available algorithms:

```sh
cargo run --release --locked -p miniboosts-example -- --list
```

## Development

- `miniboosts-core`: data, traits, and shared utilities.
- `boosters/`, `weak-learners/`: algorithm crates.
- `optimization/`, `hypotheses/`, `logging/`: solvers, aggregation, and measurement.
- `src/`: public exports and prelude.
- `img/example/`: command-line experiment runner.

```sh
cargo test --workspace --locked
cargo check --workspace --locked
cargo fmt --all -- --check
cargo doc --workspace --locked --no-deps --open
```

See [logging usage](logging/src/README.md) for CSV learning curves and timing
details. `Cargo.toml` lists the active workspace members and dependencies.
Dependency versions are shared through `[workspace.dependencies]`. Keep `Cargo.lock`
with experiment results. CI uses `--locked` to test those versions on stable Rust
and the minimum supported Rust version (1.85). Dependency updates should be
separate from numerical changes, with solver accuracy and convergence retested.

[adaboost]: https://www.sciencedirect.com/science/article/pii/S002200009791504X?via%3Dihub
[adaboostv]: http://jmlr.org/papers/v6/ratsch05a.html
[cerlpboost]: https://link.springer.com/article/10.1007/s10994-010-5173-z
[decisiontree]: https://www.amazon.co.jp/-/en/Leo-Breiman/dp/0412048418
[erlpboost]: https://www.stat.purdue.edu/~vishy/papers/WarGloVis08.pdf
[graphsepboost]: https://theoretics.episciences.org/10757
[lpboost]: https://link.springer.com/content/pdf/10.1023/A:1012470815092.pdf
[mlpboost]: https://arxiv.org/abs/2209.10831
[madaboost]: https://www.learningtheory.org/colt2000/papers/DomingoWatanabe.pdf
[smoothboost]: https://link.springer.com/chapter/10.1007/3-540-44581-1_31
[softboost]: https://proceedings.neurips.cc/paper/2007/file/cfbce4c1d7c425baf21d6b6f2babe6be-Paper.pdf
[totalboost]: https://dl.acm.org/doi/10.1145/1143844.1143970

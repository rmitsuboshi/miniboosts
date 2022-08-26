# Lycaon
A collection of boosting algorithms written in Rust.
This library provides some boosting algorithms for binary classification.

You can implement your original boosting algorithm by implementing the `Booster` trait.
You can also implement your original base learning algorithm by implementing the `BaseLearner` trait.


In this code, I use the [Gurobi optimizer](https://www.gurobi.com).
You need to acquire the license to use TotalBoost, LPBoost, ERLPBoost, and SoftBoost.
I'm planning to write code that solves linear and quadratic programming.


## Implemented:
You can combine the following boosters and base learners arbitrarily.

- Boosters
    - Empirical Risk Minimization
        * [AdaBoost](https://www.sciencedirect.com/science/article/pii/S002200009791504X?via%3Dihub)
    - Hard Margin Maximization
        * [AdaBoostV](http://jmlr.org/papers/v6/ratsch05a.html)
        * [TotalBoost](https://dl.acm.org/doi/10.1145/1143844.1143970)
    - Soft Margin Maximization
        * [LPBoost](https://link.springer.com/content/pdf/10.1023/A:1012470815092.pdf)
        * [ERLPBoost](https://www.stat.purdue.edu/~vishy/papers/WarGloVis08.pdf)
        * [Corrective ERLPBoost](https://core.ac.uk/download/pdf/207934763.pdf)
        * [SoftBoost](https://proceedings.neurips.cc/paper/2007/file/cfbce4c1d7c425baf21d6b6f2babe6be-Paper.pdf)


- Base Learner
    - Decision tree (`DTree`)

## What I will implement:

- Booster
    - LogitBoost
    - [AnyBoost](https://www.researchgate.net/publication/243689632_Functional_gradient_techniques_for_combining_hypotheses)
    - [GradientBoost](https://www.jstor.org/stable/2699986)
    - [SparsiBoost](http://proceedings.mlr.press/v97/mathiasen19a/mathiasen19a.pdf)


- Base Learner
    - Bag of words
    - TF-IDF
    - Two Layer Neural Networks
    - Regression Tree


- Others
    - Parallelization
    - LP/QP solver


I'm also planning to implement the other booster/base learner in the future.


## How to use


You need to write the following line to `cargo.toml`.

```TOML
lycaon = { git = "https://github.com/rmitsuboshi/lycaon" }
```


Here is a sample code:

```rust
use polars::prelude::*;

use lycaon::{
    Booster,
    AdaBoost, // You can use other boosters enumerated above.
    DTree,    // Decision Tree
    Criterion,
    Classifier,
};


fn main() {
    // Set file name
    let file = "/path/to/input/data.csv";

    // Read a CSV file
    let mut data = CsvReader::from_path(file)
        .unwrap()
        .has_header(true)
        .finish()
        .unwrap();


    // Pick the target class. Each element is 1 or -1.
    let target: Series = data.drop_in_place(&"class").unwrap();


    // Initialize Booster
    let mut booster = AdaBoost::init(&sample);


    // Initialize Base Learner
    // For decision tree, the default `max_depth` is `None` so that 
    // The tree grows extremely large.
    let base_learner = DTree::init(&data)
        .max_depth(2)
        .criterion(Criterion::Edge); // Choose the split rule that maximizes the edge.


    // Set tolerance parameter
    let tolerance = 0.01;


    // Run boosting algorithm
    // Each booster returns a combined hypothesis.
    let f = booster.run(&base_learner, &data, &target, tolerance);


    // These assertion may fail if the dataset are not linearly separable.
    let predictions = f.predict_all(&data);


    // You can predict the `i`th instance.
    let i = 0_usize;
    let prediction = f.predict(&data, i);
}
```


If you use soft margin maximizing boosting, initialize booster like this:
```rust
let (m, _) = df.shape();
let capping_param = m as f64 * 0.2;
let lpboost = LPBoost::init(&sample)
    .capping(capping_param);
```

Note that the capping parameter satisfies `1 <= capping_param && capping_param <= m`.

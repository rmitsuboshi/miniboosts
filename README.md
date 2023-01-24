# MiniBoosts
**A collection of boosting algorithms written in Rust 🦀.**


This library uses [Gurobi optimizer](https://www.gurobi.com), 
so you must acquire a license to use this library. 
**Note** that you need to put `gurobi.lic` in your home directory; 
otherwise, the compile fails. 
See [this repository](https://github.com/ykrist/rust-grb) for details.


## Features
Currently, I implemented the following Boosters and Weak Learners.
You can combine them arbitrarily.


### Classification


- Boosters
    * [AdaBoost](https://www.sciencedirect.com/science/article/pii/S002200009791504X?via%3Dihub) by Freund and Schapire, 1997
    * [AdaBoostV](http://jmlr.org/papers/v6/ratsch05a.html) by Rätsch and Warmuth, 2005
    * [TotalBoost](https://dl.acm.org/doi/10.1145/1143844.1143970) by Warmuth, Liao, and Rätsch, 2006
    * [LPBoost](https://link.springer.com/content/pdf/10.1023/A:1012470815092.pdf) by Demiriz, Bennett, and Shawe-Taylor, 2002
    * [SmoothBoost](https://link.springer.com/chapter/10.1007/3-540-44581-1_31) by Rocco A. Servedio, 2003
    * [SoftBoost](https://proceedings.neurips.cc/paper/2007/file/cfbce4c1d7c425baf21d6b6f2babe6be-Paper.pdf) by Warmuth, Glocer, and Rätsch, 2007
    * [ERLPBoost](https://www.stat.purdue.edu/~vishy/papers/WarGloVis08.pdf) by Warmuth and Glocer, and Vishwanathan, 2008
    * [CERLPBoost](https://link.springer.com/article/10.1007/s10994-010-5173-z) (The Corrective ERLPBoost) by Shalev-Shwartz and Singer, 2010
    * [MLPBoost](https://arxiv.org/abs/2209.10831) by Mitsuboshi, Hatano, and Takimoto, 2022


- Weak Learners
    - [DTree](https://www.amazon.co.jp/-/en/Leo-Breiman/dp/0412048418) (Decision Tree)
    - GaussianNB (Naive Bayes), **beta version**
    - WLUnion, a union of multiple weak learners.


### Regression
- Weak Learner
    - [RTree](https://www.amazon.co.jp/-/en/Leo-Breiman/dp/0412048418) (Regression Tree)

## Future work

- Booster
    - [AnyBoost](https://www.researchgate.net/publication/243689632_Functional_gradient_techniques_for_combining_hypotheses)
    - [GradientBoost](https://www.jstor.org/stable/2699986)
    - [SparsiBoost](http://proceedings.mlr.press/v97/mathiasen19a/mathiasen19a.pdf)


- Weak Learner
    - Bag of words
    - TF-IDF
    - Two-Layer Neural Network
    - [RBF-Net](https://link.springer.com/content/pdf/10.1023/A:1007618119488.pdf)


- Others
    - Parallelization
    - LP/QP solver (This work allows you to use this library without a license).


## How to use
This library uses 
the `DataFrame` of [`polars`](https://github.com/pola-rs/polars) crate, 
so that you need to import `polars`.

You need to write the following line to `Cargo.toml`.

```TOML
miniboosts = { git = "https://github.com/rmitsuboshi/miniboosts" }
```


Here is a sample code:

```rust
use polars::prelude::*;
use miniboosts::prelude::*;


fn main() {
    // Set file name
    let file = "/path/to/input/data.csv";

    // Read a CSV file
    // Note that each feature of `data`, except the target column,
    // must be the `f64` type with no missing values.
    let mut data = CsvReader::from_path(file)
        .unwrap()
        .has_header(true)
        .finish()
        .unwrap();


    // Pick the target class. Each element is 1 or -1 of type `i64`.
    let target: Series = data.drop_in_place(&"class").unwrap();


    // Set tolerance parameter
    let tol: f64 = 0.01;


    // Initialize Booster
    let mut booster = AdaBoost::init(&data, &target)
        .tolerance(tol); // Set the tolerance parameter.


    // Initialize Weak Learner
    // For decision tree, the default `max_depth` is `None` so that 
    // The tree grows extremely large.
    let weak_learner = DTree::init(&data, &target)
        .max_depth(2) // Specify the max depth (default is not specified)
        .criterion(Criterion::Edge); // Choose the split rule that maximizes the edge.


    // Run boosting algorithm
    // Each booster returns a combined hypothesis.
    let f = booster.run(&weak_learner);


    // Get the batch prediction for all examples in `data`.
    let predictions = f.predict_all(&data);


    // You can predict the `i`th instance.
    let i = 0_usize;
    let prediction = f.predict(&data, i);
}
```


If you use boosting for soft margin optimization, 
initialize booster like this:
```rust
let m = df.shape().0;
let nu = m as f64 * 0.2;
let lpboost = LPBoost::init(&sample)
    .tolerance(tol)
    .nu(nu); // Setting the capping parameter.
```

Note that the capping parameter satisfies `1 <= nu && nu <= m`.

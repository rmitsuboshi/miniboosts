# Lycaon
A collection of boosting algorithms written in Rust.
This library provides some boosting algorithms for the binary classification.

You can implement your original boosting algorithm by implementing the `Booster` trait.
You can also implement your original base learning algorithm by implementing the `BaseLearner` trait.


In this code, I use the [Gurobi optimizer](https://www.gurobi.com).
You need to acquire the license if you want to use TotalBoost, LPBoost, ERLPBoost, and SoftBoost.
I'm planning to write code that solves linear and quadratic programming.

## Implemented:
You can combine the following boosting algorithms and base learner arbitrary.

- Boosters
    - Empirical Risk Minimization
        - [AdaBoost](https://www.sciencedirect.com/science/article/pii/S002200009791504X?via%3Dihub)
    - Hard Margin Maximization
        - [AdaBoostV](http://jmlr.org/papers/v6/ratsch05a.html)
        - [TotalBoost](https://dl.acm.org/doi/10.1145/1143844.1143970)
    - Soft Margin Maximization
        - [LPBoost](https://link.springer.com/content/pdf/10.1023/A:1012470815092.pdf)
        - [ERLPBoost](https://www.stat.purdue.edu/~vishy/papers/WarGloVis08.pdf)
        - [Corrective ERLPBoost](https://core.ac.uk/download/pdf/207934763.pdf)
        - [SoftBoost](https://proceedings.neurips.cc/paper/2007/file/cfbce4c1d7c425baf21d6b6f2babe6be-Paper.pdf)


- Base Learner
    - Decision stump class

## What I will implement:

- Booster
    - LogitBoost

- Base Learner
  - Decision tree
  - Bag of words
  - Two Layer Neural Network


I'm also planning to implement the other booster/base learner in the future.


## How to use

Here is a sample code:

```rust

// `run()` and is the method of Booster trait
use lycaon::Booster;

// `predict(&data)` is the method of Classifier trait
use lycaon::Classifier;

// In this example, we use AdaBoost as the booster
use lycaon::AdaBoost;

// In this example,
// we use Decision stump (Decision Tree of depth 1) as the base learner
use lycaon::DStump;

// This function reads a file with LIBSVM format
use boost::data_reader::read_csv;

fn main() {
    // Set file name
    let file = "/path/to/input/data.csv";

    // Read file
    let sample = read_csv(file).unwrap();

    // Initialize Booster
    let mut adaboost = AdaBoost::init(&sample);

    // Initialize Base Learner
    let dstump = DStump::init(&sample);

    // Set tolerance parameter
    let tolerance = 0.1;

    // Run boosting algorithm
    let f = adaboost.run(&dstump, &sample, tolerance);


    // These assertion may fail if the dataset are not linearly separable.
    for (dat, lab) in sample.iter() {
        // Check the predictions
        assert_eq!(f.predict(dat), *lab);
    }
}
```


If you use soft margin maximizing boosting, initialize booster like this:
```rust
let m = sample.len() as f64;
let capping_param = m * 0.2;
let lpboost = LPBoost::init(&sample).capping(capping_param);
```

Note that the capping parameter satisfies `1 <= capping_param && capping_param <= m`.

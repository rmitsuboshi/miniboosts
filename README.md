# Lycaon
A collection of boosting algorithms written in Rust


In this code, I use the [Gurobi optimizer](https://www.gurobi.com).
You need to acquire the license if you want to use TotalBoost, LPBoost, ERLPBoost, and SoftBoost.
I'm planning to write code that solves linear and quadratic programming.

## What I wrote:

- Boosters
    - Empirical Risk Minimization
        - [AdaBoost](https://www.sciencedirect.com/science/article/pii/S002200009791504X?via%3Dihub)
        - [AdaBoostV](http://jmlr.org/papers/v6/ratsch05a.html)
    - Hard Margin Maximization
        - [TotalBoost](https://dl.acm.org/doi/10.1145/1143844.1143970)
    - Soft Margin Maximization
        - [LPBoost](https://link.springer.com/content/pdf/10.1023/A:1012470815092.pdf)
        - [ERLPBoost](https://www.stat.purdue.edu/~vishy/papers/WarGloVis08.pdf)
        - [SoftBoost](https://proceedings.neurips.cc/paper/2007/file/cfbce4c1d7c425baf21d6b6f2babe6be-Paper.pdf)


- Base Learner
    - Decision stump class

## What I will write:

- Booster
    - LogitBoost
    - [Corrective ERLPBoost](https://core.ac.uk/download/pdf/207934763.pdf)

- Base Learner
  - Decision tree
  - Bag of words


I'm also planning to implement the other boosting algorithms in the future.


## How to use

Here is a sample code:

```rust

extern crate boost;

// `run()` and `predict()` are the methods of Booster trait
use boost::booster::Booster;
use boost::booster::AdaBoost;
use boost::base_learner::DStump;

// This function reads a file with LIBSVM format
use boost::data_reader::read_csv;

fn main() {
    // Set file name
    let file = "/path/to/input/data";

    // Read file
    let sample = read_csv(file).unwrap();

    // Initialize Booster
    let mut adaboost = AdaBoost::init(&sample);

    // Initialize Base Learner
    let dstump = DStump::init(&sample);

    // Set accuracy parameter
    let accuracy = 0.1;

    // Run boosting algorithm
    adaboost.run(dstump, &sample, accuracy);


    for example in sample.iter() {
        let data  = &example.data;
        let label =  example.label;

        // Check the predictions
        assert_eq!(adaboost.predict(data), label);
    }
}
```


If you use soft margin maximizing boosting, initialize booster like this:
```rust
let m = sample.len() as f64;
let capping_param = m * 0.2;
let lpboost = LPBoost::init(&sample).capping(capping_param)
```

Note that the capping parameter satisfies `0 <= capping_param && capping_param <= m`.

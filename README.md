# boost
A collection of boosting algorithms written in Rust

## What I wrote:

- Boosters
    - [AdaBoost](https://www.sciencedirect.com/science/article/pii/S002200009791504X?via%3Dihub)
    - [AdaBoostV](http://jmlr.org/papers/v6/ratsch05a.html)
    - [LPBoost](https://link.springer.com/content/pdf/10.1023/A:1012470815092.pdf)
    - [ERLPBoost](https://www.stat.purdue.edu/~vishy/papers/WarGloVis08.pdf)


- Base Learner
    - Decision stump

## What I will write:

- Booster
  - LPBoost
  - ERLPBoost

- Base Learner
  - Decision tree
  - Bag of words

I'm planning to implement the other boosting algorithms in the future.


## How to use

Here is a sample code:

```rust

extern crate boost;

// `run()` and `predict()` are the methods of Booster trait
use boost::booster::core::Booster;
use boost::booster::adaboost::AdaBoost;
use boost::base_learner::dstump::DStump;

// This function reads a file with LIBSVM format
use boost::data_reader::read_libsvm;

fn main() {
    // Set file name
    let file = "/path/to/input/data";

    // Read file
    let sample = read_libsvm(file).unwrap();

    // Initialize Booster
    let mut adaboost = AdaBoost::init(&sample);

    // Initialize Base Learner
    let dstump = Box::new(DStump::init(&sample));

    // Set accuracy parameter
    let accuracy = 0.1;

    // Run boosting algorithm
    adaboost.run(dstump, &sample, accuracy);


    for i in 0..sample.len() {
        let data  = &sample[i].data;
        let label = sample[i].label;

        // Check the predictions
        assert_eq!(adaboost.predict(data), label);
    }
}
```

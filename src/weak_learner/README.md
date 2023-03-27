# `miniboosts/src/weak_learner` directory

This directory defines weak learners.
Weak learners are algorithms that take distribution over examples as input
and output a hypothesis with slightly better accuracy than random guessing.

The weak learners defined in this directory are listed below:

### Weak learners

- [DTree](https://www.amazon.co.jp/-/en/Leo-Breiman/dp/0412048418),  
    Decision Tree.
    Defined in `decision_tree/` directory.
- [RTree](https://www.amazon.co.jp/-/en/Leo-Breiman/dp/0412048418),  
    Regression Tree.
    Defined in `regression_tree/` directory.
- [NeuralNetwork],  
    A naive implementation of neural network.
    Defined in `neural_network/` directory.
- GaussianNB,  
    Naive bayes algorithm.  
    Defined in `naive_bayes/` directory.
    **Note that current implementation is a beta version**.
- WLUnion.  
    Sometimes one wants to use the union of multiple weak learners
    as a single one. This weak learner enables you to do that.


### `WeakLearner` trait
`core.rs` defines `WeakLearner` trait.
If you want to implement your own weak learner,
you must implement `WeakLearner` trait.

See the doc string for further information.

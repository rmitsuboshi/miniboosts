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


### Directory structure

```txt
./
├─ core.rs                    Defines `WeakLearner` trait
│
├─ decision_tree
│  ├ bin.rs                           Defines Feature binning for decision tree
│  ├ builder.rs                       Defines a struct that constructs a decision tree weak learner
│  ├ criterion.rs                     Defines splitting criterion
│  ├ decision_tree_algorithm.rs       Defines decision tree weak learner
│  ├ decision_tree_weak_learner.rs    Defines decision tree classifier
│  ├ node.rs                          Defines the inner representation of `DecisionTreeClassifier`
│  └ train_node.rs                    Defines a node struct for training
├─ regression_tree
│  ├ bin.rs                           Defines Feature binning for regression tree
│  ├ builder.rs                       Defines a struct that constructs a regression tree weak learner
│  ├ loss.rs                          Defines loss functions
│  ├ regression_tree_algorithm.rs     Defines regression tree weak learner
│  ├ regression_tree_weak_learner.rs  Defines regression tree classifier
│  ├ node.rs                          Defines the inner representation of `RegressionTreeClassifier`
│  └ train_node.rs                    Defines a node struct for training
└─ neural_network
   ├ activation.rs                    Defines activation functions
   ├ layer.rs                         Defines layers in neural networks.
   ├ nn_hypothesis.rs                 Defines neural network hypotheses
   ├ nn_loss.rs                       Defines loss functions
   └ nn_weak_learner.rs               Defines neural network weak learner
```

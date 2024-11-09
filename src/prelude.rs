//! Exports the standard boosting algorithms and traits.
//! 
pub use crate::booster::{
    // Booster trait
    Booster,


    // Classification ---------------------------
    // ERM boostings
    AdaBoost,
    MadaBoost,


    // Hard margin maximizing boostings
    AdaBoostV,
    TotalBoost,

    // Soft margin optimization
    SmoothBoost,
    CERLPBoost,
    LPBoost,
    MLPBoost,
    ERLPBoost,
    SoftBoost,


    // Regression
    GBM,


    // Others
    GraphSepBoost,
};


pub use crate::weak_learner::{
    // Base Learner trait
    WeakLearner,


    // Classification ---------------------------
    DecisionTree,
    DecisionTreeBuilder,
    DecisionTreeClassifier,
    Criterion,


    GaussianNB,
    NBayesClassifier,


    NeuralNetwork,
    NNHypothesis,
    Activation,
    NNLoss,


    BadClassifier,
    BadBaseLearner,
    BadBaseLearnerBuilder,


    // Regression -------------------------------
    RegressionTree,
    RegressionTreeBuilder,
    RegressionTreeRegressor,
    LossType,
};


pub use crate::hypothesis::{
    Classifier,
    Regressor,
    WeightedMajority,
};

pub use crate::{
    SampleReader,
    Sample,
};

pub use crate::common::{
    loss_functions::GBMLoss,
    frank_wolfe::FWType,
};


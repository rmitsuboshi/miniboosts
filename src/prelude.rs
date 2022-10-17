//! Exports the standard boosting algorithms and traits.
//! 
pub use crate::booster::{
    Booster,


    // ERM boostings
    AdaBoost,


    // Hard margin maximizing boostings
    AdaBoostV,
    TotalBoost,


    // Soft margin maximizing boostings
    LPBoost,
    ERLPBoost,
    SoftBoost,
    CERLPBoost,

    MLPBoost,
};

pub use crate::booster::mlpboost::{
    Primary,
    Secondary,
    StopCondition,
};

pub use crate::booster::{
};


pub use crate::base_learner::{
    BaseLearner,


    // Decision tree
    DTree,
    DTreeClassifier,
    Criterion,


    // Naive Bayes
    GaussianNB,
    NBayesClassifier,
};


pub use crate::classifier::{
    Classifier,
    CombinedClassifier,
};

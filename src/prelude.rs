//! Exports the standard boosting algorithms and traits.
//! 
pub use crate::booster::{
    // Booster trait
    Booster,


    // Classification ---------------------------
    // ERM boostings
    AdaBoost,


    // Hard margin maximizing boostings
    AdaBoostV,
    TotalBoost,


    // Soft margin maximizing boostings
    LPBoost,
    ERLPBoost,
    SoftBoost,
    SmoothBoost,
    CERLPBoost,

    MLPBoost,

    // Regression -------------------------------

};

pub use crate::booster::mlpboost::{
    Primary,
    Secondary,
    StopCondition,
};


pub use crate::base_learner::{
    // Base Learner trait
    BaseLearner,


    // Classification ---------------------------
    // Decision tree
    DTree,
    DTreeClassifier,
    Criterion,


    // Naive Bayes
    GaussianNB,
    NBayesClassifier,


    // Regression -------------------------------
    RTree,
    RTreeRegressor,
};


pub use crate::classifier::{
    Classifier,
    CombinedClassifier,
};


pub use crate::regressor::{
    Regressor,
    CombinedRegressor,
};

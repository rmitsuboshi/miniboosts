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
};


pub use crate::base_learner::{
    BaseLearner,


    // Decision tree
    DTree,
    DTreeClassifier,
    Criterion,
};


pub use crate::classifier::{
    Classifier,
    CombinedClassifier,
};

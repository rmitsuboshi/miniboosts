#![warn(missing_docs)]

//! 
//! A crate that provides some boosting algorithms.
//! All the boosting algorithm in this crate, 
//! except `LPBoost`, has theoretical iteration bound 
//! until finding a combined hypothesis. 
//! 
//! This crate includes three types of boosting algorithms. 
//! 
//! * Empirical risk minimizing (ERM) boosting
//!     - `AdaBoost`.
//! 
//! 
//! * Hard margin maximizing boosting
//!     - `AdaBoostV`,
//!     - `TotalBoost`.
//! 
//! 
//! * Soft margin maximizing boosting
//!     - `LPBoost`,
//!     - `SoftBoost`,
//!     - `ERLPBoost`,
//!     - `CERLPBoost`.
//! 

pub mod hypothesis;
pub mod booster;
pub mod weak_learner;
pub mod prelude;

pub mod research;


// Export some traits and the combined hypothesis struct.
pub use hypothesis::{
    Classifier,
    Regressor,
    CombinedHypothesis
};




// Export the `Booster` trait.
pub use booster::{
    Booster,
    State,
};


// Export the boosting algorithms that minimizes the empirical loss.
pub use booster::{
    AdaBoost,
};


// Export the boosting algorithms that maximizes the hard margin.
pub use booster::{
    AdaBoostV,
    TotalBoost,
};


// Export the boosting algorithms that maximizes the soft margin.
pub use booster::{
    LPBoost,
    ERLPBoost,
    SoftBoost,
    SmoothBoost,
    CERLPBoost,

    MLPBoost,
};


// Export the boosting algorithms for regression
pub use booster::{
    SquareLevR,
};


// Export the `WeakLearner` trait.
pub use weak_learner::WeakLearner;


// Export the instances of the `WeakLearner` trait.
pub use weak_learner::{
    DTree,
    Criterion,

    WLUnion,


    GaussianNB,
};


// Export the instances of the `Classifier` trait.
// The `CombinedClassifier` is the output of the `Boosting::run(..)`.
pub use weak_learner::{
    DTreeClassifier,
    NBayesClassifier,
};

pub use weak_learner::{
    RTree,
    RTreeRegressor,
    Loss,
};

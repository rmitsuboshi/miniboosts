//! The AdaBoost algorithm proposed
//! by Robert E. Schapire and Yoav Freund.
//! This algorithm is based on the book: 
//! [Boosting: Foundations and Algorithms](https://direct.mit.edu/books/oa-monograph/5342/BoostingFoundations-and-Algorithms)
//! by Robert E. Schapire and Yoav Freund.
//! 
//! AdaBoost is a boosting algorithm for binary classification 
//! that minimizes exponential loss.
//!
pub mod tadaboost_algorithm;

pub use tadaboost_algorithm::TAdaBoost;

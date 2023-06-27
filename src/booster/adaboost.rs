//! The AdaBoost algorithm proposed
//! by Robert E. Schapire and Yoav Freund.
//! This struct is based on the book: 
//! [Boosting: Foundations and Algorithms](https://direct.mit.edu/books/oa-monograph/5342/BoostingFoundations-and-Algorithms)
//! by Robert E. Schapire and Yoav Freund.
//! 
//! AdaBoost is a boosting algorithm for binary classification 
//! that minimizes exponential loss.
//! 
//! As some paper suggests, AdaBoost **approximately** maximizes
//! the hard margin.
//! 
//! `[AdaBoostV](crate::booster::AdaBoostV)`, 
//! a successor of AdaBoost, maximizes the hard margin.
pub mod adaboost_algorithm;

pub use adaboost_algorithm::AdaBoost;

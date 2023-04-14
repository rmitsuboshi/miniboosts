//! Defines `AdaBoostV`.
//! This struct is based on the paper: 
//! [Efficient Margin Maximizing with Boosting](https://www.jmlr.org/papers/v6/ratsch05a.html)
//! by Gunnar Rätsch and Manfred K. Warmuth.
pub mod adaboostv_algorithm;

pub use adaboostv_algorithm::AdaBoostV;

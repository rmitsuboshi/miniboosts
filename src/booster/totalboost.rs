//! `TotalBoost`.
//! This algorithm is originally invented in this paper:
//! [Totally corrective boosting algorithms that maximize the margin](https://dl.acm.org/doi/10.1145/1143844.1143970)
//! by Manfred K. Warmuth, Jun Liao, and Gunnar Rätsch.
pub mod totalboost;

pub use totalboost::TotalBoost;

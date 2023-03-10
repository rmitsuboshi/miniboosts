//! Corrective ERLPBoost struct.  
//! This algorithm is based on this paper:
//! [On the equivalence of weak learnability and linear separability: new relaxations and efficient boosting algorithms](https://link.springer.com/article/10.1007/s10994-010-5173-z)
//! by Shai Shalev-Shwartz and Yoram Singer.
pub mod cerlpboost;

pub use cerlpboost::CERLPBoost;

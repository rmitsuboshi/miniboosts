use polars::prelude::*;

use serde::{
    Serialize,
    Deserialize,
};


use crate::Regressor;
use super::node::*;


/// Regression Tree regressor.
/// This struct is just a wrapper of `Node`.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct RTreeRegressor {
    root: Node,
}

impl From<Node> for RTreeRegressor {
    #[inline]
    fn from(root: Node) -> Self {
        Self { root }
    }
}


impl Regressor for RTreeRegressor {
    fn predict(&self, data: &DataFrame, row: usize) -> f64 {
        self.root.predict(data, row)
    }
}

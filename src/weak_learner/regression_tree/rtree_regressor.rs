use polars::prelude::*;

use serde::{
    Serialize,
    Deserialize,
};

use crate::Regressor;
use super::node::*;

use std::path::Path;
use std::fs::File;
use std::io::prelude::*;


/// Regression Tree regressor.
/// This struct is just a wrapper of `Node`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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



impl RTreeRegressor {
    /// Write the current regression tree to dot file.
    #[inline]
    pub fn to_dot_file<P>(&self, path: P) -> std::io::Result<()>
        where P: AsRef<Path>
    {
        let mut f = File::create(path)?;
        f.write_all(b"graph RegressionTree {")?;


        let info = self.root.to_dot_info(0).0;
        info.into_iter()
            .for_each(|row| {
                f.write_all(row.as_bytes()).unwrap();
            });

        f.write_all(b"}")?;

        Ok(())
    }
}


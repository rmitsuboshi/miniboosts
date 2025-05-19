use miniboosts_core::{
    Regressor,
    Sample,
};

use decision_tree::node::*;
use serde::{Serialize, Deserialize};

use std::path::Path;
use std::fs::File;
use std::io::prelude::*;

/// Regression Tree regressor.
/// This struct is just a wrapper of `Node`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegressionTreeRegressor {
    root: Node,
}

impl From<Box<Node>> for RegressionTreeRegressor {
    #[inline]
    fn from(root: Box<Node>) -> Self {
        Self { root }
    }
}

impl Regressor for RegressionTreeRegressor {
    fn predict(&self, sample: &Sample, row: usize) -> f64 {
        self.root.predict(sample, row)
    }
}

impl RegressionTreeRegressor {
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


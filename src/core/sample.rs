//! Training and test data: [`Sample`], its columns ([`Feature`]), and a file
//! reader ([`SampleReader`]).
//!
//! Data are stored column-wise. Each column is a [`Feature`], either dense
//! (one value per example) or sparse (nonzero entries only). Supported input
//! formats are CSV (with or without a header row), in-memory dense matrices,
//! and SVMlight.

pub mod feature;
pub mod reader;
pub mod sample_struct;

pub use feature::Feature;
pub use reader::SampleReader;
pub use sample_struct::Sample;

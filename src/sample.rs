//! Struct `Sample` represents a batch sample for training.

/// provides feature struct.
pub mod feature_struct;
/// provides sample struct.
pub mod sample_struct;



pub use sample_struct::Sample;
pub use feature_struct::Feature;


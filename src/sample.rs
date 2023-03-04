//! Struct `Sample` represents a batch sample for training.

/// provides feature struct.
pub mod feature;
/// provides sample struct.
pub mod sample;

// pub mod sample_ref;


pub use sample::Sample;
// pub use sample_ref::SampleRef;
pub use feature::Feature;

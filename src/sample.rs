//! Struct `Sample` represents a batch sample.  

// Provides feature struct.
pub(crate) mod feature_struct;
// Provides sample struct.
pub(crate) mod sample_struct;

// Provides a struct that reads a file.
pub(crate) mod sample_reader;


pub use sample_reader::SampleReader;
pub use sample_struct::Sample;
pub use feature_struct::Feature;


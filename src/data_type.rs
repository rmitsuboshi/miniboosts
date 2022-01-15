//! Defines some data structure used in this crate.
use std::collections::HashMap;
use std::ops::Index;


/// Enum of the sparse and dense data.
/// A sparse data is represented by a `HashMap<usize, f64>`,
/// and a dense data is represented by a `Vec<f64>`.
#[derive(Clone, Debug)]
pub enum Data {
    /// Sparse(..) holds the pair of (index, value), where the
    /// value has non-zero value.
    Sparse(HashMap<usize, f64>),
    /// Dense holds the entire data.
    Dense(Vec<f64>),
}

/// Introduce the `Label` for clarity.
pub type Label = f64;


impl Data {
    /// Returns the `index`-th value of the `Data`.
    /// If the `Data` is the sparse data, returns the default value.
    pub fn value_at(&self, index: usize) -> f64 {
        match self {
            Data::Sparse(_data) => {
                match _data.get(&index) {
                    Some(_value) => _value.clone(),
                    None => Default::default()
                }
            },
            Data::Dense(_data) => {
                _data[index].clone()
            }
        }
    }
}


/// Represents the data type of the data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    /// Enum for sparse data
    Sparse,
    /// Enum for dense data
    Dense
}


/// A pair of the instance and its label.
pub struct LabeledData {
    /// Instance
    pub data:  Data,
    /// Label
    pub label: Label,
}


/// A sequence of the `LabeledData`.
/// We assume that all the example in `sample` has the same format.
pub struct Sample {
    /// Vector of the `LabeledData`
    pub sample: Vec<LabeledData>,
    /// Type of the sample
    pub dtype:  DType,
}


impl Sample {

    /// Returns the number of training examples.
    pub fn len(&self) -> usize {
        self.sample.len()
    }

    /// Returns the number of features of examples.
    pub fn feature_len(&self) -> usize {
        let mut feature_size = 0_usize;
        for labeled_data in self.sample.iter() {
            let data = &labeled_data.data;
            feature_size = match data {
                Data::Sparse(_data) => {
                    let l = match _data.keys().max() {
                        Some(&k) => k + 1,
                        None => 0
                    };
                    std::cmp::max(l, feature_size)
                },
                Data::Dense(_data) => std::cmp::max(_data.len(), feature_size)
            }
        }
        feature_size
    }
}


impl Index<usize> for Sample {
    type Output = LabeledData;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.sample[idx]
    }
}


/// Converts the sequence of `Data` and `Label` to `Sample`
pub fn to_sample(examples: Vec<Data>, labels: Vec<Label>)
    -> Sample
{
    let dtype = match &examples[0] {
        &Data::Sparse(_) => DType::Sparse,
        &Data::Dense(_)  => DType::Dense,
    };

    let sample = examples.into_iter()
        .zip(labels)
        .map(|(data, label)| LabeledData { data, label })
        .collect::<Vec<_>>();

    Sample { sample, dtype }
}



/// A struct for implementing the iterator over `Sample`.
pub struct SampleIter<'a> {
    sample: &'a [LabeledData]
}


impl Sample {
    /// Iterator for `Sample`
    pub fn iter(&self) -> SampleIter<'_> {
        SampleIter { sample: &self.sample[..] }
    }
}


impl<'a> Iterator for SampleIter<'a> {
    type Item = &'a LabeledData;

    fn next(&mut self) -> Option<Self::Item> {
        match self.sample.get(0) {
            Some(labeled_data) => {
                self.sample = &self.sample[1..];

                Some(labeled_data)
            },
            None => None
        }
    }
}



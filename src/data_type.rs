//! Defines some data structure used in this crate.
use std::collections::HashMap;
use std::ops::Index;

/// Generic label type.
pub type Label<L> = L;


/// Enum of the sparse and dense data.
/// A sparse data is represented by a `HashMap<usize, D>`,
/// and a dense data is represented by a `Vec<D>`.
#[derive(Clone, Debug)]
pub enum Data<D> {
    /// Sparse data
    Sparse(HashMap<usize, D>),
    /// Dense data
    Dense(Vec<D>),
}


impl<D: Clone + Default> Data<D> {
    /// Returns the `index`-th value of the `Data<D>`.
    /// If the `Data<D>` is the sparse data, returns the default value.
    pub fn value_at(&self, index: usize) -> D {
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
pub struct LabeledData<D, L> {
    /// Instance
    pub data: Data<D>,
    /// Label
    pub label: Label<L>
}


/// A sequence of the `LabeledData<D, L>`.
/// We assume that all the example in `sample` has the same format.
pub struct Sample<D, L> {
    /// Vector of the `LabeledData<D, L>`
    pub sample: Vec<LabeledData<D, L>>,
    /// Type of the sample
    pub dtype: DType
}




impl<D, L> Sample<D, L> {

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


impl<D, L> Index<usize> for Sample<D, L> {
    type Output = LabeledData<D, L>;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.sample[idx]
    }
}


/// Converts the sequence of `Data<D>` and `Label<L>` to `Sample<D, L>`
pub fn to_sample<D, L>(examples: Vec<Data<D>>, labels: Vec<Label<L>>)
    -> Sample<D, L>
{
    let dtype = match &examples[0] {
        &Data::Sparse(_) => DType::Sparse,
        &Data::Dense(_)  => DType::Dense,
    };

    let sample = examples.into_iter()
        .zip(labels)
        .map(|(data, label)| LabeledData { data, label })
        .collect::<Vec<LabeledData<D, L>>>();

    Sample { sample, dtype }
}



/// A struct for implementing the iterator over `Sample<D, L>`.
pub struct SampleIter<'a, D, L> {
    sample: &'a [LabeledData<D, L>]
}


impl<D, L> Sample<D, L> {
    /// Iterator for `Sample<D, L>`
    pub fn iter(&self) -> SampleIter<'_, D, L> {
        SampleIter { sample: &self.sample[..] }
    }
}


impl<'a, D, L> Iterator for SampleIter<'a, D, L> {
    type Item = &'a LabeledData<D, L>;

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




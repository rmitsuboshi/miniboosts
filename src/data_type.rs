use std::collections::HashMap;
use std::ops::Index;

pub type Label<L> = L;


#[derive(Clone, Debug)]
pub enum Data<D> {
    Sparse(HashMap<usize, D>),
    Dense(Vec<D>),
}


impl<D: Clone + Default> Data<D> {
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


#[derive(Clone, Debug)]
pub enum DType {
    Sparse,
    Dense
}


pub struct LabeledData<D, L> {
    pub data: Data<D>,
    pub label: Label<L>
}


pub struct Sample<D, L> {
    pub sample: Vec<LabeledData<D, L>>,
    pub dtype: DType
}




impl<D, L> Sample<D, L> {
    pub fn len(&self) -> usize {
        self.sample.len()
    }

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


pub fn to_sample<D, L>(examples: Vec<Data<D>>, labels: Vec<Label<L>>) -> Sample<D, L> {
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


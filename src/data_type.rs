use std::collections::HashMap;
use std::path::Path;
use std::io;
use std::io::prelude::*;
use std::fs::File;
use std::ops::Index;

pub type Label<L> = L;


#[derive(Clone)]
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


// pub type Example<D, L> = (Data<D>, Label<L>);
// pub type SparseSample<D, L> = Vec<(SparseData<D>, Label<L>)>;
// pub type Sample<D, L> = Vec<(Data<D>, Label<L>)>;
#[derive(Debug, Clone)]
pub enum DType {
    Sparse,
    Dense
}


// pub type Sample<D, L> = Vec<(Data<D>, Label<L>)>;
pub struct Sample<D, L> {
    pub sample: Vec<(Data<D>, Label<L>)>,
    pub dtype: DType
}


impl<D, L> Sample<D, L> {
    pub fn len(&self) -> usize {
        self.sample.len()
    }

    pub fn feature_len(&self) -> usize {
        let mut feature_size = 0_usize;
        let examples = &self.sample;
        for (data, _) in examples.iter() {
            feature_size = match data {
                Data::Sparse(_data) => std::cmp::max(*_data.keys().max().unwrap() + 1_usize, feature_size),
                Data::Dense(_data) => std::cmp::max(_data.len(), feature_size)
            }
        }
        feature_size
    }
}


impl<D, L> Index<usize> for Sample<D, L> {
    type Output = (Data<D>, Label<L>);
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
                         .collect::<Vec<(Data<D>, Label<L>)>>();

    Sample { sample, dtype }
}


// fn libsvm_row2instance<I, D, L>(row: &str) -> Result<(SparseData<I, D>, Label<L>)> {
//     let row_data = row.split_whitespace().iter();
//     let label = match row_data.next() {
//         Some(l) => l.parse::<L>()?,
//         None => Err("Invalid format."),
//     };
// }
// 
// 
// pub fn read_libsvm<P, D, L>(path_arg: P) -> io::Result<Sample<D, L>>
//     where P: AsRef<Path>
// {
//     let path = path_arg.as_ref();
//     println!("path: {:?}", path);
// 
//     let mut file = File::open(path)?;
// 
//     let mut contents = String::new();
//     file.read_to_string(&mut contents)?;
// 
//     let mut examples: Vec<Vec<(usize, f64)>> = Vec::new();
// 
//     for line in contents.lines() {
//         let line = line.split_whitespace().skip(1);
// 
//         let _example = line.map(|s| -> (usize, f64) {
//             let mut _t = s.split(':');
//             let _idx = _t.next().unwrap().parse::<usize>().unwrap();
//             let _val = _t.next().unwrap().parse::<f64>().unwrap();
//             (_idx, _val)
//         }).collect::<Vec<(usize, f64)>>();
// 
//         examples.push(_example);
//     }
//     Ok(examples)
// }

use crate::data_type::{DType, Data, LabeledData, Sample};
use std::str::FromStr;
use std::collections::HashMap;
use std::path::Path;
use std::io;
use std::io::prelude::*;
use std::fs::File;


/// The function `read_libsvm` reads file with LIBSVM format.
pub fn read_libsvm<P, D, L>(path_arg: P) -> io::Result<Sample<D, L>>
    where P: AsRef<Path>,
          D: FromStr,
          <D as FromStr>::Err: std::fmt::Debug,
          L: FromStr,
          <L as FromStr>::Err: std::fmt::Debug
{
    let path = path_arg.as_ref();

    let mut file = File::open(path)?;

    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let mut labels: Vec<L> = Vec::new();
    let mut examples: Vec<HashMap<usize, D>> = Vec::new();

    for line in contents.lines() {
        let mut line = line.split_whitespace();

        labels.push(
            line.next().unwrap().parse::<L>().unwrap()
        );

        let _example = line.map(|s| -> (usize, D) {
            let mut _t = s.split(':');
            let _idx = _t.next().unwrap().parse::<usize>().unwrap();
            let _val = _t.next().unwrap().parse::<D>().unwrap();
            (_idx, _val)
        }).collect::<HashMap<usize, D>>();

        examples.push(_example);
    }


    let sample = examples.into_iter()
                         .map(|data| Data::Sparse(data))
                         .zip(labels)
                         .map(|(data, label)| LabeledData {data, label})
                         .collect::<Vec<LabeledData<D, L>>>();

    let dtype  = DType::Sparse;

    let sample = Sample { sample, dtype };

    Ok(sample)
}

//! Provides some function that reads `Sample<D, L>`
//! from CSV or LIBSVM file.
use crate::data_type::{Sample, Label};
// use crate::data_type::{DType, Data, LabeledData, Sample};
use std::collections::HashMap;
use std::path::Path;
use std::io;
use std::io::prelude::*;
use std::fs::File;


/// The function `read_libsvm` reads file with LIBSVM format.
/// The format must has the following form:
///     label index1:value1 index2:value2 ...
/// Note that since the LIBSVM format is 1-indexed,
/// we subtract `1_usize` to become 0-indexed
pub fn read_libsvm<P>(path_arg: P)
    -> io::Result<Sample<HashMap<usize, f64>>>
    where P: AsRef<Path>
{
    let path = path_arg.as_ref();

    let mut file = File::open(path)?;

    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let mut labels: Vec<Label> = Vec::new();
    let mut examples: Vec<HashMap<usize, f64>> = Vec::new();

    for line in contents.lines() {
        let mut line = line.split_whitespace();

        labels.push(
            line.next().unwrap().parse::<Label>().unwrap()
        );

        let _example = line.map(|s| -> (usize, f64) {
            let mut _t = s.split(':');
            let _idx = _t.next().unwrap().parse::<usize>().unwrap();
            let _val = _t.next().unwrap().parse::<f64>().unwrap();
            (_idx - 1_usize, _val)
        }).collect::<HashMap<usize, f64>>();

        examples.push(_example);
    }


    Ok(Sample::from((examples, labels)))
}



/// The function `read_csv` reads file with CSV format.
/// Each row (data) must have the following form:
///     label,feature_1,feature_2, ...
pub fn read_csv<P>(path_arg: P) -> io::Result<Sample<Vec<f64>>>
    where P: AsRef<Path>
{
    let path = path_arg.as_ref();

    let mut file = File::open(path)?;

    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let mut labels:   Vec<Label>    = Vec::new();
    let mut examples: Vec<Vec<f64>> = Vec::new();

    for line in contents.lines() {
        let mut line = line.split(",");

        labels.push(
            line.next().unwrap().trim().parse::<Label>().unwrap()
        );

        let _example = line.map(|s| -> f64 {
            s.trim().parse::<_>().unwrap()
        }).collect::<Vec<_>>();

        examples.push(_example);
    }


    Ok(Sample::from((examples, labels)))
}

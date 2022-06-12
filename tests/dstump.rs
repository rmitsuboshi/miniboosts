extern crate lycaon;

use std::collections::HashMap;

use lycaon::data_type::*;

use lycaon::{BaseLearner, Classifier};
use lycaon::DStump;



#[test]
fn produce() {
    let examples = vec![
        vec![  1.2, 0.5, -1.0,  2.0],
        vec![  0.1, 0.2,  0.3, -9.0],
        vec![-21.0, 2.0,  1.9,  7.1]
    ];
    let labels = vec![1.0, -1.0, 1.0];


    let sample = Sample::from((examples, labels));


    let dstump = DStump::init(&sample);

    let distribution = vec![1.0/3.0; 3];
    let h = dstump.produce(&sample, &distribution);

    for (dat, lab) in sample.iter() {
        assert_eq!(h.predict(dat), *lab);
    }



    let distribution = vec![0.7, 0.1, 0.2];
    let h = dstump.produce(&sample, &distribution);
    for (dat, lab) in sample.iter() {
        assert_eq!(h.predict(dat), *lab);
    }
}


#[test]
fn produce_sparse() {
    let tuples: Vec<(usize, f64)> = vec![
        (1, 0.2), (3, -12.5), (8, -4.0), (9, 0.8)
    ];

    let mut examples = vec![HashMap::new(); 10];

    for (i, v) in tuples {
        examples[i].insert(0, v);
    }

    let mut labels = vec![1.0; 10];
    labels[3] = -1.0; labels[8] = -1.0;


    let sample = Sample::from((examples, labels));


    let dstump = DStump::init(&sample);

    let distribution = vec![1.0/10.0; 10];
    let h = dstump.produce(&sample, &distribution);
    for (dat, lab) in sample.iter() {
        assert_eq!(h.predict(dat), *lab);
    }
}



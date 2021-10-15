extern crate boost;

use boost::data_type::*;

use boost::base_learner::core::BaseLearner;
use boost::base_learner::dstump::DStump;


#[test]
fn dstump_new() {
    let dstump = DStump::new();
    assert_eq!(dstump.sample_size, 0);
    assert_eq!(dstump.feature_size, 0);
    assert_eq!(dstump.indices.len(), 0);
}


#[test]
fn dstump_with_sample() {
    let examples = vec![
        Data::Dense(vec![  1.2, 0.5, -1.0,  2.0]),
        Data::Dense(vec![  0.1, 0.2,  0.3, -9.0]),
        Data::Dense(vec![-21.0, 2.0,  1.9,  7.1])
    ];
    let labels = vec![1.0, -1.0, 1.0];


    let sample = to_sample(examples, labels);
    let dstump = DStump::with_sample(&sample);


    let ans = vec![
        vec![2, 1, 0],
        vec![1, 0, 2],
        vec![0, 1, 2],
        vec![1, 0, 2]
    ];


    assert_eq!(dstump.sample_size, 3);
    assert_eq!(dstump.feature_size, 4);
    assert_eq!(dstump.indices, ans);
}


#[test]
fn dstump_hypothesis() {
    let examples = vec![
        Data::Dense(vec![  1.2, 0.5, -1.0,  2.0]),
        Data::Dense(vec![  0.1, 0.2,  0.3, -9.0]),
        Data::Dense(vec![-21.0, 2.0,  1.9,  7.1])
    ];
    let labels = vec![1.0, -1.0, 1.0];


    let sample = to_sample(examples, labels);


    let dstump = DStump::with_sample(&sample);

    let distribution = vec![1.0/3.0; 3];
    let h = dstump.best_hypothesis(&sample, &distribution);

    assert_eq!(h.predict(&sample[0].0), sample[0].1);
    assert_eq!(h.predict(&sample[1].0), sample[1].1);
    assert_eq!(h.predict(&sample[2].0), sample[2].1);


    let distribution = vec![0.7, 0.1, 0.2];
    let h = dstump.best_hypothesis(&sample, &distribution);
    assert_eq!(h.predict(&sample[0].0), sample[0].1);
    assert_eq!(h.predict(&sample[1].0), sample[1].1);
    assert_eq!(h.predict(&sample[2].0), sample[2].1);
}





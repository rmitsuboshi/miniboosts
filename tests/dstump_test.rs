extern crate boost;

use boost::base_learner::dstump::DStump;


#[test]
fn dstump_new() {
    let dstump = DStump::new();
    assert_eq!(dstump.example_size, 0);
    assert_eq!(dstump.feature_size, 0);
    assert_eq!(dstump.indices.len(), 0);
}


#[test]
fn dstump_with_sample() {
    let examples = vec![
        vec![  1.2, 0.5, -1.0,  2.0],
        vec![  0.1, 0.2,  0.3, -9.0],
        vec![-21.0, 2.0,  1.9,  7.1]
    ];
    let labels = vec![1.0, -1.0, 1.0];


    let dstump = DStump::with_sample(&examples, &labels);


    let ans = vec![
        vec![2, 1, 0],
        vec![1, 0, 2],
        vec![0, 1, 2],
        vec![1, 0, 2]
    ];


    assert_eq!(dstump.example_size, 3);
    assert_eq!(dstump.feature_size, 4);
    assert_eq!(dstump.indices, ans);
}


#[test]
fn dstump_hypothesis() {
    let examples = vec![
        vec![  1.2, 0.5, -1.0,  2.0],
        vec![  0.1, 0.2,  0.3, -9.0],
        vec![-21.0, 2.0,  1.9,  7.1]
    ];
    let labels = vec![1.0, -1.0, 1.0];


    let dstump = DStump::with_sample(&examples, &labels);

    let distribution = vec![1.0/3.0; 3];
    let h = dstump.best_hypothesis(&examples, &labels, &distribution);

    assert_eq!(h(&examples[0]), labels[0]);
    assert_eq!(h(&examples[1]), labels[1]);
    assert_eq!(h(&examples[2]), labels[2]);


    let distribution = vec![0.7, 0.1, 0.2];
    let h = dstump.best_hypothesis(&examples, &labels, &distribution);
    assert_eq!(h(&examples[0]), labels[0]);
    assert_eq!(h(&examples[1]), labels[1]);
    assert_eq!(h(&examples[2]), labels[2]);
}





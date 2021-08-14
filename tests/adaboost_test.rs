extern crate boost;

use boost::booster::adaboost::AdaBoost;


#[test]
fn adaboost_new() {
    let adaboost = AdaBoost::new();

    assert_eq!(adaboost.dist.len(), 0);
}


#[test]
fn adaboost_with_samplesize() {
    let adaboost = AdaBoost::with_samplesize(10);

    assert_eq!(adaboost.dist.len(), 10);

    let v = adaboost.dist[0];

    assert_eq!(v, 1.0 / 10.0);
}


#[test]
fn predict_test() {
    let examples = vec![ vec![1.0], vec![-1.0] ];
    let labels = vec![1.0, -1.0];
    let h = Box::new(
        |data: &[f64]| -> f64 { data[0].signum() }
    );

    let mut adaboost = AdaBoost::with_samplesize(examples.len());

    adaboost.update_params(h, &examples, &labels);


    let predictions = adaboost.predict_all(&examples);

    for i in 0..examples.len() {
        assert_eq!(labels[i], predictions[i]);
    }
}

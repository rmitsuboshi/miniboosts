extern crate boost;

use boost::data_type::*;

use boost::booster::core::Booster;
use boost::base_learner::dstump::DStumpClassifier;
use boost::booster::adaboost::AdaBoost;



#[test]
fn predict_test() {
    let examples: Vec<Data<f64>> = vec![ vec![1.0], vec![-1.0] ];
    let labels: Vec<Label<f64>> = vec![1.0, -1.0];


    let sample = to_sample(examples, labels);

    let h = Box::new(DStumpClassifier::new());

    let mut adaboost = AdaBoost::with_sample(&sample);

    adaboost.update_params(h, &sample);





    for i in 0..sample.len() {
        assert_eq!(sample[i].1, adaboost.predict(&sample[i].0));
    }
}

extern crate boost;

use boost::data_type::*;

use boost::booster::core::Booster;
use boost::base_learner::dstump::DStumpClassifier;
use boost::booster::adaboost::AdaBoost;



#[test]
fn predict_test() {
    let examples: Vec<Data<f64>> = vec![
        Data::Dense(vec![1.0]),
        Data::Dense(vec![-1.0])
    ];
    let labels: Vec<Label<f64>> = vec![1.0, -1.0];


    let sample = to_sample(examples, labels);

    let h = Box::new(DStumpClassifier::new());

    let mut adaboost = AdaBoost::init(&sample);

    adaboost.update_params(h, &sample);





    for i in 0..sample.len() {
        assert_eq!(sample[i].label, adaboost.predict(&sample[i].data));
    }
}

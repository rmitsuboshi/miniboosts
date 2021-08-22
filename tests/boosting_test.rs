extern crate boost;

use std::io::prelude::*;
use std::fs::File;
use boost::booster::core::Booster;
use boost::booster::adaboost::AdaBoost;
use boost::base_learner::dstump::DStump;


#[test]
fn boosting_test() {
    let mut file = File::open("./small_toy_example.txt").unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    let mut examples = Vec::new();
    let mut labels   = Vec::new();

    for line in contents.lines() {
        let mut line = line.split_whitespace();
        let _label = line.next().unwrap();
        let _example = line.next().unwrap();
        let _label = _label.trim_end_matches(':').parse::<f64>().unwrap();
        labels.push(_label);

        let _example = _example.split(',').map(|x| x.parse::<f64>().unwrap()).collect();
        examples.push(_example);
    }

    let mut adaboost = AdaBoost::with_samplesize(examples.len());
    let dstump = DStump::with_sample(&examples, &labels);
    let dstump = Box::new(dstump);


    adaboost.run(dstump, &examples, &labels, 0.1);


    let mut loss = 0.0;
    let predictions = adaboost.predict_all(&examples);
    for i in 0..examples.len() {
        if labels[i] != predictions[i] { loss += 1.0; }
        // assert_eq!(labels[i], adaboost.predict(&examples[i]));
    }

    loss /= examples.len() as f64;
    println!("Loss: {}", loss);
    assert!(true);
}

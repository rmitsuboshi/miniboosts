extern crate boost;


use std::io::prelude::*;
use std::fs::File;
use std::env;

use boost::data_type::*;
use boost::booster::core::Booster;
use boost::booster::adaboost::AdaBoost;
use boost::base_learner::dstump::DStump;


#[test]
fn boosting_test() {
    let mut path = env::current_dir().unwrap();
    println!("path: {:?}", path);
    path.push("tests/small_toy_example.txt");
    let mut file = File::open(path).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    let mut examples: Vec<Data<f64>> = Vec::new();
    let mut labels: Vec<Label<f64>> = Vec::new();

    for line in contents.lines() {
        let mut line = line.split_whitespace();
        let _label = line.next().unwrap();
        let _example = line.next().unwrap();
        let _label = _label.trim_end_matches(':').parse::<f64>().unwrap();
        labels.push(_label);

        let _example = _example.split(',').map(|x| x.parse::<f64>().unwrap()).collect();
        examples.push(Data::Dense(_example));
    }

    let sample = to_sample(examples, labels);

    let mut adaboost = AdaBoost::with_sample(&sample);
    let dstump = DStump::with_sample(&sample);
    let dstump = Box::new(dstump);


    adaboost.run(dstump, &sample, 0.1);


    let mut loss = 0.0;
    for i in 0..sample.len() {
        let p = adaboost.predict(&sample[i].0);
        if sample[i].1 != p { loss += 1.0; }
    }

    loss /= sample.len() as f64;
    println!("Loss: {}", loss);
    assert!(true);
}

extern crate boost;

use std::env;

use boost::booster::Booster;
use boost::booster::{AdaBoost, LPBoost, ERLPBoost};
use boost::base_learner::DStump;

use boost::data_reader::{read_libsvm, read_csv};


/// Tests for `AdaBoost`.
#[cfg(test)]
pub mod adaboost_tests {
    use super::*;
    #[test]
    fn run() {
        let mut path = env::current_dir().unwrap();
        println!("path: {:?}", path);
        path.push("tests/small_toy_example.csv");

        let sample = read_csv(&path).unwrap();


        let mut adaboost = AdaBoost::init(&sample);
        let dstump = DStump::init(&sample);
        let dstump = Box::new(dstump);


        adaboost.run(dstump, &sample, 0.1);


        let mut loss = 0.0;
        for i in 0..sample.len() {
            let p = adaboost.predict(&sample[i].data);
            if sample[i].label != p { loss += 1.0; }
        }

        loss /= sample.len() as f64;
        println!("Loss: {}", loss);
        assert!(true);
    }


    #[test]
    fn run_with_libsvm() {
        let mut path = env::current_dir().unwrap();
        println!("path: {:?}", path);
        path.push("tests/small_toy_example_libsvm.txt");
        let sample = read_libsvm(path).unwrap();
        println!("sample.len() is: {:?}, sample.feature_len() is: {:?}", sample.len(), sample.feature_len());


        let mut adaboost = AdaBoost::init(&sample);
        let dstump = DStump::init(&sample);
        let dstump = Box::new(dstump);


        adaboost.run(dstump, &sample, 0.1);


        let mut loss = 0.0;
        for i in 0..sample.len() {
            let p = adaboost.predict(&sample[i].data);
            if sample[i].label != p { loss += 1.0; }
        }

        loss /= sample.len() as f64;
        println!("Loss: {}", loss);
        assert!(true);
    }
}


/// Tests for `LPBoost`.
#[cfg(test)]
pub mod lpboost_tests {
    use super::*;
    #[test]
    fn run() {
        let mut path = env::current_dir().unwrap();
        println!("path: {:?}", path);
        path.push("tests/small_toy_example.csv");
        let sample = read_csv(&path).unwrap();
    
        let mut lpboost = LPBoost::init(&sample);
        let dstump = DStump::init(&sample);
        let dstump = Box::new(dstump);


        lpboost.run(dstump, &sample, 0.1);


        let mut loss = 0.0;
        for i in 0..sample.len() {
            let p = lpboost.predict(&sample[i].data);
            if sample[i].label != p { loss += 1.0; }
        }

        loss /= sample.len() as f64;
        println!("Loss: {}", loss);
        assert!((1.0 - lpboost.weights.iter().sum::<f64>().abs()) < 1e-9);
    }


    #[test]
    fn run_with_libsvm() {
        let mut path = env::current_dir().unwrap();
        println!("path: {:?}", path);
        path.push("tests/small_toy_example_libsvm.txt");
        let sample = read_libsvm(path).unwrap();
        println!("sample.len() is: {:?}, sample.feature_len() is: {:?}", sample.len(), sample.feature_len());


        let mut lpboost = LPBoost::init(&sample);
        let dstump = DStump::init(&sample);
        let dstump = Box::new(dstump);


        lpboost.run(dstump, &sample, 0.1);


        let mut loss = 0.0;
        for i in 0..sample.len() {
            let p = lpboost.predict(&sample[i].data);
            if sample[i].label != p { loss += 1.0; }
        }

        loss /= sample.len() as f64;
        println!("Loss: {}", loss);
        assert!((1.0 - lpboost.weights.iter().sum::<f64>().abs()) < 1e-9);
    }
}



/// Tests for `ERLPBoost`.
#[cfg(test)]
pub mod erlpboost_tests {
    use super::*;
    #[test]
    fn run() {
        let mut path = env::current_dir().unwrap();
        println!("path: {:?}", path);
        path.push("tests/small_toy_example.csv");
        let sample = read_csv(&path).unwrap();
    
        let mut erlpboost = ERLPBoost::init(&sample);
        let dstump = DStump::init(&sample);
        let dstump = Box::new(dstump);


        erlpboost.run(dstump, &sample, 0.1);


        let mut loss = 0.0;
        for i in 0..sample.len() {
            let p = erlpboost.predict(&sample[i].data);
            if sample[i].label != p { loss += 1.0; }
        }

        loss /= sample.len() as f64;
        println!("Loss: {}", loss);
        assert!((1.0 - erlpboost.weights.iter().sum::<f64>().abs()) < 1e-9);
    }


    #[test]
    fn run_with_libsvm() {
        let mut path = env::current_dir().unwrap();
        println!("path: {:?}", path);
        path.push("tests/small_toy_example_libsvm.txt");
        let sample = read_libsvm(path).unwrap();
        println!("sample.len() is: {:?}, sample.feature_len() is: {:?}", sample.len(), sample.feature_len());


        let mut erlpboost = ERLPBoost::init(&sample);
        let dstump = DStump::init(&sample);
        let dstump = Box::new(dstump);


        erlpboost.run(dstump, &sample, 0.1);


        let mut loss = 0.0;
        for i in 0..sample.len() {
            let p = erlpboost.predict(&sample[i].data);
            if sample[i].label != p { loss += 1.0; }
        }

        loss /= sample.len() as f64;
        println!("Loss: {}", loss);
        assert!((1.0 - erlpboost.weights.iter().sum::<f64>().abs()) < 1e-9);
    }
}



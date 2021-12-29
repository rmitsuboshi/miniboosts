extern crate lycaon;

use std::env;

use lycaon::booster::Booster;
use lycaon::booster::{AdaBoost, LPBoost, ERLPBoost, SoftBoost};
use lycaon::base_learner::DStump;

use lycaon::{read_libsvm, read_csv};


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
        // let dstump = Box::new(dstump);


        adaboost.run(&dstump, &sample, 0.1);


        let mut loss = 0.0;
        for ex in sample.iter() {
            let p = adaboost.predict(&ex.data);
            if ex.label != p { loss += 1.0; }
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
        // let dstump = Box::new(dstump);


        adaboost.run(&dstump, &sample, 0.1);


        let mut loss = 0.0;
        for ex in sample.iter() {
            let p = adaboost.predict(&ex.data);
            if ex.label != p { loss += 1.0; }
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
        // let dstump = Box::new(dstump);


        lpboost.run(&dstump, &sample, 0.1);


        let mut loss = 0.0;
        for ex in sample.iter() {
            let p = lpboost.predict(&ex.data);
            if ex.label != p { loss += 1.0; }
        }

        loss /= sample.len() as f64;
        println!("Loss: {}", loss);
        assert!((1.0 - lpboost.weights.iter().sum::<f64>().abs()) < 1e-9);
    }

    #[test]
    fn run_softmargin() {
        let mut path = env::current_dir().unwrap();
        println!("path: {:?}", path);
        path.push("tests/small_toy_example.csv");
        let sample = read_csv(&path).unwrap();

        let cap = sample.len() as f64 * 0.2;
    
        let mut lpboost = LPBoost::init(&sample).capping(cap);
        let dstump = DStump::init(&sample);
        // let dstump = Box::new(dstump);


        lpboost.run(&dstump, &sample, 0.1);


        let mut loss = 0.0;
        for ex in sample.iter() {
            let p = lpboost.predict(&ex.data);
            if ex.label != p { loss += 1.0; }
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
        // let dstump = Box::new(dstump);


        lpboost.run(&dstump, &sample, 0.1);


        let mut loss = 0.0;
        for ex in sample.iter() {
            let p = lpboost.predict(&ex.data);
            if ex.label != p { loss += 1.0; }
        }

        loss /= sample.len() as f64;
        println!("Loss: {}", loss);
        assert!((1.0 - lpboost.weights.iter().sum::<f64>().abs()) < 1e-9);
    }


    #[test]
    fn run_with_libsvm_german() {
        let path = "/Users/ryotaroMitsuboshi/Documents/Datasets/german.libsvm";
        let sample = read_libsvm(path).unwrap();
        println!("sample.len() is: {:?}, sample.feature_len() is: {:?}", sample.len(), sample.feature_len());


        let cap = sample.len() as f64 * 0.8;
        let mut lpboost = LPBoost::init(&sample).capping(cap);
        let dstump = DStump::init(&sample);
        // let dstump = Box::new(dstump);


        lpboost.run(&dstump, &sample, 0.001);
        println!("Optimal value(LIBSVM): {}", lpboost.gamma_hat);
    }


    #[test]
    fn run_with_csv_german() {
        let path = "/Users/ryotaroMitsuboshi/Documents/Datasets/german.csv";
        let sample = read_csv(path).unwrap();
        println!("sample.len() is: {:?}, sample.feature_len() is: {:?}", sample.len(), sample.feature_len());


        let cap = sample.len() as f64 * 0.8;
        let mut lpboost = LPBoost::init(&sample).capping(cap);
        let dstump = DStump::init(&sample);
        // let dstump = Box::new(dstump);


        lpboost.run(&dstump, &sample, 0.001);
        println!("Optimal value(CSV): {}", lpboost.gamma_hat);
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
        // let dstump = Box::new(dstump);


        erlpboost.run(&dstump, &sample, 0.1);


        let mut loss = 0.0;
        for ex in sample.iter() {
            let p = erlpboost.predict(&ex.data);
            if ex.label != p { loss += 1.0; }
        }

        loss /= sample.len() as f64;
        println!("Loss: {}", loss);
        assert!((1.0 - erlpboost.weights.iter().sum::<f64>().abs()) < 1e-9);
    }


    #[test]
    fn run_softmargin() {
        let mut path = env::current_dir().unwrap();
        println!("path: {:?}", path);
        path.push("tests/small_toy_example.csv");
        let sample = read_csv(&path).unwrap();


        let cap = sample.len() as f64 * 0.2;
    
        let mut erlpboost = ERLPBoost::init(&sample).capping(cap);
        let dstump = DStump::init(&sample);
        // let dstump = Box::new(dstump);


        erlpboost.run(&dstump, &sample, 0.1);


        let mut loss = 0.0;
        for ex in sample.iter() {
            let p = erlpboost.predict(&ex.data);
            if ex.label != p { loss += 1.0; }
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
        // let dstump = Box::new(dstump);


        erlpboost.run(&dstump, &sample, 0.1);


        let mut loss = 0.0;
        for ex in sample.iter() {
            let p = erlpboost.predict(&ex.data);
            if ex.label != p { loss += 1.0; }
        }

        loss /= sample.len() as f64;
        println!("Loss: {}", loss);
        assert!((1.0 - erlpboost.weights.iter().sum::<f64>().abs()) < 1e-9);
    }


    #[test]
    fn run_with_libsvm_german() {
        let path = "/Users/ryotaroMitsuboshi/Documents/Datasets/german.libsvm";
        let sample = read_libsvm(path).unwrap();
        println!("sample.len() is: {:?}, sample.feature_len() is: {:?}", sample.len(), sample.feature_len());


        let cap = sample.len() as f64 * 0.8;
        let mut erlpboost = ERLPBoost::init(&sample).capping(cap);
        let dstump = DStump::init(&sample);
        // let dstump = Box::new(dstump);


        erlpboost.run(&dstump, &sample, 0.001);
        println!("Optimal value: {}", erlpboost.gamma_hat);
    }
}




/// Tests for `SoftBoost`.
/// Since `TotalBoost` is implemented via `SoftBoost`,
/// we do not need to test `TotalBoost`.
#[cfg(test)]
pub mod softboost_tests {
    use super::*;
    #[test]
    fn run() {
        let mut path = env::current_dir().unwrap();
        println!("path: {:?}", path);
        path.push("tests/small_toy_example.csv");
        let sample = read_csv(&path).unwrap();
    
        let mut softboost = SoftBoost::init(&sample);
        let dstump = DStump::init(&sample);
        // let dstump = Box::new(dstump);


        softboost.run(&dstump, &sample, 0.1);


        let mut loss = 0.0;
        for ex in sample.iter() {
            let p = softboost.predict(&ex.data);
            if ex.label != p { loss += 1.0; }
        }

        loss /= sample.len() as f64;
        println!("Loss: {}", loss);
        assert!((1.0 - softboost.weights.iter().sum::<f64>().abs()) < 1e-9);
    }


    #[test]
    fn run_softmargin() {
        let mut path = env::current_dir().unwrap();
        println!("path: {:?}", path);
        path.push("tests/small_toy_example.csv");
        let sample = read_csv(&path).unwrap();


        let cap = sample.len() as f64 * 0.2;
    
        let mut softboost = SoftBoost::init(&sample).capping(cap);
        let dstump = DStump::init(&sample);
        // let dstump = Box::new(dstump);


        softboost.run(&dstump, &sample, 0.1);


        let mut loss = 0.0;
        for ex in sample.iter() {
            let p = softboost.predict(&ex.data);
            if ex.label != p { loss += 1.0; }
        }

        loss /= sample.len() as f64;
        println!("Loss: {}", loss);
        assert!((1.0 - softboost.weights.iter().sum::<f64>().abs()) < 1e-9);
    }


    #[test]
    fn run_with_libsvm() {
        let mut path = env::current_dir().unwrap();
        println!("path: {:?}", path);
        path.push("tests/small_toy_example_libsvm.txt");
        let sample = read_libsvm(path).unwrap();
        println!("sample.len() is: {:?}, sample.feature_len() is: {:?}", sample.len(), sample.feature_len());


        let mut softboost = SoftBoost::init(&sample);
        let dstump = DStump::init(&sample);
        // let dstump = Box::new(dstump);


        softboost.run(&dstump, &sample, 0.1);


        let mut loss = 0.0;
        for ex in sample.iter() {
            let p = softboost.predict(&ex.data);
            if ex.label != p { loss += 1.0; }
        }

        loss /= sample.len() as f64;
        println!("Loss: {}", loss);
        assert!((1.0 - softboost.weights.iter().sum::<f64>().abs()) < 1e-9);
    }
}

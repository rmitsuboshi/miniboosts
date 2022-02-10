extern crate lycaon;

use std::env;

use lycaon::Booster;
use lycaon::AdaBoost;
use lycaon::{LPBoost, ERLPBoost, SoftBoost};
use lycaon::{Classifier, DStump};

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


        let f = adaboost.run(&dstump, &sample, 0.1);


        let mut loss = 0.0;
        for (dat, lab) in sample.iter() {
            let p = f.predict(dat);
            if *lab != p { loss += 1.0; }
        }

        loss /= sample.len() as f64;
        println!("Loss (LPBoost): {loss}");
        assert!(true);
    }


    #[test]
    fn run_with_libsvm() {
        let mut path = env::current_dir().unwrap();
        println!("path: {:?}", path);
        path.push("tests/small_toy_example_libsvm.txt");
        let sample = read_libsvm(path).unwrap();
        println!(
            "sample.len() is: {m}, sample.dim() is: {dim}",
            m = sample.len(), dim = sample.dim()
        );


        let mut adaboost = AdaBoost::init(&sample);
        let dstump = DStump::init(&sample);


        let f = adaboost.run(&dstump, &sample, 0.1);


        let mut loss = 0.0;
        for (dat, lab) in sample.iter() {
            let p = f.predict(dat);
            if *lab != p { loss += 1.0; }
        }

        loss /= sample.len() as f64;
        println!("Loss (LPBoost): {loss}");
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


        let f = lpboost.run(&dstump, &sample, 0.1);


        let mut loss = 0.0;
        for (dat, lab) in sample.iter() {
            let p = f.predict(dat);
            if *lab != p { loss += 1.0; }
        }

        loss /= sample.len() as f64;
        println!("Loss (LPBoost): {loss}");
        assert!(true);
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


        let f = lpboost.run(&dstump, &sample, 0.1);


        let mut loss = 0.0;
        for (dat, lab) in sample.iter() {
            let p = f.predict(dat);
            if *lab != p { loss += 1.0; }
        }

        loss /= sample.len() as f64;
        println!("Loss (LPBoost): {loss}");
    }


    #[test]
    fn run_with_libsvm() {
        let mut path = env::current_dir().unwrap();
        println!("path: {:?}", path);
        path.push("tests/small_toy_example_libsvm.txt");
        let sample = read_libsvm(path).unwrap();
        println!(
            "sample.len() is: {m}, sample.dim() is: {dim}",
            m = sample.len(), dim = sample.dim()
        );


        let mut lpboost = LPBoost::init(&sample);
        let dstump = DStump::init(&sample);


        let f = lpboost.run(&dstump, &sample, 0.1);


        let mut loss = 0.0;
        for (dat, lab) in sample.iter() {
            let p = f.predict(dat);
            if *lab != p { loss += 1.0; }
        }

        loss /= sample.len() as f64;
        println!("Loss: {}", loss);
        assert!(true);
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


        let f = erlpboost.run(&dstump, &sample, 0.1);


        let mut loss = 0.0;
        for (dat, lab) in sample.iter() {
            let p = f.predict(dat);
            if *lab != p { loss += 1.0; }
        }

        loss /= sample.len() as f64;
        println!("Loss (ERLPBoost): {loss}");
        assert!(true);
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


        let f = erlpboost.run(&dstump, &sample, 0.1);


        let mut loss = 0.0;
        for (dat, lab) in sample.iter() {
            let p = f.predict(dat);
            if *lab != p { loss += 1.0; }
        }

        loss /= sample.len() as f64;
        println!("Loss (ERLPBoost): {loss}");
        assert!(true);
    }


    #[test]
    fn run_with_libsvm() {
        let mut path = env::current_dir().unwrap();
        println!("path: {:?}", path);
        path.push("tests/small_toy_example_libsvm.txt");
        let sample = read_libsvm(path).unwrap();
        println!(
            "sample.len() is: {m}, sample.dim() is: {dim}",
            m = sample.len(), dim = sample.dim()
        );


        let mut erlpboost = ERLPBoost::init(&sample);
        let dstump = DStump::init(&sample);


        let f = erlpboost.run(&dstump, &sample, 0.1);


        let mut loss = 0.0;
        for (dat, lab) in sample.iter() {
            let p = f.predict(dat);
            if *lab != p { loss += 1.0; }
        }

        loss /= sample.len() as f64;
        println!("Loss (ERLPBoost): {loss}");
        assert!(true);
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


        let f = softboost.run(&dstump, &sample, 0.1);


        let mut loss = 0.0;
        for (dat, lab) in sample.iter() {
            let p = f.predict(dat);
            if *lab != p { loss += 1.0; }
        }

        loss /= sample.len() as f64;
        println!("Loss: {loss}");
        assert!(true);
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


        let f = softboost.run(&dstump, &sample, 0.1);


        let mut loss = 0.0;
        for (dat, lab) in sample.iter() {
            let p = f.predict(dat);
            if *lab != p { loss += 1.0; }
        }

        loss /= sample.len() as f64;
        println!("Loss: {loss}");
        assert!(true);
    }


    #[test]
    fn run_with_libsvm() {
        let mut path = env::current_dir().unwrap();
        println!("path: {:?}", path);
        path.push("tests/small_toy_example_libsvm.txt");
        let sample = read_libsvm(path).unwrap();
        println!(
            "sample.len() is: {m}, sample.dim() is: {dim}",
            m = sample.len(), dim = sample.dim()
        );


        let mut softboost = SoftBoost::init(&sample);
        let dstump = DStump::init(&sample);


        let f = softboost.run(&dstump, &sample, 0.1);


        let mut loss = 0.0;
        for (dat, lab) in sample.iter() {
            let p = f.predict(dat);
            if *lab != p { loss += 1.0; }
        }

        loss /= sample.len() as f64;
        println!("Loss: {loss}");
        assert!(true);
    }
}

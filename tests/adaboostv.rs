use std::env;
use miniboosts::prelude::*;


/// Tests for `AdaBoostV`.
#[cfg(test)]
pub mod adaboostv_tests {
    use super::*;
    #[test]
    fn german() {
        let mut path = env::current_dir().unwrap();
        path.push("tests/dataset/german.csv");

        let sample = Sample::from_csv(path, true)
            .unwrap()
            .set_target("class");


        let mut booster = AdaBoostV::init(&sample)
            .tolerance(0.1);
        let weak_learner = DTree::init(&sample)
            .max_depth(3)
            .criterion(Criterion::Entropy);


        let f = booster.run(&weak_learner);


        let (m, _) = sample.shape();
        let predictions = f.predict_all(&sample);

        let loss = sample.target()
            .into_iter()
            .zip(predictions)
            .map(|(t, p)| if *t != p as f64 { 1.0 } else { 0.0 })
            .sum::<f64>() / m as f64;

        println!("Loss (german.csv, AdaBoostV, DTree): {loss}");
        assert!(true);
    }
}



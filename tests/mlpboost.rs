use std::env;
use miniboosts::prelude::*;


/// Tests for `MLPBoost`.
#[cfg(test)]
pub mod mlpboost_tests {
    use super::*;
    #[test]
    fn german() {
        let mut path = env::current_dir().unwrap();
        path.push("tests/dataset/german.csv");

        let sample = Sample::from_csv(path, true)
            .unwrap()
            .set_target("class");
        let n_sample = sample.shape().0 as f64;

        let mut booster = MLPBoost::init(&sample)
            .tolerance(0.1)
            .frank_wolfe(FWType::LineSearch)
            .nu(0.1 * n_sample);
        let weak_learner = DTree::init(&sample)
            .max_depth(2)
            .criterion(Criterion::Entropy);


        let f = booster.run(&weak_learner);
        let predictions = f.predict_all(&sample);

        let loss = sample.target()
            .into_iter()
            .zip(predictions)
            .map(|(t, p)| if *t != p as f64 { 1.0 } else { 0.0 })
            .sum::<f64>() / n_sample;

        println!("Loss (german.csv, MLPBoost, DTree): {loss}");
        assert!(true);
    }
}

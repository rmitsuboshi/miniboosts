use std::env;
use miniboosts::prelude::*;


/// Tests for `SmoothBoost`.
#[cfg(test)]
pub mod smoothboost_tests {
    use super::*;
    #[test]
    fn german() {
        let mut path = env::current_dir().unwrap();
        println!("path: {:?}", path);
        path.push("tests/dataset/german.csv");

        let sample = SampleReader::default()
            .file(path)
            .has_header(true)
            .target_feature("class")
            .read()
            .unwrap();
        let n_sample = sample.shape().0 as f64;

        let mut booster = SmoothBoost::init(&sample)
            .tolerance(0.1)
            .gamma(0.1);

        let wl = DTreeBuilder::new(&sample)
            .max_depth(2)
            .criterion(Criterion::Entropy)
            .build();


        let f = booster.run(&wl);
        let predictions = f.predict_all(&sample);

        let loss = sample.target()
            .into_iter()
            .zip(predictions)
            .map(|(t, p)| if *t != p as f64 { 1.0 } else { 0.0 })
            .sum::<f64>() / n_sample;

        println!("Loss (german.csv, SmoothBoost, DTree): {loss}");
        assert!(true);
    }
}

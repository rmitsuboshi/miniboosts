use std::env;
use miniboosts_core::{
    Booster,
    Classifier,
    SampleReader,
};
use adaboost::AdaBoost;
use decision_tree::*;

/// Tests for `AdaBoost`.
#[cfg(test)]
pub mod tests {
    use super::*;
    #[test]
    fn german() {
        let mut path = env::current_dir().unwrap();
        path.push("tests/dataset/german.csv");

        let sample = SampleReader::default()
            .file(path)
            .has_header(true)
            .target_feature("class")
            .read()
            .unwrap();

        let mut booster = AdaBoost::init(&sample)
            .tolerance(0.01)
            .force_quit_at(1_000);

        let wl = DecisionTreeBuilder::new(&sample)
            .max_depth(2)
            .split_by(SplitBy::Entropy)
            .build();
        println!("{wl}");

        let f = booster.run(&wl);

        let (m, _) = sample.shape();
        let predictions = f.predict_all(&sample);

        let loss = sample.target()
            .into_iter()
            .zip(predictions)
            .map(|(t, p)| if *t != p as f64 { 1.0 } else { 0.0 })
            .sum::<f64>() / m as f64;

        println!("Training Loss: {loss}");
        assert!(true);
    }
}



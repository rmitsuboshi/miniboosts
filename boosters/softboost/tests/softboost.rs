use std::env;
use miniboosts_core::{
    Booster,
    Classifier,
    SampleReader,
};
use softboost::SoftBoost;
use decision_tree::*;

/// Tests for `SoftBoost`.
#[cfg(test)]
pub mod tests {
    use super::*;
    // #[test]
    // fn german() {
    //     let mut path = env::current_dir().unwrap();
    //     path.push("tests/dataset/german.csv");

    //     let sample = SampleReader::default()
    //         .file(path)
    //         .has_header(true)
    //         .target_feature("class")
    //         .read()
    //         .unwrap();

    //     let mut booster = SoftBoost::init(&sample)
    //         .tolerance(0.01);

    //     let wl = DecisionTreeBuilder::new(&sample)
    //         .max_depth(2)
    //         .split_by(SplitBy::Entropy)
    //         .build();
    //     println!("{wl}");

    //     let f = booster.run(&wl);

    //     let (m, _) = sample.shape();
    //     let predictions = f.predict_all(&sample);

    //     let loss = sample.target()
    //         .into_iter()
    //         .zip(predictions)
    //         .map(|(t, p)| if *t != p as f64 { 1.0 } else { 0.0 })
    //         .sum::<f64>() / m as f64;

    //     println!("Training Loss: {loss}");
    //     assert!(true);
    // }

    use research::*;
    use optimization::SoftMarginObjective;
    use miniboosts_core::Sample;
    #[test]
    fn research() {
        let mut path = env::current_dir().unwrap();
        path.push("tests/dataset/german.csv");

        let sample = SampleReader::default()
            .file(path)
            .has_header(true)
            .target_feature("class")
            .read()
            .unwrap();
        let m = sample.shape().0;

        const TOLERANCE: f64 = 0.001;
        let weak_learner = DecisionTreeBuilder::new(&sample)
            .max_depth(2)
            .split_by(SplitBy::Entropy)
            .build();
        let nu = 0.01 * m as f64;
        let booster = SoftBoost::init(&sample)
            .tolerance(TOLERANCE)
            .nu(nu);
        let objective_function = SoftMarginObjective::new(nu);

        let mut logger = LoggerBuilder::new()
            .booster(booster)
            .weak_learner(weak_learner)
            .train_sample(&sample)
            .test_sample(&sample)
            .objective_function(objective_function)
            .loss_function(zero_one_loss)
            .print_every(1)
            .build();
        let f = logger.run("test.csv");
        // println!("f = {f:?}");
    }

    fn zero_one_loss<H: Classifier>(sample: &Sample, f: &H) -> f64 {
        let m = sample.shape().0 as f64;

        let y = sample.target();

        f.predict_all(sample)
            .into_iter()
            .zip(y)
            .map(|(fx, &y)| if y * fx as f64 <= 0f64 { 1f64 } else { 0f64 })
            .sum::<f64>()
            / m
    }
}


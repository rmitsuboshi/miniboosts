use std::env;
use miniboosts::prelude::*;


/// Tests for `GBM`.
#[cfg(test)]
pub mod gbm_boston {
    use super::*;
    #[test]
    fn boston() {
        let mut path = env::current_dir().unwrap();
        path.push("tests/dataset/california-housing.csv");

        let sample = SampleReader::new()
            .file(path)
            .has_header(true)
            .target_feature("MedHouseVal")
            .read()
            .unwrap();


        let n_sample = sample.shape().0 as f64;

        let mut gbm = GBM::init_with_loss(&sample, GBMLoss::L2)
            .loss(GBMLoss::L2);
        let tree = RegressionTreeBuilder::new(&sample)
            .max_depth(3)
            .loss(LossType::L2)
            .build();

        println!("{tree}");

        let f = gbm.run(&tree);
        let predictions = f.predict_all(&sample);


        let target = sample.target();
        let loss = target.iter()
            .copied()
            .zip(&predictions[..])
            .map(|(t, p)| (t - p).abs())
            .sum::<f64>() / n_sample;
        println!("L1-Loss (boston_housing.csv, GBM, RegressionTree): {loss}");

        let loss = target.iter()
            .copied()
            .zip(&predictions[..])
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f64>() / n_sample;

        println!("L2-Loss (boston_housing.csv, GBM, RegressionTree): {loss}");

        let mean = target.iter().sum::<f64>() / n_sample;
        let loss = target.iter()
            .copied()
            .map(|y| (y - mean).powi(2))
            .sum::<f64>()
            / n_sample;

        println!("Loss (mean prediction): {loss}");

        assert!(true);
    }
}

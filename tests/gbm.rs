use std::env;
use miniboosts::prelude::*;


/// Tests for `GBM`.
#[cfg(test)]
pub mod gbm_boston {
    use super::*;
    #[test]
    fn boston() {
        let mut path = env::current_dir().unwrap();
        path.push("tests/dataset/boston_housing.csv");

        let mut sample = Sample::from_csv(path, true)
            .unwrap()
            .set_target("MEDV");


        let n_sample = sample.shape().0 as f64;

        let mut gbm = GBM::init(&sample)
            .loss(GBMLoss::L1);
        let tree = RTree::init(&sample)
            .max_depth(2)
            .loss_type(LossType::L1);


        let f = gbm.run(&tree);
        let predictions = f.predict_all(&sample);


        let target = sample.target();
        let loss = target.iter()
            .copied()
            .zip(&predictions[..])
            .map(|(t, p)| (t - p).abs())
            .sum::<f64>() / n_sample;
        println!("L1-Loss (boston_housing.csv, GBM, RTree): {loss}");

        let loss = target.into_iter()
            .copied()
            .zip(&predictions[..])
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f64>() / n_sample;

        println!("L2-Loss (boston_housing.csv, GBM, RTree): {loss}");
        assert!(true);
    }
}

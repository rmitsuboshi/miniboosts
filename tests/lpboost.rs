use std::env;
use miniboosts::prelude::*;



/// Tests for `LPBoost`.
#[cfg(test)]
pub mod lpboost_tests {
    use super::*;
    #[test]
    fn german() {
        let mut path = env::current_dir().unwrap();
        path.push("tests/dataset/iris_binary.csv");

        let sample = Sample::from_csv(path, true)
            .unwrap()
            .set_target("class");
        let n_sample = sample.shape().0 as f64;

        let mut booster = ERLPBoost::init(&sample)
            .tolerance(0.001)
            .nu(1.0);

        let wl = DecisionTreeBuilder::new(&sample)
            .max_depth(1)
            .criterion(Criterion::Entropy)
            .build();


        let f = booster.run(&wl);
        println!("f = {f:?}");
        let predictions = f.predict_all(&sample);

        let loss = sample.target()
            .into_iter()
            .zip(predictions)
            .map(|(t, p)| if *t != p as f64 { 1.0 } else { 0.0 })
            .sum::<f64>() / n_sample;

        println!("Loss (german.csv, LPBoost, DTree): {loss}");
        assert!(true);
    }


    // #[test]
    // fn german_svmlight() {
    //     let mut path = env::current_dir().unwrap();
    //     path.push("tests/dataset/german.svmlight");

    //     let sample = Sample::from_svmlight(path).unwrap();
    //     let n_sample = sample.shape().0 as f64;

    //     let mut booster = LPBoost::init(&sample)
    //         .tolerance(0.1)
    //         .nu(0.1 * n_sample);

    //     let wl = DecisionTreeBuilder::new(&sample)
    //         .max_depth(2)
    //         .criterion(Criterion::Entropy)
    //         .build();


    //     let f = booster.run(&wl);
    //     let predictions = f.predict_all(&sample);

    //     let loss = sample.target()
    //         .into_iter()
    //         .zip(predictions)
    //         .map(|(t, p)| if *t != p as f64 { 1.0 } else { 0.0 })
    //         .sum::<f64>() / n_sample;

    //     println!("Loss (german.svmlight, LPBoost, DTree): {loss}");
    //     assert!(true);
    // }


    // #[test]
    // fn german_svmlight_nn() {
    //     let mut path = env::current_dir().unwrap();
    //     path.push("tests/dataset/german.svmlight");

    //     let sample = Sample::from_svmlight(path).unwrap();
    //     let n_sample = sample.shape().0 as f64;

    //     let mut booster = LPBoost::init(&sample)
    //         .tolerance(0.1)
    //         .nu(0.1 * n_sample);
    //     let weak_learner = NeuralNetwork::init(&sample)
    //         .append(100, Activation::ReLu(1.0))
    //         .append(2, Activation::SoftMax(1.0))
    //         .n_epoch(10)
    //         .n_iter(100);


    //     let f = booster.run(&weak_learner);
    //     let predictions = f.predict_all(&sample);

    //     let loss = sample.target()
    //         .into_iter()
    //         .zip(predictions)
    //         .map(|(t, p)| if *t != p as f64 { 1.0 } else { 0.0 })
    //         .sum::<f64>() / n_sample;

    //     println!("Loss (german.svmlight, LPBoost, NN): {loss}");
    //     assert!(true);
    // }
}

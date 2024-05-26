use std::env;
use miniboosts::prelude::*;
use miniboosts::research::Logger;
use miniboosts::SoftMarginObjective;

fn zero_one_loss<H>(sample: &Sample, f: &H)
    -> f64
    where H: Classifier
{
    let n_sample = sample.shape().0 as f64;

    let target = sample.target();

    f.predict_all(sample)
        .into_iter()
        .zip(target.into_iter())
        .map(|(hx, &y)| if hx != y as i64 { 1.0 } else { 0.0 })
        .sum::<f64>()
        / n_sample
}

const TIME_LIMIT: u128 = 60_000; // 1 minute as millisecond.


/// Tests for `LPBoost`.
#[cfg(test)]
pub mod lpboost_tests {
    use super::*;
    #[test]
    fn german() {
        let mut path = env::current_dir().unwrap();
        path.push("tests/dataset/iris_binary.csv");

        let sample = SampleReader::new()
            .file(path)
            .has_header(true)
            .target_feature("class")
            .read()
            .unwrap();
        let n_sample = sample.shape().0 as f64;

        let mut booster = LPBoost::init(&sample)
            .tolerance(0.001)
            .nu(1.0);

        let wl = DecisionTreeBuilder::new(&sample)
            .max_depth(2)
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


    #[test]
    fn worstcase() {
        let n_sample = 1_000;
        let nu = n_sample as f64 * 0.001;
        let tol = 0.01;

        let sample = Sample::dummy(n_sample);
        let booster = LPBoost::init(&sample)
            .tolerance(tol)
            .nu(nu);

        let wl = BadBaseLearnerBuilder::new(&sample)
            .tolerance(tol)
            .nu(nu)
            .build();

        let objective = SoftMarginObjective::new(nu);

        let mut logger = Logger::new(
                booster, wl, objective, zero_one_loss, &sample, &sample
            )
            .time_limit_as_millis(TIME_LIMIT)
            .print_every(1);

        let f = logger.run("dummy.csv").unwrap();
        println!("f = {f:?}");
        let predictions = f.predict_all(&sample);
        let loss = sample.target()
            .into_iter()
            .zip(predictions)
            .map(|(t, p)| if *t != p as f64 { 1.0 } else { 0.0 })
            .sum::<f64>() / n_sample as f64;

        println!("Loss (Dummy, BadLearner): {loss}");
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

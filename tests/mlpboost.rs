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

/// Tests for `MLPBoost`.
#[cfg(test)]
pub mod mlpboost_tests {
    use super::*;
    #[test]
    fn bcancer() {
        const TOLERANCE: f64 = 0.001;
        const TIME_LIMIT: u128 = 60_000; // 1 minute as millisecond.
        let path = "img/csv/breast-cancer-train.csv";

        let train = SampleReader::default()
            .file(path)
            .has_header(true)
            .target_feature("class")
            .read()
            .unwrap();

        let n_sample = train.shape().0 as f64;
        let nu = 0.01 * n_sample;
        // let nu = 1.0;

        let path = "img/csv/breast-cancer-test.csv";
        let test = SampleReader::default()
            .file(path)
            .has_header(true)
            .target_feature("class")
            .read()
            .unwrap();

        let objective = SoftMarginObjective::new(nu);
        let booster = MLPBoost::init(&train)
            .tolerance(TOLERANCE)
            .frank_wolfe(FWType::Classic)
            .nu(nu);
        let tree = DecisionTreeBuilder::new(&train)
            .max_depth(1)
            .criterion(Criterion::Entropy)
            .build();
        let mut logger = Logger::new(
            booster, tree, objective, zero_one_loss, &train, &test
        ).time_limit_as_millis(TIME_LIMIT);
        let _ = logger.run("mlpboost.csv");
    }
    // #[test]
    // fn german() {
    //     let mut path = env::current_dir().unwrap();
    //     path.push("tests/dataset/german.csv");

    //     let sample = SampleReader::new()
    //         .file(path)
    //         .has_header(true)
    //         .target_feature("class")
    //         .read()
    //         .unwrap();
    //     let n_sample = sample.shape().0 as f64;

    //     let mut booster = MLPBoost::init(&sample)
    //         .tolerance(0.1)
    //         .frank_wolfe(FWType::ShortStep)
    //         .nu(0.1 * n_sample);

    //     let wl = DTreeBuilder::new(&sample)
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

    //     println!("Loss (german.csv, MLPBoost, DTree): {loss}");
    //     assert!(true);
    // }
}

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


/// Tests for `SoftBoost`.
#[cfg(test)]
pub mod softboost_tests {
    use super::*;
    #[test]
    fn bcancer() {
        const TOLERANCE: f64 = 0.001;
        let path = "img/csv/breast-cancer-train.csv";

        let train = SampleReader::new()
            .file(path)
            .has_header(true)
            .target_feature("class")
            .read()
            .unwrap();

        let n_sample = train.shape().0 as f64;
        let nu = 0.01 * n_sample;
        println!("capping is: {nu}");
        // let nu = 1.0;

        let path = "img/csv/breast-cancer-test.csv";

        let test = SampleReader::new()
            .file(path)
            .has_header(true)
            .target_feature("class")
            .read()
            .unwrap();
        let objective = SoftMarginObjective::new(nu);
        let booster = SoftBoost::init(&train)
            .tolerance(TOLERANCE)
            .nu(nu);
        let tree = DecisionTreeBuilder::new(&train)
            .max_depth(1)
            .criterion(Criterion::Entropy)
            .build();
        let time_limit = 1000;
        let mut logger = Logger::new(
                booster, tree, objective, zero_one_loss, &train, &test
            )
            .time_limit_as_secs(time_limit)
            .print_every(10);
        let _ = logger.run("softboost.csv");
    }
}

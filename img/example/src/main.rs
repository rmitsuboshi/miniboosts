use miniboosts::prelude::*;
use miniboosts::research::{
    Logger,
};
use miniboosts::common::objective_functions::{
    SoftMarginObjective,
};


fn zero_one_loss<H>(sample: &Sample, f: &CombinedHypothesis<H>)
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


fn main() {
    let mut args = std::env::args().skip(1);

    // Read the train file
    let path = args.next()
        .expect("[USAGE] ./example [csv file (train)] [csv file (test)]");

    let train = Sample::from_csv(path, true)
        .unwrap()
        .set_target("class");

    let n_sample = train.shape().0 as f64;
    let nu = 0.01 * n_sample;

    let path = args.next()
        .expect("[USAGE] ./example [csv file (train)] [csv file (test)]");
    let test = Sample::from_csv(path, true)
        .unwrap()
        .set_target("class");


    // // Run AdaBoost
    // let objective = SoftMarginObjective::new(1.0);
    // println!("Running AdaBoost");
    // let booster = AdaBoost::init(&train)
    //     .tolerance(0.01);
    // let tree = DTree::init(&train)
    //     .max_depth(2)
    //     .criterion(Criterion::Entropy);
    // let mut logger = Logger::new(
    //     booster, tree, objective, zero_one_loss, &train, &test
    // );
    // let _ = logger.run("adaboost.csv");


    // // Run AdaBoostV
    // let objective = SoftMarginObjective::new(1.0);
    // println!("Running AdaBoostV");
    // let booster = AdaBoostV::init(&train)
    //     .tolerance(0.01);
    // let tree = DTree::init(&train)
    //     .max_depth(2)
    //     .criterion(Criterion::Entropy);
    // let mut logger = Logger::new(
    //     booster, tree, objective, zero_one_loss, &train, &test
    // );
    // let _ = logger.run("adaboostv.csv");


    // // Run TotalBoost
    // let objective = SoftMarginObjective::new(1.0);
    // println!("Running TotalBoost");
    // let booster = TotalBoost::init(&train)
    //     .tolerance(0.01);
    // let tree = DTree::init(&train)
    //     .max_depth(2)
    //     .criterion(Criterion::Entropy);
    // let mut logger = Logger::new(
    //     booster, tree, objective, zero_one_loss, &train, &test
    // );
    // let _ = logger.run("totalboost.csv");


    // // Run SmoothBoost
    // // `gamma` is the weak-learner guarantee.
    // // For this case, the following holds;
    // // for any distribution, the weak learner returns a hypothesis
    // // such that the edge is at least 0.006.
    // // This value `0.006` is derived from the LPBoost.
    // let objective = SoftMarginObjective::new(nu);
    // println!("Running SmoothBoost");
    // let booster = SmoothBoost::init(&train)
    //     .tolerance(0.01)
    //     .gamma(0.006);
    // let tree = DTree::init(&train)
    //     .max_depth(2)
    //     .criterion(Criterion::Entropy);
    // let mut logger = Logger::new(
    //     booster, tree, objective, zero_one_loss, &train, &test
    // );
    // let _ = logger.run("smoothboost.csv");


    // // Run SoftBoost
    // let objective = SoftMarginObjective::new(nu);
    // println!("Running SoftBoost");
    // let booster = SoftBoost::init(&train)
    //     .tolerance(0.01)
    //     .nu(nu);
    // let tree = DTree::init(&train)
    //     .max_depth(2)
    //     .criterion(Criterion::Entropy);
    // let mut logger = Logger::new(
    //     booster, tree, objective, zero_one_loss, &train, &test
    // );
    // let _ = logger.run("softboost.csv");


    // // Run LPBoost
    // let objective = SoftMarginObjective::new(nu);
    // println!("Running LPBoost");
    // let booster = LPBoost::init(&train)
    //     .tolerance(0.01)
    //     .nu(nu);
    // let tree = DTree::init(&train)
    //     .max_depth(2)
    //     .criterion(Criterion::Entropy);
    // let mut logger = Logger::new(
    //     booster, tree, objective, zero_one_loss, &train, &test
    // );
    // let _ = logger.run("lpboost.csv");


    // // Run ERLPBoost
    // let objective = SoftMarginObjective::new(nu);
    // println!("Running ERLPBoost");
    // let booster = ERLPBoost::init(&train)
    //     .tolerance(0.01)
    //     .nu(nu);
    // let tree = DTree::init(&train)
    //     .max_depth(2)
    //     .criterion(Criterion::Entropy);
    // let mut logger = Logger::new(
    //     booster, tree, objective, zero_one_loss, &train, &test
    // );
    // let _ = logger.run("erlpboost.csv");


    // Run Corrective ERLPBoost
    let objective = SoftMarginObjective::new(nu);
    println!("Running CERLPBoost");
    let booster = CERLPBoost::init(&train)
        .tolerance(0.01)
        .nu(nu);
    let tree = DTree::init(&train)
        .max_depth(2)
        .criterion(Criterion::Entropy);
    let mut logger = Logger::new(
        booster, tree, objective, zero_one_loss, &train, &test
    );
    let _ = logger.run("cerlpboost.csv");


    // // Run MLPBoost
    // let objective = SoftMarginObjective::new(nu);
    // println!("Running MLPBoost");
    // let booster = MLPBoost::init(&train)
    //     .tolerance(0.01)
    //     .nu(nu);
    // let tree = DTree::init(&train)
    //     .max_depth(2)
    //     .criterion(Criterion::Entropy);
    // let mut logger = Logger::new(
    //     booster, tree, objective, zero_one_loss, &train, &test
    // );
    // let _ = logger.run("mlpboost.csv");
}

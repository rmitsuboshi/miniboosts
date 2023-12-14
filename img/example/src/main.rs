use miniboosts::prelude::*;
use miniboosts::research::{
    LoggerBuilder,
    Logger,
};
use miniboosts::{
    SoftMarginObjective,
    ExponentialLoss,
};

// use miniboosts::booster::perturbed_lpboost::PerturbedLPBoost;


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


const TOLERANCE: f64 = 0.001;
// const TIME_LIMIT: u128 = 300_000; // 5 minutes as millisecond.
const TIME_LIMIT: u128 = 60_000; // 1 minute as millisecond.


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
    // let nu = 1.0;

    let path = args.next()
        .expect("[USAGE] ./example [csv file (train)] [csv file (test)]");
    let test = Sample::from_csv(path, true)
        .unwrap()
        .set_target("class");


    // // Run AdaBoost
    // let objective = ExponentialLoss::new();
    // let booster = AdaBoost::init(&train)
    //     .tolerance(TOLERANCE);
    // let tree = DecisionTreeBuilder::new(&train)
    //     .max_depth(1)
    //     .criterion(Criterion::Entropy)
    //     .build();
    // let mut logger = Logger::new(
    //         booster, tree, objective, zero_one_loss, &train, &test
    //     ).time_limit_as_millis(TIME_LIMIT);
    // let _ = logger.run("adaboost.csv");


    // // Run AdaBoostV
    // let objective = ExponentialLoss::new();
    // let booster = AdaBoostV::init(&train)
    //     .tolerance(TOLERANCE);
    // let tree = DecisionTreeBuilder::new(&train)
    //     .max_depth(1)
    //     .criterion(Criterion::Entropy)
    //     .build();
    // let mut logger = Logger::new(
    //         booster, tree, objective, zero_one_loss, &train, &test
    //     ).time_limit_as_millis(TIME_LIMIT);
    // let _ = logger.run("adaboostv.csv");


    // // Run TotalBoost
    // let objective = HardMarginObjective::new();
    // let booster = TotalBoost::init(&train)
    //     .tolerance(TOLERANCE);
    // let tree = DecisionTreeBuilder::new(&train)
    //     .max_depth(1)
    //     .criterion(Criterion::Entropy)
    //     .build();
    // let time_limit = 2;
    // let mut logger = Logger::new(
    //         booster, tree, objective, zero_one_loss, &train, &test
    //     )
    //     .time_limit_as_secs(time_limit)
    //     .print_every(5);
    // let _ = logger.run("totalboost.csv");


    // // Run SmoothBoost
    // // `gamma` is the weak-learner guarantee.
    // // For this case, the following holds;
    // // for any distribution, the weak learner returns a hypothesis
    // // such that the edge is at least 0.006.
    // // This value `0.006` is derived from the LPBoost.
    // let objective = SoftMarginObjective::new(nu);
    // let booster = SmoothBoost::init(&train)
    //     .tolerance(TOLERANCE)
    //     .gamma(0.006);
    // let tree = DecisionTreeBuilder::new(&train)
    //     .max_depth(1)
    //     .criterion(Criterion::Entropy)
    //     .build();
    // let mut logger = Logger::new(
    //     booster, tree, objective, zero_one_loss, &train, &test
    // ).time_limit_as_millis(TIME_LIMIT);
    // let _ = logger.run("smoothboost.csv");


    // // Run SoftBoost
    // let objective = SoftMarginObjective::new(nu);
    // let booster = SoftBoost::init(&train)
    //     .tolerance(TOLERANCE)
    //     .nu(nu);
    // let tree = DecisionTreeBuilder::new(&train)
    //     .max_depth(1)
    //     .criterion(Criterion::Entropy)
    //     .build();
    // let mut logger = Logger::new(
    //     booster, tree, objective, zero_one_loss, &train, &test
    // ).time_limit_as_millis(TIME_LIMIT);
    // let _ = logger.run("softboost.csv");


    // Run LPBoost
    let objective = SoftMarginObjective::new(nu);
    let booster = LPBoost::init(&train)
        .tolerance(TOLERANCE)
        .nu(nu);
    let tree = DecisionTreeBuilder::new(&train)
        .max_depth(1)
        .criterion(Criterion::Entropy)
        .build();
    let time_limit = 10;
    let mut logger = Logger::new(
            booster, tree, objective, zero_one_loss, &train, &test
        )
        .time_limit_as_secs(time_limit)
        .print_every(10);
    let _ = logger.run("lpboost.csv");


    // // // Run Perturbed LPBoost
    // // let objective = SoftMarginObjective::new(nu);
    // // let booster = PerturbedLPBoost::init(&train)
    // //     .tolerance(TOLERANCE)
    // //     .nu(nu);
    // // let tree = DecisionTreeBuilder::new(&train)
    // //     .max_depth(1)
    // //     .criterion(Criterion::Entropy)
    // //     .build();
    // // let mut logger = Logger::new(
    // //     booster, tree, objective, zero_one_loss, &train, &test
    // // ).time_limit_as_millis(TIME_LIMIT);
    // // let _ = logger.run("stochastic-lpboost.csv");


    // Run ERLPBoost
    let objective = SoftMarginObjective::new(nu);
    let booster = ERLPBoost::init(&train)
        .tolerance(TOLERANCE)
        .nu(nu);
    let tree = DecisionTreeBuilder::new(&train)
        .max_depth(1)
        .criterion(Criterion::Entropy)
        .build();
    let mut logger = LoggerBuilder::new()
        .booster(booster)
        .weak_learner(tree)
        .train_sample(&train)
        .test_sample(&test)
        .objective_function(objective)
        .loss_function(zero_one_loss)
        .time_limit_as_millis(TIME_LIMIT)
        .print_every(10)
        .build();
    let _ = logger.run("erlpboost.csv");


    // Run MLPBoost
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


    // // Run Corrective ERLPBoost
    // let objective = SoftMarginObjective::new(nu);
    // let booster = CERLPBoost::init(&train)
    //     .tolerance(TOLERANCE)
    //     .fw_type(FWType::Classic)
    //     .nu(nu);
    // let tree = DecisionTreeBuilder::new(&train)
    //     .max_depth(1)
    //     .criterion(Criterion::Entropy)
    //     .build();
    // let mut logger = Logger::new(
    //         booster, tree, objective, zero_one_loss, &train, &test
    //     )
    //     .time_limit_as_millis(TIME_LIMIT)
    //     .print_every(100);
    // let _ = logger.run("cerlpboost.csv");


    // // Run Graph Separation Boosting
    // let objective = SoftMarginObjective::new(nu);
    // let booster = GraphSepBoost::init(&train)
    //     .tolerance(TOLERANCE);
    // let tree = DecisionTreeBuilder::new(&train)
    //     .max_depth(1)
    //     .criterion(Criterion::Entropy)
    //     .build();
    // let mut logger = Logger::new(
    //     booster, tree, objective, zero_one_loss, &train, &test
    // ).time_limit_as_millis(TIME_LIMIT);
    // let _ = logger.run("graphsepboost.csv");
}

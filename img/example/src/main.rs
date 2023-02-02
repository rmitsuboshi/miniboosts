use polars::prelude::*;
use miniboosts::prelude::*;
use miniboosts::research::{
    boost_logger::with_log,
    loss_functions::zero_one_loss,
};


fn main() {
    let mut args = std::env::args().skip(1);

    // Read the train file
    let path = args.next()
        .expect("[USAGE] ./example [csv file (train)] [csv file (test)]");
    let mut train_x = CsvReader::from_path(path)
        .unwrap()
        .has_header(true)
        .finish()
        .unwrap();
    let train_y = train_x.drop_in_place("class").unwrap();


    // Read the test file
    let path = args.next()
        .expect("[USAGE] ./example [csv file (train)] [csv file (test)]");
    let mut test_x = CsvReader::from_path(path)
        .unwrap()
        .has_header(true)
        .finish()
        .unwrap();
    let test_y = test_x.drop_in_place("class").unwrap();



    let n_sample = train_x.shape().0 as f64;


    // Run AdaBoost
    // let booster = AdaBoost::init(&train_x, &train_y)
    //     .tolerance(0.01);
    // let tree = DTree::init(&train_x, &train_y)
    //     .max_depth(2)
    //     .criterion(Criterion::Edge);
    // let _ = with_log(
    //     booster, tree, zero_one_loss, &test_x, &test_y, "adaboost.csv"
    // ).unwrap();


    // Run AdaBoostV
    // let booster = AdaBoostV::init(&train_x, &train_y)
    //     .tolerance(0.01);
    // let tree = DTree::init(&train_x, &train_y)
    //     .max_depth(2)
    //     .criterion(Criterion::Edge);
    // let _ = with_log(
    //     booster, tree, zero_one_loss, &test_x, &test_y, "adaboostv.csv"
    // ).unwrap();


    // Run TotalBoost
    // println!("Running TotalBoost");
    // let booster = TotalBoost::init(&train_x, &train_y)
    //     .tolerance(0.01);
    // let tree = DTree::init(&train_x, &train_y)
    //     .max_depth(2)
    //     .criterion(Criterion::Edge);
    // let _ = with_log(
    //     booster, tree, zero_one_loss, &test_x, &test_y, "totalboost.csv"
    // ).unwrap();


    // // Run SmoothBoost
    // // `gamma` is the weak-learner guarantee.
    // // For this case, the following holds;
    // // for any distribution, the weak learner returns a hypothesis
    // // such that the edge is at least 0.017.
    // // This value `0.017` is derived from the LPBoost.
    // println!("Running SmoothBoost");
    // let booster = SmoothBoost::init(&train_x, &train_y)
    //     .tolerance(0.01)
    //     .gamma(0.017);
    // let tree = DTree::init(&train_x, &train_y)
    //     .max_depth(2)
    //     .criterion(Criterion::Edge);
    // let _ = with_log(
    //     booster, tree, zero_one_loss, &test_x, &test_y, "smoothboost.csv"
    // ).unwrap();


    // // Run SoftBoost
    // println!("Running SoftBoost");
    // let booster = SoftBoost::init(&train_x, &train_y)
    //     .tolerance(0.01)
    //     .nu(0.1 * n_sample);
    // let tree = DTree::init(&train_x, &train_y)
    //     .max_depth(2)
    //     .criterion(Criterion::Edge);
    // let _ = with_log(
    //     booster, tree, zero_one_loss, &test_x, &test_y, "softboost.csv"
    // ).unwrap();


    // // Run LPBoost
    // println!("Running LPBoost");
    // let booster = LPBoost::init(&train_x, &train_y)
    //     .tolerance(0.01)
    //     .nu(0.1 * n_sample);
    // let tree = DTree::init(&train_x, &train_y)
    //     .max_depth(2)
    //     .criterion(Criterion::Edge);
    // let _ = with_log(
    //     booster, tree, zero_one_loss, &test_x, &test_y, "lpboost.csv"
    // ).unwrap();


    // // Run ERLPBoost
    // println!("Running ERLPBoost");
    // let booster = ERLPBoost::init(&train_x, &train_y)
    //     .tolerance(0.01)
    //     .nu(0.1 * n_sample);
    // let tree = DTree::init(&train_x, &train_y)
    //     .max_depth(2)
    //     .criterion(Criterion::Edge);
    // let _ = with_log(
    //     booster, tree, zero_one_loss, &test_x, &test_y, "erlpboost.csv"
    // ).unwrap();


    // Run Corrective ERLPBoost
    println!("Running CERLPBoost");
    let booster = CERLPBoost::init(&train_x, &train_y)
        .tolerance(0.01)
        .nu(0.1 * n_sample);
    let tree = DTree::init(&train_x, &train_y)
        .max_depth(2)
        .criterion(Criterion::Edge);
    let _ = with_log(
        booster, tree, zero_one_loss, &test_x, &test_y, "cerlpboost.csv"
    ).unwrap();


    // // Run MLPBoost
    // println!("Running MLPBoost");
    // let booster = MLPBoost::init(&train_x, &train_y)
    //     .tolerance(0.01)
    //     .nu(0.1 * n_sample);
    // let tree = DTree::init(&train_x, &train_y)
    //     .max_depth(2)
    //     .criterion(Criterion::Edge);
    // let _ = with_log(
    //     booster, tree, zero_one_loss, &test_x, &test_y, "mlpboost.csv"
    // ).unwrap();
}

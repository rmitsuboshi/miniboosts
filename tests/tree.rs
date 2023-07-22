use miniboosts::prelude::*;
use polars::prelude::*;

// Toy example  (o/x are the pos/neg examples)
// This partition is a decisiton tree for the unit prior.
// 
// 15|                     |
//   |                   5 |
//   |                  -  |
//   |                     |         6
//   |                     |        -
// 10|       4             |________________________ 9.5
//   |      -              |             1
//   |                     |            +
//   |                     |
//   |                     |   0
//  5|                     |  +
//   |                     |                 2
//   |                     |                +
//   |            3        |
//   |           -         |
//   |_____________________|____________________
//  0            5         | 10            15
//                         |
//                        9.0
// 
// 


#[test]
fn from_raw_representation() {
    let s1 = Series::new("x", &[10.0, 14.0, 15.0, 5.0, 3.0,  8.0, 12.0]);
    let s2 = Series::new("y", &[ 5.0,  8.0,  3.0, 1.0, 9.0, 13.0, 11.0]);
    let df = DataFrame::new(vec![s1, s2]).unwrap();
    let target = Series::new(
        "class", &[1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]
    );

    let sample = Sample::from_dataframe(df, target).unwrap();


    let dtree = DecisionTreeBuilder::new(&sample)
        .criterion(Criterion::Entropy)
        .build();
    let dist = vec![1.0/7.0; 7];
    let f = dtree.produce(&sample, &dist[..]);

    println!("SCRATCH:  {f:?}");
    let loss = f.predict_all(&sample)
        .into_iter()
        .zip(sample.target().into_iter())
        .map(|(p, y)| if p != *y as i64 { 1.0 } else { 0.0 })
        .sum::<f64>();
    println!("LOSS (Scratch): {loss}");
}


#[test]
fn from_lightsvm() {
    let mut path = std::env::current_dir().unwrap();
    path.push("tests/dataset/toy.svmlight");

    let mut sample = Sample::from_svmlight(path).unwrap();
    sample.replace_names(["x", "y"]);
    println!("{sample:?}");

    let dtree = DecisionTreeBuilder::new(&sample)
        .criterion(Criterion::Entropy)
        .build();
    let dist = vec![1.0/7.0; 7];
    let f = dtree.produce(&sample, &dist[..]);

    println!("SVMLight: {f:?}");
    let loss = f.predict_all(&sample)
        .into_iter()
        .zip(sample.target().into_iter())
        .map(|(p, y)| if p != *y as i64 { 1.0 } else { 0.0 })
        .sum::<f64>();
    println!("LOSS (SVMLight): {loss}");
}


#[test]
fn from_csv() {
    let mut path = std::env::current_dir().unwrap();
    path.push("tests/dataset/boston_housing.csv");

    let sample = Sample::from_csv(path, true)
        .unwrap()
        .set_target("MEDV");
    let n_sample = sample.shape().0;
    let tree = RegressionTreeBuilder::new(&sample)
        .loss(LossType::L2)
        .max_depth(5)
        .build();
    let dist = vec![1.0/n_sample as f64; n_sample];
    let f = tree.produce(&sample, &dist[..]);
    println!("{f:?}");
    f.to_dot_file("tests/output/result.dot").unwrap();
}

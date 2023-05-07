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


    let dtree = DTreeBuilder::new(&sample)
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

    let dtree = DTreeBuilder::new(&sample)
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

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


    let dtree = DTree::init(&sample)
        .criterion(Criterion::Edge);
    let dist = vec![1.0/7.0; 7];
    let f = dtree.produce(&sample, &dist[..]);

    println!("SCRATCH:  {f:?}");
}


#[test]
fn from_lightsvm() {
    let mut path = std::env::current_dir().unwrap();
    path.push("tests/dataset/toy.lightsvm");

    let sample = Sample::from_lightsvm(path).unwrap();

    let dtree = DTree::init(&sample)
        .criterion(Criterion::Entropy);
    let dist = vec![1.0/7.0; 7];
    let f = dtree.produce(&sample, &dist[..]);

    println!("LIGHTSVM: {f:?}");
}

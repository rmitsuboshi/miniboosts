use miniboosts::DTree;
use miniboosts::WeakLearner;


use polars::prelude::*;


// Toy example  (o/x are the pos/neg examples)
// This partition is a decisiton tree for the unit prior.
// 
// 15|                     |
//   |                   5 |
//   |                  x  |
//   |                     |         6
//   |                     |        x
// 10|       4             |________________________ 9.5
//   |      x              |             1
//   |                     |            o
//   |                     |
//   |                     |   0
//  5|                     |  o
//   |                     |                 2
//   |                     |                o
//   |            3        |
//   |           x         |
//   |_____________________|____________________
//  0            5         | 10            15
//                         |
//                        9.0
// 
// 


#[test]
fn full_binary_tree() {
    let s1 = Series::new("x", &[10.0, 14.0, 15.0, 5.0, 3.0, 8.0, 12.0]);
    let s2 = Series::new("y", &[5.0, 8.0, 3.0, 1.0, 9.0, 13.0, 11.0]);
    let target = Series::new("class", &[1_i64, 1, 1, -1, -1, -1, -1]);


    let df = DataFrame::new(vec![s1, s2]).unwrap();

    println!("{df}");


    let dtree = DTree::init(&df, &target);

    let dist = vec![1.0/7.0; 7];

    let f = dtree.produce(&df, &target, &dist[..]);

    println!("{f:?}");
}

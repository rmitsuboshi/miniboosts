extern crate lycaon;

use lycaon::Sample;
use lycaon::DTree;
use lycaon::BaseLearner;


// Toy example  (o/x are the pos/neg examples)
// 
// 15|
//   |
//   |                  x
//   |
//   |                              x
// 10|
//   |      x
//   |                                  o
//   |
//   |
//  5|                        o
//   |
//   |                                      o
//   |
//   |           x
//   |__________________________________________
//  0            5           10            15



#[test]
fn full_binary_tree() {
    let examples = vec![
        vec![10.0,  5.0],
        vec![14.0,  8.0],
        vec![15.0,  3.0],
        vec![ 5.0,  1.0],
        vec![ 3.0,  9.0],
        vec![ 8.0, 13.0],
        vec![12.0, 11.0],
    ];
    let labels: Vec<i32> = vec![1, 1, 1, -1, -1, -1, -1];

    let sample = Sample::from((examples, labels));


    let dtree = DTree::init(&sample);

    let dist = vec![1.0/7.0; 7];

    let f = dtree.best_hypothesis(&sample, &dist[..]);

    println!("{f:?}");
}

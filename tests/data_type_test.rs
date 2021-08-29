extern crate boost;

use boost::data_type;


#[test]
fn test() {
    let data = vec![
        vec![0.0, 1.0],
        vec![2.0, 1.0],
        vec![0.0, 9.0],
        vec![1.0, 6.0],
    ];
    let label = vec![1.0, -1.0, -1.0, -1.0];

    let sample = data_type::to_sample(data, label);

    let expected = vec![
        (vec![0.0, 1.0],  1.0),
        (vec![2.0, 1.0], -1.0),
        (vec![0.0, 9.0], -1.0),
        (vec![1.0, 6.0], -1.0),
    ];

    assert_eq!(sample, expected);
}

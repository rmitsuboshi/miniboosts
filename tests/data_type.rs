extern crate boost;

use boost::data_type::*;


#[test]
fn test() {
    let data = vec![
        Data::Dense(vec![0.0, 1.0]),
        Data::Dense(vec![2.0, 1.0]),
        Data::Dense(vec![0.0, 9.0]),
        Data::Dense(vec![1.0, 6.0]),
    ];
    let label = vec![
         1.0,
        -1.0,
        -1.0,
        -1.0
    ];

    let sample = to_sample(data, label);


    assert_eq!(sample[0].data.value_at(0), 0.0);
    assert_eq!(sample[1].data.value_at(0), 2.0);
    assert_eq!(sample[2].data.value_at(0), 0.0);
    assert_eq!(sample[3].data.value_at(0), 1.0);
    assert_eq!(sample[0].data.value_at(1), 1.0);
    assert_eq!(sample[1].data.value_at(1), 1.0);
    assert_eq!(sample[2].data.value_at(1), 9.0);
    assert_eq!(sample[3].data.value_at(1), 6.0);

    // let expected = vec![
    //     (Data::Dense(vec![0.0, 1.0]),  1.0),
    //     (Data::Dense(vec![2.0, 1.0]), -1.0),
    //     (Data::Dense(vec![0.0, 9.0]), -1.0),
    //     (Data::Dense(vec![1.0, 6.0]), -1.0),
    // ];

    // assert_eq!(sample, expected);
}

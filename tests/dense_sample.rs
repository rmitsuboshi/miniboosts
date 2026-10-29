use miniboosts::Sample;

#[test]
fn dense_sample_preserves_columns_and_allows_unlabeled_prediction() {
    let rows = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let sample = Sample::from_dense(&rows, Some(&[-1.0, 1.0])).unwrap();
    assert_eq!(sample.shape(), (2, 2));
    assert_eq!(sample.target(), &[-1.0, 1.0]);
    assert_eq!(sample.features()[1][0], 2.0);
    let unlabeled = Sample::from_dense(&rows, None).unwrap();
    assert!(unlabeled.target().is_empty());
    assert_eq!(unlabeled.features()[0][1], 3.0);
}

#[test]
fn dense_sample_rejects_invalid_shapes_and_nonfinite_values() {
    for rows in [
        vec![],
        vec![vec![]],
        vec![vec![1.], vec![1., 2.]],
        vec![vec![f64::NAN]],
    ] {
        assert!(Sample::from_dense(&rows, None).is_err());
    }
    assert!(Sample::from_dense(&[vec![1.]], Some(&[])).is_err());
    assert!(Sample::from_dense(&[vec![1.]], Some(&[f64::INFINITY])).is_err());
}

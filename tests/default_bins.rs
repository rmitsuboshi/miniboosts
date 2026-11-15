//! The tree learners' default bins follow the data: one bin per value when
//! the values fit, and bins of equal shares of the examples otherwise.

use miniboosts::{
    Classifier, DecisionTreeBuilder, RegressionTreeBuilder, Regressor, Sample, WeakLearner,
};

/// 600 examples whose single feature takes 600 distinct values in (-3, 3),
/// labelled by the sign of the value.
fn values() -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..600_u32)
        .map(|i| f64::from(i * 37 % 600) / 100.0 - 2.995)
        .collect();
    let y = x
        .iter()
        .map(|&v| if v > 0.0 { 1.0 } else { -1.0 })
        .collect();
    (x, y)
}

fn sample(x: &[f64], y: &[f64]) -> Sample {
    let rows: Vec<Vec<f64>> = x.iter().map(|&v| vec![v]).collect();
    Sample::from_dense(&rows, Some(y)).unwrap()
}

#[test]
fn trees_are_invariant_under_increasing_transformations() {
    // More distinct values than bins, so the bins are quantiles. They hold
    // the same examples for `x` and for `exp(3x)`, whose values span five
    // orders of magnitude, so the trees send every example the same way.
    let (x, y) = values();
    let stretched: Vec<f64> = x.iter().map(|&v| (3.0 * v).exp()).collect();
    let (plain, heavy) = (sample(&x, &y), sample(&stretched, &y));
    let dist = vec![1.0 / 600.0; 600];

    let fit = |s: &Sample| {
        DecisionTreeBuilder::new(s)
            .max_depth(3)
            .build()
            .produce(s, &dist)
    };
    assert_eq!(
        fit(&plain).confidence_all(&plain),
        fit(&heavy).confidence_all(&heavy)
    );

    let responses: Vec<f64> = x.iter().map(|&v| (2.0 * v).sin()).collect();
    let fit = |s: &Sample| {
        RegressionTreeBuilder::new(s)
            .max_depth(3)
            .build()
            .produce(&s.with_target(responses.clone()), &dist)
    };
    assert_eq!(
        fit(&plain).predict_all(&plain),
        fit(&heavy).predict_all(&heavy)
    );
}

#[test]
fn a_stump_finds_the_boundary_of_a_heavy_tailed_feature() {
    // Labels change at x = 0, i.e. at exp(3x) = 1, while the values reach
    // exp(9) ≈ 8100: 255 equal-width bins would put every value below 31
    // into the first bin. The quantile bins separate the classes up to the
    // width of one bin (600 / 255 < 3 examples).
    let (x, y) = values();
    let stretched: Vec<f64> = x.iter().map(|&v| (3.0 * v).exp()).collect();
    let heavy = sample(&stretched, &y);
    let dist = vec![1.0 / 600.0; 600];
    let stump = DecisionTreeBuilder::new(&heavy).max_depth(1).build();
    let wrong = stump
        .produce(&heavy, &dist)
        .predict_all(&heavy)
        .iter()
        .zip(&y)
        .filter(|&(&p, &y)| p as f64 != y)
        .count();
    assert!(wrong <= 2, "{wrong} errors");
}

#[test]
fn a_feature_with_few_values_gets_every_split() {
    // Unevenly spaced values: equal-width bins would merge 0 and 0.26.
    let x = [0.0, 0.0, 0.0, 0.26, 0.95, 1.09];
    let y = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
    let s = sample(&x, &y);
    let stump = DecisionTreeBuilder::new(&s).max_depth(1).build();
    let h = stump.produce(&s, &[1.0 / 6.0; 6]);
    assert_eq!(h.predict_all(&s), vec![-1, -1, -1, 1, 1, 1]);
}

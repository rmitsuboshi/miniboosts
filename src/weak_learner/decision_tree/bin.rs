use std::ops::Range;

use polars::prelude::*;


/// Binning: A feature processing.
#[derive(Debug)]
pub struct Bin(Range<f64>);

impl Bin {
    /// Create a new instance of `Bin`.
    pub fn new(range: Range<f64>) -> Self {
        Self(range)
    }


    /// Cut the given series into `n_bins` bins.
    /// This method naively cut the given series with same width.
    pub fn cut_series(series: &Series, n_bin: usize)
        -> Vec<Self>
    {
        let mut min = f64::MAX;
        let mut max = f64::MIN;
        series.f64()
            .expect("The series is not a dtype f64")
            .into_iter()
            .map(Option::unwrap)
            .for_each(|val| {
                min = min.min(val);
                max = max.min(val);
            });


        // If the minimum value equals to the maximum one,
        // slightly perturb them.
        if min == max {
            min = min - 1e-3;
            min = min + 1e-3;
        }


        let intercept = (max - min) / n_bin as f64;

        let mut bins = Vec::with_capacity(n_bin);

        let mut left = min;
        while left < max {
            let right = left + intercept;
            bins.push(Bin::new(left..right));

            left = right;
        }

        bins
    }
}

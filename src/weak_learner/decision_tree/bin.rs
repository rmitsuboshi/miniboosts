use std::ops::Range;

use crate::sample::{
    Feature,
    DenseFeature,
    SparseFeature,
};


const EPS: f64 = 0.01;


/// Binning: A feature processing.
#[derive(Debug)]
pub struct Bin(Range<f64>);

impl Bin {
    /// Create a new instance of `Bin`.
    pub fn new(range: Range<f64>) -> Self {
        Self(range)
    }


    /// Cut the given slice into `n_bins` bins.
    /// This method naively cut the given slice with same width.
    pub fn cut(feature: &Feature, n_bin: usize)
        -> Vec<Self>
    {
        match feature {
            Feature::Dense(feat) => {
                Bin::cut_dense(feat, n_bin)
            },
            Feature::Sparse(feat) => {
                Bin::cut_sparse(feat, n_bin)
            },
        }
    }


    /// Cut the given slice into `n_bins` bins.
    /// This method naively cut the given slice with same width.
    pub fn qcut(feature: &[f64], n_bin: usize)
        -> Vec<Self>
    {
        assert!(feature.len() >= n_bin);
        let mut feature = feature.to_vec();
        feature.sort_by(|a, b| a.partial_cmp(&b).unwrap());


        let n_sample = feature.len();
        let intercept = n_sample / n_bin;
        let mut right_idx = intercept;

        let mut bins = Vec::with_capacity(n_bin);


        let mut left = feature[0] - EPS;
        let mut k = 1;
        while k < n_sample {
            if k + 1 >= n_sample {
                let right = *feature.last().unwrap() + EPS;
                bins.push(Bin::new(left..right));
                break;
            }

            if k == right_idx {
                let right = (feature[right_idx-1] + feature[right_idx]) / 2.0;
                bins.push(Bin::new(left..right));
                left = feature[right_idx];
                right_idx += intercept;
            }
            k += 1;
        }

        assert_eq!(n_bin, bins.len());

        bins
    }


    fn cut_dense(feature: &DenseFeature, n_bin: usize)
        -> Vec<Self>
    {
        let mut min = f64::MAX;
        let mut max = f64::MIN;
        feature.sample[..]
            .iter()
            .copied()
            .for_each(|val| {
                min = min.min(val);
                max = max.max(val);
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


        assert_eq!(bins.len(), n_bin);

        bins
    }


    fn cut_sparse(feature: &SparseFeature, n_bin: usize)
        -> Vec<Self>
    {
        let mut min = f64::MAX;
        let mut max = f64::MIN;
        feature.sample[..]
            .into_iter()
            .copied()
            .for_each(|(_, val)| {
                min = min.min(val);
                max = max.max(val);
            });


        // If the minimum value equals to the maximum one,
        // slightly perturb them.
        if min == max {
            min = min - 1e-3;
            min = min + 1e-3;
        }


        if min > 0.0 && feature.has_zero() {
            min = 0.0;
        }


        if max < 0.0 && feature.has_zero() {
            max = 0.0;
        }


        let intercept = (max - min) / n_bin as f64;

        let mut bins = Vec::with_capacity(n_bin);

        let mut left = min;
        while left < max {
            let right = left + intercept;
            bins.push(Bin::new(left..right));

            left = right;
        }


        assert_eq!(bins.len(), n_bin);

        bins
    }


    // /// Cut the given slice into `n_bins` bins.
    // /// This method naively cut the given slice with same width.
    // pub fn qcut(feature: &[f64], n_bin: usize)
    //     -> Vec<Self>
    // {
    //     assert!(feature.len() >= n_bin);
    //     let mut feature = feature.to_vec();
    //     feature.sort_by(|a, b| a.partial_cmp(&b).unwrap());


    //     let n_sample = feature.len();
    //     let intercept = n_sample / n_bin;
    //     let mut right_idx = intercept;

    //     let mut bins = Vec::with_capacity(n_bin);


    //     let mut left = feature[0] - EPS;
    //     let mut k = 1;
    //     while k < n_sample {
    //         if k + 1 >= n_sample {
    //             let right = *feature.last().unwrap() + EPS;
    //             bins.push(Bin::new(left..right));
    //             break;
    //         }

    //         if k == right_idx {
    //             let right = (feature[right_idx-1] + feature[right_idx]) / 2.0;
    //             bins.push(Bin::new(left..right));
    //             left = feature[right_idx];
    //             right_idx += intercept;
    //         }
    //         k += 1;
    //     }

    //     assert_eq!(n_bin, bins.len());

    //     bins
    // }
}



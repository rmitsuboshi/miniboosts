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


    /// Cut the given `Feature` into `n_bins` bins.
    /// This method naively cut the given slice with same width.
    pub fn cut(feature: &Feature, n_bin: usize)
        -> Vec<Self>
    {
        match feature {
            Feature::Dense(feat) => {
                Self::cut_dense(feat, n_bin)
            },
            Feature::Sparse(feat) => {
                Self::cut_sparse(feat, n_bin)
            },
        }
    }


    /// Cut the given `Feature` into `n_bins` bins.
    /// This method returns a vector of `Bin`,
    /// where each `Bin` has almost the same elements.
    pub fn qcut(feature: &Feature, n_bin: usize)
        -> Vec<Self>
    {
        let items = value_counts(feature);

        if items.is_empty() {
            return Vec::with_capacity(0);
        }

        let n_feature = feature.len();
        let item_per_bin = n_feature / n_bin;

        let mut iter = items.into_iter()
            .peekable();

        let mut bins: Vec<Self> = Vec::with_capacity(n_bin);


        todo!()
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


fn value_counts(feature: &Feature) -> Vec<(f64, usize)> {
    match feature {
        Feature::Dense(feat) => value_counts_dense(feat),
        Feature::Sparse(feat) => value_counts_sparse(feat),
    }
}


fn value_counts_dense(feature: &DenseFeature) -> Vec<(f64, usize)> {
    let mut values = feature.sample[..].to_vec();

    let n_items = values.len();

    let mut items = Vec::with_capacity(n_items);

    inner_value_counts(values, &mut items);

    items.shrink_to_fit();

    items
}


fn value_counts_sparse(feature: &SparseFeature) -> Vec<(f64, usize)> {
    let mut items = if feature.has_zero() {
        Vec::with_capacity(feature.len() + 1)
    } else {
        Vec::with_capacity(feature.n_sample)
    };

    let mut values = feature.sample[..]
        .into_iter()
        .map(|(_, v)| *v)
        .collect::<Vec<_>>();

    inner_value_counts(values, &mut items);

    if feature.has_zero() {
        let count = feature.zero_counts();
        let pos = items.binary_search_by(|v| v.0.partial_cmp(&0.0).unwrap());
        match pos {
            Ok(_) => {
                panic!("Sparse feature contains zero value!");
            },
            Err(p) => {
                items.insert(p, (0.0, count));
            },
        }

    }
    items.shrink_to_fit();
    items
}


/// Count the number of items in `src` that has the same value.
/// The given vector `src` is assumed to be sorted in ascending order.
fn inner_value_counts(mut src: Vec<f64>, dst: &mut Vec<(f64, usize)>) {
    src.sort_by(|a, b| a.partial_cmp(&b).unwrap());
    let mut iter = src.into_iter();
    let mut value = match iter.next() {
        Some(v) => v,
        None => { return; }
    };

    let mut count: usize = 1;
    while let Some(v) = iter.next() {
        if v == value {
            count += 1;
        } else {
            dst.push((value, count));
            value = v;
            count = 1;
        }
    }

    dst.push((value, count));
}

use std::fmt;
use std::ops::Range;
use std::cmp::Ordering;


use crate::sample::{
    Feature,
    feature_struct::{
        DenseFeature,
        SparseFeature,
    },
};


const EPS: f64 = 0.001;
/// A tolerance parameter for numerical error.
/// This program ignores the difference smaller than this value.
const NUM_TOLERANCE: f64 = 1e-9;


/// A struct that stores the first/second order derivative information.
#[derive(Clone,Default)]
pub(crate) struct GradientHessian {
    pub(crate) grad: f64,
    pub(crate) hess: f64,
}


impl GradientHessian {
    pub(super) fn new(grad: f64, hess: f64) -> Self {
        Self { grad, hess }
    }
}


/// Binning: A feature processing.
#[derive(Debug)]
pub struct Bin(pub Range<f64>);

impl Bin {
    /// Create a new instance of `Bin`.
    #[inline(always)]
    pub fn new(range: Range<f64>) -> Self {
        Self(range)
    }


    /// Check whether the given `item` is conteined by `self.`
    #[inline(always)]
    pub fn contains(&self, item: &f64) -> bool {
        self.0.contains(item)
    }
}


/// A wrapper of `Vec<Bin>`.
pub struct Bins(Vec<Bin>);

impl Bins {
    /// Returns the number of bins.
    pub fn len(&self) -> usize {
        self.0.len()
    }


    /// Returns whether the bins are empty or not.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }


    /// Cut the given `Feature` into `n_bins` bins.
    /// This method naively cut the given slice with same width.
    #[inline(always)]
    pub fn cut(feature: &Feature, n_bin: usize) -> Self
    {
        let mut bins = match feature {
            Feature::Dense(feat) => Self::cut_dense(feat, n_bin),
            Feature::Sparse(feat) => Self::cut_sparse(feat, n_bin),
        };

        // The `start` of the left-most bin should be `f64::MIN`.
        bins.0.first_mut().unwrap().0.start = f64::MIN;
        // The `end` of the right-most bin should be `f64::MAX`.
        bins.0.last_mut().unwrap().0.end = f64::MAX;

        bins
    }


    // /// Cut the given `Feature` into `n_bins` bins.
    // /// This method returns a vector of `Bin`,
    // /// where each `Bin` has almost the same elements.
    // pub fn qcut(feature: &Feature, n_bin: usize)
    //     -> Vec<Self>
    // {
    //     let items = value_counts(feature);

    //     if items.is_empty() {
    //         return Vec::with_capacity(0);
    //     }

    //     let n_feature = feature.len();
    //     let item_per_bin = n_feature / n_bin;

    //     let mut iter = items.into_iter()
    //         .peekable();

    //     let mut bins: Vec<Self> = Vec::with_capacity(n_bin);


    //     todo!()
    // }


    fn cut_dense(feature: &DenseFeature, n_bin: usize) -> Self
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
            min = min - EPS;
            max = max + EPS;
        }


        let intercept = (max - min) / n_bin as f64;

        let mut bins = Vec::with_capacity(n_bin);

        let mut left = min;
        while left < max {
            let right = left + intercept;
            bins.push(Bin::new(left..right));

            // Numerical error leads an unexpected split.
            // So, we ignore the bin with width smaller than 1e-9.
            if (right - max).abs() < NUM_TOLERANCE { break; }

            left = right;
        }


        assert_eq!(bins.len(), n_bin);

        Self(bins)
    }


    fn cut_sparse(feature: &SparseFeature, n_bin: usize) -> Self
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


        if min > 0.0 && feature.has_zero() {
            min = 0.0;
        }


        if max < 0.0 && feature.has_zero() {
            max = 0.0;
        }


        // If the minimum value equals to the maximum one,
        // slightly perturb them.
        if min == max {
            min = min - EPS;
            max = max + EPS;
        }


        let intercept = (max - min) / n_bin as f64;

        let mut bins = Vec::with_capacity(n_bin);

        let mut left = min;
        while left < max {
            let right = left + intercept;
            bins.push(Bin::new(left..right));

            // Numerical error leads an unexpected split.
            // So, we ignore the bin with width smaller than 1e-9.
            if (right - max).abs() < NUM_TOLERANCE { break; }

            left = right;
        }


        assert_eq!(bins.len(), n_bin);

        Self(bins)
    }


    pub(crate) fn pack(
        &self,
        indices: &[usize],
        feat: &Feature,
        gh: &[GradientHessian],
    ) -> Vec<(Bin, GradientHessian)>
    {
        let n_bins = self.0.len();
        let mut packed = vec![GradientHessian::default(); n_bins];

        for &i in indices {
            let xi = feat[i];


            let pos = self.0.binary_search_by(|range| {
                    if range.contains(&xi) {
                        return Ordering::Equal;
                    }
                    range.0.start.partial_cmp(&xi).unwrap()
                })
                .unwrap();
            packed[pos].grad += gh[i].grad;
            packed[pos].hess += gh[i].hess;
        }
        self.remove_zero_weight_pack_and_normalize(packed)
    }



    /// This method removes bins with zero weights.
    /// # Example
    /// Assume that we have bins and its weight.
    /// ```text
    /// Bins       | [-3.0, 2.5), [2.5, 7.0), [7.0, 8.1), [8.1, 9.0)
    /// Weights(+) |     0.5,        0.0,        0.0,        0.2
    /// Weights(-) |     0.0,        0.0,        0.0,        0.1
    /// ```
    /// This method remove zero-weight bins and normalize the weights
    /// like this:
    /// ```text
    /// Bins       | [-3.0, 5.3), [5.3, 9.0)
    /// Weights(+) |     0.625,      0.25
    /// Weights(-) |     0.0,        0.125
    /// ```
    /// That is, this method
    /// - Change the bin bounds,
    /// - 
    fn remove_zero_weight_pack_and_normalize(
        &self,
        pack: Vec<GradientHessian>,
    ) -> Vec<(Bin, GradientHessian)>
    {
        let mut iter = self.0.iter().zip(pack);

        let (prev_bin, mut prev_gh) = iter.next().unwrap();

        let mut prev_bin = Bin::new(prev_bin.0.clone());
        let mut iter = iter.filter(|(_, gh)| {
            gh.grad != 0.0 || gh.hess != 0.0
        });

        // The left-most bin might have zero weight.
        // In this case, find the next non-zero weight bin and merge.
        if prev_gh.grad == 0.0 && prev_gh.hess == 0.0 {
            let (next_bin, next_gh) = iter.next().unwrap();

            let start = prev_bin.0.start;
            let end = next_bin.0.end;
            prev_bin = Bin::new(start..end);
            prev_gh = next_gh;
        }

        let mut bin_and_gh = Vec::new();
        for (next_bin, next_gh) in iter {
            let start = prev_bin.0.start;
            let end = (prev_bin.0.end + next_bin.0.start) / 2.0;
            let bin = Bin::new(start..end);
            bin_and_gh.push((bin, prev_gh));


            prev_bin = Bin::new(next_bin.0.clone());
            prev_bin.0.start = end;
            prev_gh = next_gh;
        }
        bin_and_gh.push((prev_bin, prev_gh));

        bin_and_gh
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


// fn value_counts(feature: &Feature) -> Vec<(f64, usize)> {
//     match feature {
//         Feature::Dense(feat) => value_counts_dense(feat),
//         Feature::Sparse(feat) => value_counts_sparse(feat),
//     }
// }
// 
// 
// fn value_counts_dense(feature: &DenseFeature) -> Vec<(f64, usize)> {
//     let mut values = feature.sample[..].to_vec();
// 
//     let n_items = values.len();
// 
//     let mut items = Vec::with_capacity(n_items);
// 
//     inner_value_counts(values, &mut items);
// 
//     items.shrink_to_fit();
// 
//     items
// }
// 
// 
// fn value_counts_sparse(feature: &SparseFeature) -> Vec<(f64, usize)> {
//     let mut items = if feature.has_zero() {
//         Vec::with_capacity(feature.len() + 1)
//     } else {
//         Vec::with_capacity(feature.n_sample)
//     };
// 
//     let mut values = feature.sample[..]
//         .into_iter()
//         .map(|(_, v)| *v)
//         .collect::<Vec<_>>();
// 
//     inner_value_counts(values, &mut items);
// 
//     if feature.has_zero() {
//         let count = feature.zero_counts();
//         let pos = items.binary_search_by(|v| v.0.partial_cmp(&0.0).unwrap());
//         match pos {
//             Ok(_) => {
//                 panic!("Sparse feature contains zero value!");
//             },
//             Err(p) => {
//                 items.insert(p, (0.0, count));
//             },
//         }
// 
//     }
//     items.shrink_to_fit();
//     items
// }
// 
// 
// /// Count the number of items in `src` that has the same value.
// /// The given vector `src` is assumed to be sorted in ascending order.
// fn inner_value_counts(mut src: Vec<f64>, dst: &mut Vec<(f64, usize)>) {
//     src.sort_by(|a, b| a.partial_cmp(&b).unwrap());
//     let mut iter = src.into_iter();
//     let mut value = match iter.next() {
//         Some(v) => v,
//         None => { return; }
//     };
// 
//     let mut count: usize = 1;
//     while let Some(v) = iter.next() {
//         if v == value {
//             count += 1;
//         } else {
//             dst.push((value, count));
//             value = v;
//             count = 1;
//         }
//     }
// 
//     dst.push((value, count));
// }


const PRINT_BIN_SIZE: usize = 3;

impl fmt::Display for Bins {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bins = &self.0;
        let n_bins = bins.len();
        if n_bins > PRINT_BIN_SIZE {
            let head = bins[..2].iter()
                .map(|bin| format!("{bin}"))
                .collect::<Vec<_>>()
                .join(", ");
            let tail = bins.last()
                .map(|bin| format!("{bin}"))
                .unwrap();
            write!(f, "{head},      ...     , {tail}")
        } else {
            let line = bins.iter()
                .map(|bin| format!("{}", bin))
                .collect::<Vec<_>>()
                .join(", ");
            write!(f, "{line}")
        }
    }
}

impl fmt::Display for Bin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let start = if self.0.start == f64::MIN {
            String::from("-Inf")
        } else {
            let start = self.0.start;
            let sgn = if start > 0.0 {
                '+'
            } else if start < 0.0 {
                '-'
            } else {
                ' '
            };
            let start = start.abs();
            format!("{sgn}{start: >.2}")
        };
        let end = if self.0.end == f64::MAX {
            String::from("+Inf")
        } else {
            let end = self.0.end;
            let sgn = if end > 0.0 {
                '+'
            } else if end < 0.0 {
                '-'
            } else {
                ' '
            };
            let end = end.abs();
            format!("{sgn}{end: >.2}")
        };

        write!(f, "[{start}, {end})")
    }
}

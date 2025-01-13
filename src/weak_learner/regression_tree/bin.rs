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


type Gradient = f64;
type Hessian  = f64;


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
        gradient: &[Gradient],
        hessian: &[Hessian],
    ) -> Vec<(Bin, Gradient, Hessian)>
    {
        let n_bins = self.0.len();
        let mut grad_pack = vec![0f64; n_bins];
        let mut hess_pack = vec![0f64; n_bins];

        for &i in indices {
            let xi = feat[i];


            let pos = self.0.binary_search_by(|range| {
                    if range.contains(&xi) {
                        return Ordering::Equal;
                    }
                    range.0.start.partial_cmp(&xi).unwrap()
                })
                .unwrap();
            grad_pack[pos] += gradient[i];
            hess_pack[pos] += hessian[i];
        }
        self.remove_zero_weight_pack_and_normalize(grad_pack, hess_pack)
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
        grad_pack: Vec<Gradient>,
        hess_pack: Vec<Hessian>,
    ) -> Vec<(Bin, Gradient, Hessian)>
    {
        let mut iter = self.0.iter().zip(grad_pack.into_iter().zip(hess_pack));

        let (prev_bin, (mut prev_grad, mut prev_hess)) = iter.next().unwrap();

        let mut prev_bin = Bin::new(prev_bin.0.clone());
        let mut iter = iter.filter(|(_, (grad, hess))| {
            *grad != 0.0 || *hess != 0.0
        });

        // The left-most bin might have zero weight.
        // In this case, find the next non-zero weight bin and merge.
        if prev_grad == 0.0 && prev_hess == 0.0 {
            let (next_bin, (next_grad, next_hess)) = iter.next().unwrap();

            let start = prev_bin.0.start;
            let end = next_bin.0.end;
            prev_bin = Bin::new(start..end);
            prev_grad = next_grad;
            prev_hess = next_hess;
        }

        let mut bin_and_gh = Vec::new();
        for (next_bin, (next_grad, next_hess)) in iter {
            let start = prev_bin.0.start;
            let end = (prev_bin.0.end + next_bin.0.start) / 2.0;
            let bin = Bin::new(start..end);
            bin_and_gh.push((bin, prev_grad, prev_hess));


            prev_bin = Bin::new(next_bin.0.clone());
            prev_bin.0.start = end;
            prev_grad = next_grad;
            prev_hess = next_hess;
        }
        bin_and_gh.push((prev_bin, prev_grad, prev_hess));

        bin_and_gh
    }
}


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
            write!(f, "{head}, ... , {tail}")
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
            format!("{sgn}{start: >.1}")
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
            format!("{sgn}{end: >.1}")
        };

        write!(f, "[{start: >8}, {end: >8})")
    }
}

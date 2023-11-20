use std::path::Path;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::collections::{HashMap, HashSet};
use std::ops::Index;
use std::mem;

use polars::prelude::*;
use rayon::prelude::*;
use super::feature_struct::*;


/// Struct `Sample` holds a batch sample with dense/sparse format.
/// # Example
/// The following code reads a CSV file and sets the column of name `"class"`
/// as the target values.
/// ```no_run
/// let path = "/path/to/dataset.csv";
/// let has_header = true;
/// let sample = Sample::from_csv(path, has_header)
///     .unwrap()
///     .set_target("class");
/// ```
#[derive(Debug)]
pub struct Sample {
    pub(super) name_to_index: HashMap<String, usize>,
    pub(super) features: Vec<Feature>,
    pub(super) target: Vec<f64>,
    pub(super) n_sample: usize,
    pub(super) n_feature: usize,
}


impl Sample {
    /// Convert [`DataFrame`] and [`Series`] to `Sample`.
    /// This method takes the ownership of the given pair of 
    /// `data` and `target`.
    pub fn from_dataframe(data: DataFrame, target: Series)
        -> io::Result<Self>
    {
        let (n_sample, n_feature) = data.shape();
        let target = target.f64()
            .expect("The target is not a dtype f64")
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .unwrap();

        let features = data.get_columns()
            .into_par_iter()
            .map(|series| 
                Feature::Dense(DenseFeature::from_series(series))
            )
            .collect::<Vec<_>>();

        let name_to_index = features.iter()
            .enumerate()
            .map(|(i, f)| (f.name().to_string(), i))
            .collect::<HashMap<_, _>>();

        let sample = Self {
            name_to_index, features, target, n_sample, n_feature,
        };
        Ok(sample)
    }


    /// Read a CSV format file to [`Sample`] type.
    /// This method returns `Err` if the file does not exist.
    /// 
    /// If the CSV file does not header row,
    /// this method assigns a default name for each column:
    /// `Feat. [0]`, `Feat. [1]`, ..., `Feat. [n]`.
    /// 
    /// **Do not forget** to call [`Sample::set_target`] to
    /// assign the class label.
    pub fn from_csv<P>(file: P, mut has_header: bool) -> io::Result<Self>
        where P: AsRef<Path>,
    {
        // Open the given `file`.
        let file = File::open(file)?;
        let mut lines = BufReader::new(file).lines();

        let mut features = Vec::new();
        if has_header {
            let line = lines.next().unwrap();
            features = line?.split(',')
                .map(DenseFeature::new)
                .collect::<Vec<_>>();
        }
        let mut n_sample = 0_usize;

        // For each line of the file
        for line in lines {
            // Split the line by white spaces
            let line = line?;

            // if the headeer does not exists,
            // construct a dummy header.
            if !has_header {
                let xs = line.split(',')
                    .map(|x| x.trim().parse::<f64>().unwrap())
                    .collect::<Vec<_>>();

                let n_feature = xs.len();
                features = (1..=n_feature).map(|i| {
                        let name = format!("Feat. [{i}]");
                        DenseFeature::new(name)
                    })
                    .collect::<Vec<_>>();

                for (feat, x) in features.iter_mut().zip(xs) {
                    feat.append(x);
                }

                has_header = true;
                n_sample += 1;
                continue;
            }

            line.split(',')
                .map(|x| x.trim().parse::<f64>().unwrap())
                .enumerate()
                .for_each(|(i, x)| {
                    features[i].append(x);
                });

            n_sample += 1;
        }

        let features = features.into_par_iter()
            .map(Feature::Dense)
            .collect::<Vec<_>>();

        let n_feature = features.len();
        let target = Vec::with_capacity(0);

        let name_to_index = features.iter()
            .enumerate()
            .map(|(i, f)| (f.name().to_string(), i))
            .collect::<HashMap<_, _>>();

        let sample = Self {
            name_to_index, features, target, n_sample, n_feature,
        };

        Ok(sample)
    }


    /// Returns the slice of target values.
    pub fn target(&self) -> &[f64] {
        &self.target[..]
    }


    /// Returns the unique target values.
    pub fn unique_target(&self) -> Vec<f64> {
        let mut target = self.target().to_vec();
        target.sort_by(|a, b| a.partial_cmp(&b).unwrap());

        target.dedup();
        target
    }


    /// Returns a slice of the features.
    pub fn features(&self) -> &[Feature] {
        &self.features[..]
    }


    /// Set the feature of name `target` to `self.target`.
    /// The old value assigned to `self.target` will be dropped.
    pub fn set_target<S: AsRef<str>>(mut self, target: S) -> Self {
        let target = target.as_ref();
        let pos = self.features.iter()
            .position(|feat| feat.name() == target)
            .expect("The target class does not exist");


        let target = self.features.remove(pos).into_target();
        self.target = target;
        self.n_feature -= 1;


        self.name_to_index = self.features.iter()
            .enumerate()
            .map(|(i, f)| (f.name().to_string(), i))
            .collect::<HashMap<_, _>>();

        self
    }


    /// Read a SVMLight format file to `Sample`.
    /// 
    /// Each line of SVMLight format file has the following form:
    /// ```txt
    /// y index:value index: value
    /// ```
    /// where `y` is the target label of type `f64`,
    /// `index` is the feature index, and `value` is the value
    /// at the feature.
    /// 
    /// **Note**
    /// The SVMLight format file is basically 1-indexed,
    /// while the `sklearn.datasets.dump_svmlight_file` outputs
    /// a svmlight format file with 0-indexed, by default.
    pub fn from_svmlight<P: AsRef<Path>>(file: P) -> io::Result<Self> {
        let mut features = Vec::new();
        let mut target = Vec::new();
        let mut n_sample = 0_usize;

        // Open the given `file`.
        let file = File::open(file)?;
        let lines = BufReader::new(file).lines();

        // For each line of the file
        for line in lines {
            // Split the line by white spaces
            let line = line?;
            let mut words = line.split_whitespace();
            // The first word corresponds to the target value.
            let y = words.next()
                .unwrap()
                .trim()
                .parse::<f64>()
                .expect("Failed to parse the target value.");
            target.push(y);

            for word in words {
                let (i, x) = index_and_feature(word);

                while features.len() <= i {
                    let k = features.len() + 1;
                    let name = format!("Feat. [{k}]");
                    features.push(SparseFeature::new(name));
                }

                features[i].append((n_sample, x));
            }
            n_sample += 1;
        }

        let n_feature = features.len();


        let features = features.into_iter()
            .map(|mut feat| {
                feat.n_sample = n_sample;
                Feature::Sparse(feat)
            })
            .collect::<Vec<_>>();

        let name_to_index = features.iter()
            .enumerate()
            .map(|(i, f)| (f.name().to_string(), i))
            .collect::<HashMap<_, _>>();

        let mut sample = Self {
            name_to_index, features, target, n_sample, n_feature,
        };

        sample.remove_allzero_features();

        Ok(sample)
    }


    /// Removes the empty features in `self.features`.
    fn remove_allzero_features(&mut self) {
        let features = mem::take(&mut self.features);
        self.name_to_index = features.iter()
            .filter_map(|feat| {
                if feat.is_empty() {
                    None
                } else {
                    Some(feat.name().to_string())
                }
            })
            .enumerate()
            .map(|(i, name)| (name, i))
            .collect();
        self.features = features.into_iter()
            .filter(|feat| !feat.is_empty())
            .collect();
        self.n_feature = self.features.len();
    }


    /// Returns the pair of the number of examples and
    /// the number of features
    pub fn shape(&self) -> (usize, usize) {
        (self.n_sample, self.n_feature)
    }


    /// Set the feature (column) names.
    /// This method panics when the length of given feature names is
    /// not equals to the one of `self.features`.
    pub fn replace_names<S, T>(&mut self, names: T) -> Vec<String>
        where S: ToString + std::fmt::Display,
              T: AsRef<[S]>,
    {
        let names = names.as_ref();

        let n_features = self.shape().1;
        let n_names = names.len();
        assert_eq!(
            n_names, n_features,
            "The number of names is \
            not equal to the one of `self.features.`"
        );

        let old_names = names.iter()
            .zip(&mut self.features[..])
            .map(|(name, feature)| feature.replace_name(name))
            .collect();

        self.name_to_index = self.features.iter()
            .map(|feature| feature.name().to_string())
            .enumerate()
            .map(|(i, name)| (name, i))
            .collect();
        old_names
    }


    /// Returns the `idx`-th instance `(x, y)`.
    pub fn at(&self, idx: usize) -> (Vec<f64>, f64) {
        let x = self.features.iter()
            .map(|feat| feat[idx])
            .collect::<Vec<f64>>();
        let y = self.target[idx];

        (x, y)
    }


    fn target_is_specified(&self) {
        let n_sample = self.shape().0;

        if n_sample != self.target.len() {
            panic!(
                "The target class is not specified.\n\
                 Use `Sample::set_target(\"Column Name\")`."
            );
        }
    }


    /// Check whether `self` is 
    /// a training set for binary classification or not.
    pub fn is_valid_binary_instance(&self) {
        // Check whether the target column is specified.
        self.target_is_specified();


        // Check whether the target values can be converted into integers.
        let non_integers = self.target.iter()
            .filter(|yi| !yi.trunc().eq(yi))
            .collect::<Vec<_>>();
        if !non_integers.is_empty() {
            let line = non_integers.iter().take(5)
                .map(|yi| yi.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            panic!(
                "Target values are non-integer types.\n\
                 Ex. [{line}, ...]."
            );
        }


        // Check whether the target values takes exactly 2 kinds.
        let set = self.target.iter()
            .copied()
            .map(|yi| yi as i32)
            .collect::<HashSet<_>>();
        let n_label = set.len();
        if n_label > 2 {
            panic!(
                "The target values take more than 2 kinds. \
                 Expected 2 kinds, got {n_label} kinds."
            );
        } else if n_label < 2 {
            panic!(
                "The target values take less than 2 kinds. \
                 Expected 2 kinds, got {n_label} kinds."
            );
        }


        // Check whether the target values takes +1 or -1.
        let is_pm = set.iter().all(|y| y.eq(&1) || y.eq(&-1));
        if !is_pm {
            let line = set.iter()
                .map(|y| y.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            println!(
                "Warning: the target values take values not in [-1.0, 1.0].\n\
                 Currently, the labels are: [{line}]."
            );
        }


        // At this point, all tests are passed
        // so that the sample is valid one for binary classification.
    }


    /// Computes the weighted mean and variance
    /// for each feature.
    ///
    /// The weight vector must be a probability vector
    /// (A vector consists of non-negative entries whose sum is `1`).
    pub fn weighted_mean_and_variance<T>(&self, weight: T)
        -> Vec<(f64, f64)>
        where T: AsRef<[f64]>
    {
        let weight = weight.as_ref();
        self.features()
            .par_iter()
            .map(|feat| feat.weighted_mean_and_variance(weight))
            .collect()
    }



    /// Compute the weighted mean of each feature.
    ///
    /// The weight vector must be a probability vector
    /// (A vector consists of non-negative entries whose sum is `1`).
    pub fn weighted_mean<T>(&self, weight: T) -> Vec<f64>
        where T: AsRef<[f64]>
    {
        let weight = weight.as_ref();
        self.features()
            .par_iter()
            .map(|feat| feat.weighted_mean(weight))
            .collect()
    }


    /// Compute the weighted mean of each feature
    /// whose label (target) is `y.`
    ///
    /// The weight vector must be a non-negative vector.
    pub fn weighted_mean_for_label<T>(
        &self,
        y: f64,
        weight: T
    ) -> Vec<f64>
        where T: AsRef<[f64]>
    {
        let weight = weight.as_ref();
        let target = self.target();
        self.features()
            .par_iter()
            .map(|feat|
                feat.weighted_mean_for_label(y, target, weight)
            )
            .collect()
    }


    /// Compute the weighted mean and variance of each feature
    /// whose label (target) is `y.`
    ///
    /// The weight vector must be a non-negative vector.
    pub fn weighted_mean_and_variance_for_label<T>(
        &self,
        y: f64,
        weight: T
    ) -> Vec<(f64, f64)>
        where T: AsRef<[f64]>
    {
        let weight = weight.as_ref();
        let target = self.target();
        self.features()
            .par_iter()
            .map(|feat|
                feat.weighted_mean_and_variance_for_label(y, target, weight)
            )
            .collect()
    }
}


/// Parse the following type of `str` to the pair of `(usize, f64)`.
/// `index:value`, where `index: usize` and `value: f64`.
pub(self) fn index_and_feature(word: &str) -> (usize, f64) {
    let mut i_x = word.split(':');
    let i = i_x.next()
        .unwrap()
        .trim()
        .parse::<usize>()
        .expect("Failed to parse an index.");
    let x = i_x.next()
        .unwrap()
        .trim()
        .parse::<f64>()
        .expect("Failed to parse a feature value.");

    (i, x)
}



impl<S> Index<S> for Sample
    where S: AsRef<str>
{
    type Output = Feature;


    fn index(&self, name: S) -> &Self::Output {
        let name: &str = name.as_ref();
        let k = *self.name_to_index.get(name).unwrap();
        &self.features[k]
    }
}

//! The [`Sample`] struct: a column-oriented set of examples with labels.
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::mem;
use std::ops::Index;
use std::path::Path;

use super::feature::*;
use rayon::prelude::*;

/// A column-oriented set of `m` examples with `n` features and a label vector.
///
/// Each column is a [`Feature`] (dense or sparse) addressed by name, e.g.,
/// `sample["x"][i]`. The label vector is [`Sample::target`]. Boosters in this
/// crate solve binary classification and require labels `-1` and `+1`
/// stored as `f64`, with both classes present in the training sample.
///
/// # Construction
/// - [`Sample::from_reader`]: CSV text from any reader.
/// - [`Sample::from_dense`]: a dense row-major matrix.
/// - [`crate::SampleReader`]: CSV or SVMlight files.
///
/// # File formats
/// **CSV.** Comma-separated values, one example per line. With a header, the
/// first line gives column names (used verbatim, without trimming). Without a
/// header, columns are named `Feat. [1]`, ..., `Feat. [n]` (1-based). Values
/// are trimmed and parsed as `f64`. The label column must then be selected
/// with [`Sample::set_target`].
///
/// **SVMlight.** Each line is
/// ```txt
/// y index:value index:value ...
/// ```
/// where `y` is the label (parsed as `f64`), `index` is a nonnegative integer
/// feature index, and `value` is an `f64`. Omitted entries are zero and
/// features are stored sparsely. Index `k` is named `Feat. [k+1]`, and
/// features with no nonzero entry (e.g., index `0` of a 1-indexed file) are
/// removed, so column positions may differ from the file's indices.
/// The SVMlight format is conventionally 1-indexed, whereas
/// `sklearn.datasets.dump_svmlight_file` writes 0-indexed files by default.
/// Comments (`#`), `qid:` fields, and blank lines are not supported.
///
/// # Examples
/// Read CSV text, select the label column, and access features by name.
/// ```
/// use std::io::BufReader;
/// use miniboosts::Sample;
///
/// let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
/// let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
///     .unwrap()
///     .set_target("class");
///
/// // 6 examples, 2 features; the label column is not a feature.
/// assert_eq!(sample.shape(), (6, 2));
/// assert_eq!(sample.target(), &[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]);
///
/// // `sample[name]` is a `Feature` column; `sample[name][i]` is a value.
/// assert_eq!(sample["x"][4], 2.0);
/// assert_eq!(sample["y"].name(), "y");
/// let names: Vec<&str> = sample.features().iter().map(|f| f.name()).collect();
/// assert_eq!(names, ["x", "y"]);
///
/// // `at(i)` returns one example as `(features in column order, label)`.
/// assert_eq!(sample.at(1), (vec![-1.0, 1.0], -1.0));
/// ```
///
/// Hold out a test set with [`Sample::split`], train on the rest, and
/// evaluate the accuracy on the held-out examples.
/// ```
/// # use std::io::BufReader;
/// use miniboosts::prelude::*;
/// use miniboosts::Sample;
///
/// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
/// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
/// #     .unwrap()
/// #     .set_target("class");
/// // Rows 2 and 5 (one per class) form the test set.
/// let (train, test) = sample.split([2, 5, 0, 1, 3, 4], 0, 2);
/// assert_eq!((train.shape().0, test.shape().0), (4, 2));
///
/// let tree = DecisionTreeBuilder::new(&train).max_depth(1).build();
/// let f = AdaBoost::init(&train).run(&tree);
///
/// // The data are separable by a threshold on `x`, which the stump finds.
/// let predictions = f.predict_all(&test);
/// let n_correct = predictions
///     .iter()
///     .zip(test.target())
///     .filter(|&(&p, &y)| p as f64 == y)
///     .count();
/// assert_eq!(n_correct, test.shape().0);
/// ```
#[derive(Debug, Clone)]
pub struct Sample {
    pub(super) name_to_index: HashMap<String, usize>,
    pub(super) features: Vec<Feature>,
    pub(super) target: Vec<f64>,
    pub(super) n_sample: usize,
    pub(super) n_feature: usize,
}

impl Sample {
    /// Constructs owned, column-oriented data from a dense row-major matrix
    /// `rows` (`rows[i][j]` is feature `j` of example `i`).
    /// Feature names are `feature_0`, `feature_1`, and so on (0-based).
    /// Pass `None` as `target` for prediction data without labels. This does
    /// not validate a task-specific label encoding; boosters still enforce
    /// their training requirements.
    ///
    /// # Errors
    /// Returns an error for empty or ragged matrices, non-finite entries, or
    /// a target that is non-finite or whose length differs from the number of
    /// rows.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::Sample;
    ///
    /// let rows = vec![vec![-1.0, 0.5], vec![2.0, 0.0], vec![3.0, -0.5]];
    /// let sample = Sample::from_dense(&rows, Some(&[-1.0, 1.0, 1.0])).unwrap();
    /// assert_eq!(sample.shape(), (3, 2));
    /// assert_eq!(sample["feature_1"][2], -0.5);
    /// assert_eq!(sample.target(), &[-1.0, 1.0, 1.0]);
    ///
    /// // Without labels, e.g. for data to predict on.
    /// let unlabeled = Sample::from_dense(&rows, None).unwrap();
    /// assert!(unlabeled.target().is_empty());
    ///
    /// // Ragged rows and non-finite values are rejected.
    /// assert!(Sample::from_dense(&[vec![1.0], vec![1.0, 2.0]], None).is_err());
    /// assert!(Sample::from_dense(&[vec![f64::NAN]], None).is_err());
    /// ```
    pub fn from_dense(rows: &[Vec<f64>], target: Option<&[f64]>) -> Result<Self, String> {
        let n_sample = rows.len();
        let n_feature = rows.first().map_or(0, Vec::len);
        if n_sample == 0 || n_feature == 0 {
            return Err("dense data must contain at least one row and one feature".into());
        }
        for (row, values) in rows.iter().enumerate() {
            if values.len() != n_feature {
                return Err(format!("row {row} has a different number of features"));
            }
            if values.iter().any(|v| !v.is_finite()) {
                return Err(format!("row {row} contains a non-finite feature"));
            }
        }
        if let Some(y) = target {
            if y.len() != n_sample || y.iter().any(|v| !v.is_finite()) {
                return Err("target must be finite and have one value per row".into());
            }
        }
        let features = (0..n_feature)
            .map(|column| Feature::Dense {
                name: format!("feature_{column}"),
                vals: rows.iter().map(|row| row[column]).collect(),
            })
            .collect::<Vec<_>>();
        let name_to_index = features
            .iter()
            .enumerate()
            .map(|(index, feature)| (feature.name().to_owned(), index))
            .collect();
        Ok(Self {
            name_to_index,
            features,
            target: target.unwrap_or_default().to_vec(),
            n_sample,
            n_feature,
        })
    }

    /// Constructs a placeholder sample with `n_sample` examples and a single
    /// all-zero feature named `dummy`, intended for tests of boosters and
    /// solvers whose hypotheses ignore the features.
    /// The first `n_sample / 2` (rounded down) labels are `+1` and the rest
    /// are `-1`, so `dummy(1)` has the single label `-1`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::Sample;
    ///
    /// let sample = Sample::dummy(5);
    /// assert_eq!(sample.shape(), (5, 1));
    /// assert_eq!(sample.target(), &[1.0, 1.0, -1.0, -1.0, -1.0]);
    /// assert_eq!(sample["dummy"][3], 0.0);
    /// ```
    pub fn dummy(n_sample: usize) -> Self {
        let half = n_sample / 2;
        let mut target = vec![1f64; n_sample];
        target[half..].iter_mut().for_each(|y| {
            *y = -1f64;
        });
        let features = vec![Feature::sparse("dummy", 0)];
        Self {
            name_to_index: HashMap::from([("dummy".to_string(), 0)]),
            features,
            target,
            n_sample,
            n_feature: 1usize,
        }
    }

    /// Reads a CSV file into a [`Sample`]; see [`Sample::from_reader`].
    /// Returns `Err` if the file cannot be opened or read.
    ///
    /// If the CSV file has no header row,
    /// this method assigns a default name to each column:
    /// `Feat. [1]`, `Feat. [2]`, ..., `Feat. [n]`.
    ///
    /// **Do not forget** to call [`Sample::set_target`] to
    /// assign the class label.
    pub(crate) fn from_csv<P>(file: P, has_header: bool) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
        // Open the given `file`.
        let file = File::open(file)?;
        let reader = BufReader::new(file);
        Self::from_reader(reader, has_header)
    }

    /// Reads CSV text from `reader`. The result has no labels yet: call
    /// [`Sample::set_target`] to turn one column into the label vector.
    ///
    /// If `has_header` is `true`, the first line gives the column names
    /// (split on `,` and used verbatim, so spaces after commas become part
    /// of the names). Otherwise columns are named
    /// `Feat. [1]`, ..., `Feat. [n]`. Each value is trimmed and parsed as
    /// `f64`; all columns are stored densely.
    ///
    /// # Errors
    /// Returns `Err` if reading a line fails (I/O error or invalid UTF-8).
    ///
    /// # Panics
    /// - if `has_header` is `true` and the input is empty;
    /// - if a value cannot be parsed as `f64` (including empty fields and
    ///   blank lines);
    /// - if a row has more columns than the header (or the first row).
    ///
    /// Rows with fewer columns are not rejected and leave columns with
    /// different lengths; callers must supply rectangular data.
    ///
    /// # Examples
    /// With a header row, the column names come from the first line.
    /// `&[u8]` implements [`Read`], so in-memory text works as well as a
    /// [`File`].
    /// ```
    /// use std::io::BufReader;
    /// use miniboosts::Sample;
    ///
    /// let csv = b"x,y,class\n-2,0,-1\n1,0.5,1\n";
    /// let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    ///     .unwrap()
    ///     .set_target("class");
    /// assert_eq!(sample.shape(), (2, 2));
    /// assert_eq!(sample["y"][1], 0.5);
    /// ```
    ///
    /// Without a header, columns are named `Feat. [1]`, `Feat. [2]`, ...
    /// (1-based), so the label column is selected by such a name.
    /// ```
    /// use std::io::BufReader;
    /// use miniboosts::Sample;
    ///
    /// let csv = b"-2,0,-1\n1,0.5,1\n";
    /// let sample = Sample::from_reader(BufReader::new(&csv[..]), false)
    ///     .unwrap()
    ///     .set_target("Feat. [3]");
    /// assert_eq!(sample.shape(), (2, 2));
    /// assert_eq!(sample.target(), &[-1.0, 1.0]);
    /// assert_eq!(sample["Feat. [1]"][0], -2.0);
    /// ```
    pub fn from_reader<R>(reader: BufReader<R>, mut has_header: bool) -> io::Result<Self>
    where
        R: Read,
    {
        let mut lines = reader.lines();

        let mut features = Vec::new();
        if has_header {
            let line = lines.next().unwrap();
            features = line?.split(',').map(Feature::dense).collect::<Vec<_>>();
        }
        let mut n_sample = 0_usize;

        // For each line of the file
        for (i, line) in lines.enumerate() {
            // Split the line by white spaces
            let line = line?;

            // if the headeer does not exists,
            // construct a dummy header.
            if !has_header {
                let xs = line
                    .split(',')
                    .map(|x| {
                        x.trim().parse::<f64>().unwrap_or_else(|_| {
                            panic!(
                                "The file contains non-numerical value. \
                                    Got {x} in Line {i}"
                            )
                        })
                    })
                    .collect::<Vec<_>>();

                let n_feature = xs.len();
                features = (1..=n_feature)
                    .map(|i| {
                        let name = format!("Feat. [{i}]");
                        Feature::dense(name)
                    })
                    .collect::<Vec<_>>();

                for (feat, x) in features.iter_mut().zip(xs) {
                    feat.append((0, x));
                }

                has_header = true;
                n_sample += 1;
                continue;
            }

            line.split(',')
                .map(|x| x.trim().parse::<f64>().unwrap())
                .enumerate()
                .for_each(|(i, x)| {
                    features[i].append((0, x));
                });

            n_sample += 1;
        }

        let n_feature = features.len();
        let target = Vec::with_capacity(0);

        let name_to_index = features
            .iter()
            .enumerate()
            .map(|(i, f)| (f.name().to_string(), i))
            .collect::<HashMap<_, _>>();

        let sample = Self {
            name_to_index,
            features,
            target,
            n_sample,
            n_feature,
        };

        Ok(sample)
    }

    /// Returns the label vector (one value per example).
    /// It is empty until [`Sample::set_target`] is called on data read
    /// from CSV.
    pub fn target(&self) -> &[f64] {
        &self.target[..]
    }

    /// Returns the distinct label values in ascending order.
    ///
    /// # Panics
    /// Panics if a label is NaN.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::Sample;
    ///
    /// let rows = vec![vec![0.0]; 4];
    /// let sample = Sample::from_dense(&rows, Some(&[1.0, -1.0, 1.0, -1.0])).unwrap();
    /// assert_eq!(sample.unique_target(), vec![-1.0, 1.0]);
    /// ```
    pub fn unique_target(&self) -> Vec<f64> {
        let mut target = self.target().to_vec();
        target.sort_by(|a, b| a.partial_cmp(b).unwrap());

        target.dedup();
        target
    }

    /// Returns the feature columns, in column order (the label column is not
    /// included).
    pub fn features(&self) -> &[Feature] {
        &self.features[..]
    }

    /// Moves the column named `target` out of the features and into the label
    /// vector, replacing any previous labels. The number of features
    /// decreases by one.
    ///
    /// # Panics
    /// Panics if no column is named `target`.
    ///
    /// # Examples
    /// ```
    /// use std::io::BufReader;
    /// use miniboosts::Sample;
    ///
    /// let csv = b"class,x,y\n-1,-2,0\n1,1,0.5\n";
    /// let sample = Sample::from_reader(BufReader::new(&csv[..]), true).unwrap();
    /// // Before `set_target`, every column is a feature and there are no labels.
    /// assert_eq!(sample.shape(), (2, 3));
    /// assert!(sample.target().is_empty());
    ///
    /// // The label column may be anywhere; it is removed from the features.
    /// let sample = sample.set_target("class");
    /// assert_eq!(sample.shape(), (2, 2));
    /// assert_eq!(sample.target(), &[-1.0, 1.0]);
    /// assert_eq!(sample.at(0), (vec![-2.0, 0.0], -1.0));
    /// ```
    pub fn set_target<S: AsRef<str>>(mut self, target: S) -> Self {
        let target = target.as_ref();
        let pos = self
            .features
            .iter()
            .position(|feat| feat.name() == target)
            .unwrap_or_else(|| panic!("The target class \"{target}\" does not exist"));

        let target = self.features.remove(pos).into_vals();
        self.target = target;
        self.n_feature -= 1;

        self.name_to_index = self
            .features
            .iter()
            .enumerate()
            .map(|(i, f)| (f.name().to_string(), i))
            .collect::<HashMap<_, _>>();

        self
    }

    /// Reads an SVMlight file into a [`Sample`] with sparse features.
    ///
    /// Each line of an SVMlight file has the following form:
    /// ```txt
    /// y index:value index:value
    /// ```
    /// where `y` is the target label of type `f64`,
    /// `index` is the feature index, and `value` is the value
    /// at the feature. Index `k` becomes a feature named `Feat. [k+1]`;
    /// all-zero features are removed afterwards.
    ///
    /// **Note**
    /// The SVMLight format file is basically 1-indexed,
    /// while the `sklearn.datasets.dump_svmlight_file` outputs
    /// a svmlight format file with 0-indexed, by default.
    pub(super) fn from_svmlight<P: AsRef<Path>>(file: P) -> io::Result<Self> {
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
            let y = words
                .next()
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
                    features.push(Feature::sparse(name, 0));
                }

                features[i].append((n_sample, x));
            }
            n_sample += 1;
        }

        let n_feature = features.len();

        features.iter_mut().for_each(|feat| {
            feat.set_size(n_sample);
        });

        let name_to_index = features
            .iter()
            .enumerate()
            .map(|(i, f)| (f.name().to_string(), i))
            .collect::<HashMap<_, _>>();

        let mut sample = Self {
            name_to_index,
            features,
            target,
            n_sample,
            n_feature,
        };

        sample.remove_allzero_features();

        Ok(sample)
    }

    /// Removes the empty features in `self.features`.
    fn remove_allzero_features(&mut self) {
        let features = mem::take(&mut self.features);
        self.name_to_index = features
            .iter()
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
        self.features = features
            .into_iter()
            .filter(|feat| !feat.is_empty())
            .collect();
        self.n_feature = self.features.len();
    }

    /// Returns a copy of `self` with the target values replaced by `target`
    /// (one per example), keeping the features and their names.
    ///
    /// Boosters that fit real-valued working responses (for example
    /// gradient boosting's pseudo-residuals) use it to pass them to a
    /// regression weak learner. The values need not be labels.
    ///
    /// Time complexity: `O(m d)`, a copy of the features.
    ///
    /// # Panics
    /// Panics if `target` does not have one value per example.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::Sample;
    ///
    /// let rows = vec![vec![0.0], vec![1.0]];
    /// let sample = Sample::from_dense(&rows, Some(&[-1.0, 1.0])).unwrap();
    /// let residuals = sample.with_target(vec![0.25, -0.5]);
    /// assert_eq!(residuals.target(), &[0.25, -0.5]);
    /// assert_eq!(residuals["feature_0"][1], 1.0);
    /// ```
    pub fn with_target(&self, target: Vec<f64>) -> Self {
        assert_eq!(
            target.len(),
            self.n_sample,
            "the target needs one value per example"
        );
        Self {
            target,
            ..self.clone()
        }
    }

    /// Returns the sample made of the rows `rows`, in that order. A row may
    /// appear several times (e.g. when drawing with replacement) or not at
    /// all. Feature names, storage (dense or sparse), and targets are kept.
    ///
    /// Time complexity: `O(n k)` for `n = rows.len()` and `k` features, with
    /// an extra `O(log m)` factor for sparse features.
    ///
    /// # Panics
    /// Panics if a row is out of range.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::Sample;
    ///
    /// let rows = vec![vec![0.0], vec![1.0], vec![2.0]];
    /// let sample = Sample::from_dense(&rows, Some(&[-1.0, 1.0, 1.0])).unwrap();
    /// let sub = sample.select_rows(&[2, 0, 2]);
    /// assert_eq!(sub.shape(), (3, 1));
    /// assert_eq!(sub["feature_0"][0], 2.0);
    /// assert_eq!(sub.target(), &[1.0, -1.0, 1.0]);
    /// ```
    pub fn select_rows(&self, rows: &[usize]) -> Self {
        assert!(
            rows.iter().all(|&i| i < self.n_sample),
            "row index out of range for a sample of {} rows",
            self.n_sample
        );
        let features = self
            .features
            .iter()
            .map(|feature| match feature {
                Feature::Dense { name, vals } => Feature::Dense {
                    name: name.clone(),
                    vals: rows.iter().map(|&i| vals[i]).collect(),
                },
                Feature::Sparse { name, .. } => Feature::Sparse {
                    name: name.clone(),
                    vals: rows
                        .iter()
                        .enumerate()
                        .map(|(new, &i)| (new, feature[i]))
                        .filter(|&(_, v)| v != 0.0)
                        .collect(),
                    size: rows.len(),
                },
            })
            .collect();
        let target = if self.target.is_empty() {
            Vec::new()
        } else {
            rows.iter().map(|&i| self.target[i]).collect()
        };
        Self {
            name_to_index: self.name_to_index.clone(),
            features,
            target,
            n_sample: rows.len(),
            n_feature: self.n_feature,
        }
    }

    /// Returns `(number of examples, number of features)`.
    /// The label column is not counted as a feature.
    ///
    /// # Examples
    /// ```
    /// use std::io::BufReader;
    /// use miniboosts::Sample;
    ///
    /// let csv = b"x,y,class\n-2,0,-1\n1,0.5,1\n3,1,1\n";
    /// let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    ///     .unwrap()
    ///     .set_target("class");
    /// let (n_sample, n_feature) = sample.shape();
    /// assert_eq!((n_sample, n_feature), (3, 2));
    /// // A probability vector over the examples, as passed to weak learners.
    /// let uniform = vec![1.0 / n_sample as f64; n_sample];
    /// assert_eq!(uniform.len(), sample.target().len());
    /// ```
    pub fn shape(&self) -> (usize, usize) {
        (self.n_sample, self.n_feature)
    }

    /// Renames all feature columns, in column order, and returns the old names.
    ///
    /// # Panics
    /// Panics if `names.len()` differs from the number of features.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::Sample;
    ///
    /// let rows = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    /// let mut sample = Sample::from_dense(&rows, Some(&[-1.0, 1.0])).unwrap();
    /// let old = sample.replace_names(["height", "weight"]);
    /// assert_eq!(old, ["feature_0", "feature_1"]);
    /// assert_eq!(sample["weight"][1], 4.0);
    /// ```
    pub fn replace_names<S, T>(&mut self, names: T) -> Vec<String>
    where
        S: ToString + std::fmt::Display,
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

        let old_names = names
            .iter()
            .zip(&mut self.features[..])
            .map(|(name, feature)| feature.replace_name(name))
            .collect();

        self.name_to_index = self
            .features
            .iter()
            .map(|feature| feature.name().to_string())
            .enumerate()
            .map(|(i, name)| (name, i))
            .collect();
        old_names
    }

    /// Returns the `idx`-th example as `(x, y)`, where `x` holds its feature
    /// values in column order and `y` is its label.
    ///
    /// # Panics
    /// Panics if `idx` is out of bounds or the labels are not set.
    pub fn at(&self, idx: usize) -> (Vec<f64>, f64) {
        let x = self
            .features
            .iter()
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

    /// Checks that `self` is a valid training set for binary classification.
    /// Boosters call this in [`crate::Booster::preprocess`].
    ///
    /// # Panics
    /// Panics (instead of returning `false`) if the labels are unset,
    /// some label is not an integer, or the labels do not take exactly two
    /// distinct values.
    ///
    /// Two distinct integer labels other than `{-1, +1}` (e.g., `{0, 1}`)
    /// only print a warning to standard output; they are **not** rejected,
    /// even though the boosters and the decision tree assume `-1` and `+1`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::Sample;
    ///
    /// let rows = vec![vec![0.0]; 3];
    /// // Valid: labels -1 and +1, both present. Returns normally.
    /// Sample::from_dense(&rows, Some(&[-1.0, 1.0, 1.0]))
    ///     .unwrap()
    ///     .is_valid_binary_instance();
    ///
    /// // Invalid: only one class. The check panics instead of returning.
    /// let one_class = Sample::from_dense(&rows, Some(&[1.0, 1.0, 1.0])).unwrap();
    /// assert!(std::panic::catch_unwind(|| one_class.is_valid_binary_instance()).is_err());
    /// ```
    pub fn is_valid_binary_instance(&self) {
        // Check whether the target column is specified.
        self.target_is_specified();

        // Check whether the target values can be converted into integers.
        let non_integers = self
            .target
            .iter()
            .filter(|&yi| !yi.trunc().eq(yi))
            .collect::<Vec<_>>();
        if !non_integers.is_empty() {
            let line = non_integers
                .iter()
                .take(5)
                .map(|yi| yi.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            panic!(
                "Target values are non-integer types.\n\
                 Ex. [{line}, ...]."
            );
        }

        // Check whether the target values takes exactly 2 kinds.
        let set = self
            .target
            .iter()
            .copied()
            .map(|yi| yi as i32)
            .collect::<HashSet<_>>();
        let n_label = set.len();
        match set.len().cmp(&2) {
            Ordering::Greater => {
                panic!(
                    "The target values take more than 2 kinds. \
                     Expected 2 kinds, got {n_label} kinds."
                );
            }
            Ordering::Less => {
                panic!(
                    "The target values take less than 2 kinds. \
                     Expected 2 kinds, got {n_label} kinds."
                );
            }
            Ordering::Equal => {}
        }

        // Check whether the target values takes +1 or -1.
        let is_pm = set.iter().all(|y| y.eq(&1) || y.eq(&-1));
        if !is_pm {
            let line = set
                .iter()
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

    /// Returns `(Σ_i w_i x_ij, Σ_i w_i (x_ij - mean_j)^2)` for each feature
    /// `j`; see [`Feature::weighted_mean_and_variance`].
    ///
    /// These are the weighted mean and variance only if `weight` is a
    /// probability vector (nonnegative entries summing to one) with one
    /// entry per example; this is not checked.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::Sample;
    ///
    /// let rows = vec![vec![0.0], vec![2.0], vec![4.0]];
    /// let sample = Sample::from_dense(&rows, Some(&[-1.0, 1.0, 1.0])).unwrap();
    /// let stats = sample.weighted_mean_and_variance([0.5, 0.0, 0.5]);
    /// // Mean 0.5 * 0 + 0.5 * 4 = 2; variance 0.5 * 4 + 0.5 * 4 = 4.
    /// // Both are exact in binary floating point.
    /// assert_eq!(stats, vec![(2.0, 4.0)]);
    /// ```
    pub fn weighted_mean_and_variance<T>(&self, weight: T) -> Vec<(f64, f64)>
    where
        T: AsRef<[f64]>,
    {
        let weight = weight.as_ref();
        self.features()
            .par_iter()
            .map(|feat| feat.weighted_mean_and_variance(weight))
            .collect()
    }

    /// Returns `Σ_i w_i x_ij` for each feature `j`;
    /// see [`Feature::weighted_mean`].
    ///
    /// This is the weighted mean only if `weight` is a probability vector
    /// (nonnegative entries summing to one) with one entry per example;
    /// this is not checked.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::Sample;
    ///
    /// let rows = vec![vec![0.0, 1.0], vec![2.0, 1.0], vec![4.0, 1.0]];
    /// let sample = Sample::from_dense(&rows, Some(&[-1.0, 1.0, 1.0])).unwrap();
    /// // Exact: every product and partial sum is a dyadic rational.
    /// assert_eq!(sample.weighted_mean([0.25, 0.25, 0.5]), vec![2.5, 1.0]);
    /// ```
    pub fn weighted_mean<T>(&self, weight: T) -> Vec<f64>
    where
        T: AsRef<[f64]>,
    {
        let weight = weight.as_ref();
        self.features()
            .par_iter()
            .map(|feat| feat.weighted_mean(weight))
            .collect()
    }

    /// Returns, for each feature, the weighted mean over the examples whose
    /// label equals `y`, normalized by those examples' total weight;
    /// see [`Feature::weighted_mean_for_label`].
    ///
    /// # Panics
    /// Panics if `weight` has the wrong length, contains a negative or
    /// non-finite entry, or the examples labeled `y` have zero total weight.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::Sample;
    ///
    /// let rows = vec![vec![0.0], vec![2.0], vec![4.0]];
    /// let sample = Sample::from_dense(&rows, Some(&[-1.0, 1.0, 1.0])).unwrap();
    /// // Positive examples have weights 1 and 3: (1 * 2 + 3 * 4) / 4 = 3.5.
    /// // The weights need not sum to one.
    /// assert_eq!(sample.weighted_mean_for_label(1.0, [5.0, 1.0, 3.0]), vec![3.5]);
    /// assert_eq!(sample.weighted_mean_for_label(-1.0, [5.0, 1.0, 3.0]), vec![0.0]);
    /// ```
    pub fn weighted_mean_for_label<T>(&self, y: f64, weight: T) -> Vec<f64>
    where
        T: AsRef<[f64]>,
    {
        let weight = weight.as_ref();
        let target = self.target();
        self.features()
            .par_iter()
            .map(|feat| feat.weighted_mean_for_label(y, target, weight))
            .collect()
    }

    /// Returns, for each feature, the weighted mean and population variance
    /// over the examples whose label equals `y`, normalized by those
    /// examples' total weight;
    /// see [`Feature::weighted_mean_and_variance_for_label`].
    ///
    /// # Panics
    /// Same conditions as [`Sample::weighted_mean_for_label`].
    ///
    /// # Examples
    /// ```
    /// use miniboosts::Sample;
    ///
    /// let rows = vec![vec![0.0], vec![2.0], vec![4.0]];
    /// let sample = Sample::from_dense(&rows, Some(&[-1.0, 1.0, 1.0])).unwrap();
    /// // Positive examples 2 and 4 with equal weight: mean 3, variance 1.
    /// let stats = sample.weighted_mean_and_variance_for_label(1.0, [1.0, 1.0, 1.0]);
    /// assert_eq!(stats, vec![(3.0, 1.0)]);
    /// ```
    pub fn weighted_mean_and_variance_for_label<T>(&self, y: f64, weight: T) -> Vec<(f64, f64)>
    where
        T: AsRef<[f64]>,
    {
        let weight = weight.as_ref();
        let target = self.target();
        self.features()
            .par_iter()
            .map(|feat| feat.weighted_mean_and_variance_for_label(y, target, weight))
            .collect()
    }

    fn append(&mut self, row: usize, feat: Vec<f64>, y: f64) {
        self.features.par_iter_mut().zip(feat).for_each(|(col, f)| {
            col.append((row, f));
        });
        self.target.push(y);
    }

    /// Splits `self` into `(training, test)`, preserving the order in `ix`.
    /// `ix` must be a permutation of all row indices. The half-open interval
    /// `ix[start..end]` selects test rows; the remainder selects training rows.
    /// Empty partitions are valid. Feature names and storage (dense/sparse)
    /// are kept. Useful for cross-validation folds.
    ///
    /// # Panics
    /// Panics for an invalid permutation or interval, or if labels are unset.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::Sample;
    ///
    /// let rows = (0..6).map(|i| vec![i as f64]).collect::<Vec<_>>();
    /// let y = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
    /// let sample = Sample::from_dense(&rows, Some(&y)).unwrap();
    ///
    /// // `ix` must contain every row index exactly once (a permutation).
    /// // Here the test set is the half-open interval `ix[2..4]`, i.e. rows
    /// // 1 and 4; the training set is rows 5, 0, 2, 3 in this order.
    /// let ix = [5, 0, 1, 4, 2, 3];
    /// let (train, test) = sample.split(ix, 2, 4);
    /// assert_eq!(test.shape(), (2, 1));
    /// assert_eq!(test.at(0), (vec![1.0], -1.0));
    /// assert_eq!(test.at(1), (vec![4.0], 1.0));
    /// assert_eq!(train.shape(), (4, 1));
    /// assert_eq!(train["feature_0"][0], 5.0);
    /// assert_eq!(train.target(), &[1.0, -1.0, -1.0, 1.0]);
    ///
    /// // k-fold cross-validation: fold `k` tests on `ix[k * size..(k + 1) * size]`.
    /// let (n_fold, size) = (3, 2);
    /// for k in 0..n_fold {
    ///     let (train, test) = sample.split(ix, k * size, (k + 1) * size);
    ///     assert_eq!((train.shape().0, test.shape().0), (4, 2));
    /// }
    ///
    /// // `start == end` gives an empty test set.
    /// let (train, test) = sample.split(ix, 0, 0);
    /// assert_eq!((train.shape().0, test.shape().0), (6, 0));
    /// ```
    pub fn split<T>(&self, ix: T, start: usize, end: usize) -> (Sample, Sample)
    where
        T: AsRef<[usize]>,
    {
        let ix = ix.as_ref();
        assert!(
            start <= end && end <= self.n_sample,
            "invalid split interval"
        );
        assert_eq!(ix.len(), self.n_sample, "split requires all row indices");
        let mut seen = vec![false; self.n_sample];
        for &i in ix {
            assert!(
                i < self.n_sample && !seen[i],
                "split requires a permutation"
            );
            seen[i] = true;
        }
        let n_feature = self.features.len();
        let test_size = end - start;
        let train_size = self.n_sample - test_size;

        let name_to_ix = self.name_to_index.clone();
        let mut train = Self {
            n_sample: train_size,
            n_feature,
            name_to_index: name_to_ix.clone(),
            features: vec![Feature::sparse("dummy", 0); n_feature],
            target: Vec::with_capacity(train_size),
        };

        let mut test = Self {
            n_sample: test_size,
            n_feature,
            name_to_index: name_to_ix,
            features: vec![Feature::sparse("dummy", 0); n_feature],
            target: Vec::with_capacity(test_size),
        };

        for (name, &i) in self.name_to_index.iter() {
            if self.features[i].is_sparse() {
                train.features[i] = Feature::sparse(name.to_string(), train_size);
                test.features[i] = Feature::sparse(name.to_string(), test_size);
            } else {
                train.features[i] = Feature::dense(name.to_string());
                test.features[i] = Feature::dense(name.to_string());
            }
        }

        for (i, &ii) in ix.iter().enumerate().take(start) {
            let (x, y) = self.at(ii);
            train.append(i, x, y);
        }

        for (i, &ii) in ix[start..end].iter().enumerate() {
            let (x, y) = self.at(ii);
            test.append(i, x, y);
        }

        for (i, &ii) in ix.iter().enumerate().take(self.n_sample).skip(end) {
            let (x, y) = self.at(ii);
            train.append(i - test_size, x, y);
        }

        (train, test)
    }
}

/// Parse the following type of `str` to the pair of `(usize, f64)`.
/// `index:value`, where `index: usize` and `value: f64`.
fn index_and_feature(word: &str) -> (usize, f64) {
    let mut i_x = word.split(':');
    let i = i_x
        .next()
        .unwrap()
        .trim()
        .parse::<usize>()
        .expect("Failed to parse an index.");
    let x = i_x
        .next()
        .unwrap()
        .trim()
        .parse::<f64>()
        .expect("Failed to parse a feature value.");

    (i, x)
}

/// `sample[name]` returns the feature column named `name`.
///
/// # Panics
/// Panics if no feature column is named `name`.
///
/// # Examples
/// ```
/// use miniboosts::Sample;
///
/// let sample = Sample::from_dense(&[vec![1.0, 2.0]], Some(&[1.0])).unwrap();
/// let column = &sample["feature_1"];
/// assert!(column.is_dense());
/// assert_eq!(column[0], 2.0);
/// ```
impl<S> Index<S> for Sample
where
    S: AsRef<str>,
{
    type Output = Feature;

    fn index(&self, name: S) -> &Self::Output {
        let name: &str = name.as_ref();
        let k = *self.name_to_index.get(name).unwrap();
        &self.features[k]
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn select_rows_keeps_sparse_features_sparse() {
        let mut sparse = Feature::sparse("s", 4);
        sparse.append((1, 2.0));
        sparse.append((3, -1.0));
        let mut dense = Feature::dense("d");
        for v in [0.5, 1.5, 2.5, 3.5] {
            dense.append((0, v));
        }
        let sample = Sample {
            name_to_index: HashMap::from([("s".to_string(), 0), ("d".to_string(), 1)]),
            features: vec![sparse, dense],
            target: vec![1.0, -1.0, 1.0, -1.0],
            n_sample: 4,
            n_feature: 2,
        };
        let sub = sample.select_rows(&[3, 3, 0, 1]);
        assert!(sub["s"].is_sparse() && sub["d"].is_dense());
        assert_eq!(
            (0..4).map(|i| sub["s"][i]).collect::<Vec<_>>(),
            vec![-1.0, -1.0, 0.0, 2.0]
        );
        assert_eq!(
            (0..4).map(|i| sub["d"][i]).collect::<Vec<_>>(),
            vec![3.5, 3.5, 0.5, 1.5]
        );
        assert_eq!(sub.target(), &[-1.0, -1.0, 1.0, -1.0]);
        assert_eq!(sub.shape(), (4, 2));
        let result = std::panic::catch_unwind(|| sample.select_rows(&[4]));
        assert!(result.is_err());
    }

    use super::*;

    fn training_examples(bytes: &[u8], has_header: bool) -> Sample {
        use std::io::BufReader;
        let reader = BufReader::new(bytes);
        Sample::from_reader(reader, has_header)
            .unwrap()
            .set_target("class")
    }

    #[test]
    fn split_reindexes_sparse_rows_and_preserves_labels() {
        let dense = training_examples(b"x,class\n1,1\n0,-1\n3,1\n4,-1", true);
        let mut sparse = dense.clone();
        sparse.features = dense
            .features
            .iter()
            .map(|f| {
                let mut result = Feature::sparse(f.name(), dense.n_sample);
                for i in 0..dense.n_sample {
                    result.append((i, f[i]));
                }
                result
            })
            .collect();
        for sample in [dense, sparse] {
            let indices = [3, 0, 2, 1];
            for (start, end) in [(1, 3), (0, 0), (0, 4), (2, 4)] {
                let (train, test) = sample.split(indices, start, end);
                let train_ix = indices[..start]
                    .iter()
                    .chain(&indices[end..])
                    .copied()
                    .collect::<Vec<_>>();
                for (part, expected) in
                    [(&train, train_ix.as_slice()), (&test, &indices[start..end])]
                {
                    assert_eq!(part.n_sample, expected.len());
                    for (i, &source) in expected.iter().enumerate() {
                        assert_eq!(part.at(i), sample.at(source));
                    }
                    for feature in &part.features {
                        assert_eq!(feature.clone().into_vals().len(), expected.len());
                    }
                }
            }
            assert!(std::panic::catch_unwind(|| sample.split([0, 0, 2, 3], 1, 2)).is_err());
        }
    }

    #[test]
    fn test_from_reader_01() {
        let bytes = b"\
            test,dummy,class\n\
            0.1,0.2,1.0\n\
            -8.0,2.0,-1.0\n\
            3.0,-9.0,1.0\n\
            -0.001,0.0,-1.0";
        let sample = training_examples(bytes, true);
        println!("{sample:?}");
    }
}

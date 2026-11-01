//! [`SampleReader`]: a builder that reads a CSV or SVMlight file into a
//! [`Sample`].
use std::io;
use std::path::Path;

use super::sample_struct::Sample;

/// A builder that reads a CSV or SVMlight file into a [`Sample`].
///
/// The format is chosen by the file extension: a file whose extension is
/// exactly `csv` is read as CSV; **any other file is read as SVMlight**.
/// Other formats are not supported.
///
/// # CSV
/// - Comma-separated numeric values, one example per line.
/// - With [`SampleReader::has_header`]`(true)`, the first line holds the
///   column names. Without a header, columns are named
///   `Feat. [1]`, `Feat. [2]`, ..., `Feat. [n]` (1-based), so the target
///   column must be selected by one of these names.
/// - [`SampleReader::target_feature`] is required: that column becomes the
///   label vector and is removed from the features.
///
/// # SVMlight
/// Each line has the form `y index:value index:value ...`; see
/// [`Sample`] for details. The header flag and target column are ignored.
///
/// # Labels
/// Boosters in this crate perform binary classification and require labels
/// `-1` and `+1`, with both classes present in the training sample. Labels
/// are not validated when reading; boosters check them when training starts
/// (see [`Sample::is_valid_binary_instance`]).
///
/// # Examples
/// Read a CSV file with a header row. (The example writes a small file to
/// the system's temporary directory and removes it afterwards.)
/// ```
/// use miniboosts::SampleReader;
///
/// let path = std::env::temp_dir()
///     .join(format!("miniboosts_doc_reader_struct_{}.csv", std::process::id()));
/// std::fs::write(&path, "x,y,class\n-2,0,-1\n-1,1,-1\n1,0,1\n2,1,1\n").unwrap();
///
/// // Any `AsRef<Path>` works as the path, including a borrowed `&PathBuf`.
/// let sample = SampleReader::default()
///     .file(&path)
///     .has_header(true)
///     .target_feature("class")
///     .read()
///     .unwrap();
/// # std::fs::remove_file(&path).unwrap();
/// assert_eq!(sample.shape(), (4, 2));
/// assert_eq!(sample.target(), &[-1.0, -1.0, 1.0, 1.0]);
/// assert_eq!(sample["y"][1], 1.0);
/// ```
pub struct SampleReader<P, S> {
    file: Option<P>,
    has_header: bool,
    target: Option<S>,
}

// Implemented by hand: `#[derive(Default)]` would require `P: Default` and
// `S: Default`, which rejects borrowed paths such as `&PathBuf`.
impl<P, S> Default for SampleReader<P, S> {
    /// Returns a reader with no file, no target column, and
    /// `has_header = false`.
    fn default() -> Self {
        Self {
            file: None,
            has_header: false,
            target: None,
        }
    }
}

impl<P, S> SampleReader<P, S> {
    /// Sets whether a CSV file has a header row. Default is `false`.
    /// Ignored for SVMlight files.
    ///
    /// # Examples
    /// A CSV file without a header: the columns are named `Feat. [1]`,
    /// `Feat. [2]`, ..., so the label column is selected by such a name.
    /// ```
    /// use miniboosts::SampleReader;
    ///
    /// let path = std::env::temp_dir()
    ///     .join(format!("miniboosts_doc_reader_noheader_{}.csv", std::process::id()));
    /// std::fs::write(&path, "-2,0,-1\n1,0.5,1\n").unwrap();
    ///
    /// let sample = SampleReader::default()
    ///     .file(&path)
    ///     .has_header(false)
    ///     .target_feature("Feat. [3]")
    ///     .read()
    ///     .unwrap();
    /// # std::fs::remove_file(&path).unwrap();
    /// assert_eq!(sample.shape(), (2, 2));
    /// assert_eq!(sample.target(), &[-1.0, 1.0]);
    /// assert_eq!(sample["Feat. [2]"][1], 0.5);
    /// ```
    pub fn has_header(mut self, flag: bool) -> Self {
        self.has_header = flag;
        self
    }
}

impl<P, S> SampleReader<P, S>
where
    P: AsRef<Path>,
{
    /// Sets the path of the file to read. Required.
    /// The extension `csv` selects the CSV reader; anything else selects
    /// the SVMlight reader.
    ///
    /// # Examples
    /// An SVMlight file (any extension other than `csv`). Features are stored
    /// sparsely, index `k` is named `Feat. [k+1]`, and no target column is
    /// needed because the first token of each line is the label.
    /// ```
    /// use miniboosts::SampleReader;
    ///
    /// let path = std::env::temp_dir()
    ///     .join(format!("miniboosts_doc_reader_svmlight_{}.svm", std::process::id()));
    /// std::fs::write(&path, "-1 1:0.5 3:2.0\n+1 2:1.0\n+1 1:-1.0 3:4.0\n").unwrap();
    ///
    /// // The type of the (unused) target name must still be fixed.
    /// let sample = SampleReader::<_, &str>::default().file(&path).read().unwrap();
    /// # std::fs::remove_file(&path).unwrap();
    /// // Index 0 never occurs, so its all-zero column `Feat. [1]` is removed.
    /// assert_eq!(sample.shape(), (3, 3));
    /// assert_eq!(sample.target(), &[-1.0, 1.0, 1.0]);
    /// assert!(sample["Feat. [2]"].is_sparse());
    /// assert_eq!(sample["Feat. [2]"][2], -1.0);
    /// assert_eq!(sample["Feat. [4]"][0], 2.0);
    /// assert_eq!(sample["Feat. [3]"][0], 0.0); // omitted entries are zero
    /// ```
    pub fn file(mut self, file: P) -> Self {
        self.file = Some(file);
        self
    }
}

impl<P, S> SampleReader<P, S>
where
    S: AsRef<str>,
{
    /// Sets the name of the CSV column used as the label vector.
    /// Required for CSV files; ignored for SVMlight files.
    /// For binary classification its values must be `-1` or `+1`.
    ///
    /// # Examples
    /// The target column may be at any position; it is removed from the
    /// features.
    /// ```
    /// use miniboosts::SampleReader;
    ///
    /// let path = std::env::temp_dir()
    ///     .join(format!("miniboosts_doc_reader_target_{}.csv", std::process::id()));
    /// std::fs::write(&path, "label,x\n1,0.5\n-1,-0.5\n").unwrap();
    ///
    /// let sample = SampleReader::default()
    ///     .file(&path)
    ///     .has_header(true)
    ///     .target_feature("label")
    ///     .read()
    ///     .unwrap();
    /// # std::fs::remove_file(&path).unwrap();
    /// assert_eq!(sample.shape(), (2, 1));
    /// assert_eq!(sample.target(), &[1.0, -1.0]);
    /// assert_eq!(sample.features()[0].name(), "x");
    /// ```
    pub fn target_feature(mut self, column: S) -> Self {
        self.target = Some(column);
        self
    }
}

impl<P, S> SampleReader<P, S>
where
    P: AsRef<Path>,
    S: AsRef<str>,
{
    /// Reads the file and returns the resulting [`Sample`].
    /// This method consumes `self`.
    ///
    /// # Errors
    /// Returns `Err` if the file cannot be opened or read.
    ///
    /// # Panics
    /// - if [`SampleReader::file`] was not called;
    /// - for a CSV file, if [`SampleReader::target_feature`] was not called
    ///   or names a column that does not exist;
    /// - if a value cannot be parsed as `f64` (or an SVMlight index as
    ///   `usize`), or a CSV row has more columns than the first row;
    /// - for a CSV file with a header, if the file is empty.
    ///
    /// # Examples
    /// A missing file is reported as an I/O error, not a panic.
    /// ```
    /// use miniboosts::SampleReader;
    ///
    /// let path = std::env::temp_dir().join("miniboosts_doc_reader_missing_file.csv");
    /// # let _ = std::fs::remove_file(&path);
    /// let result = SampleReader::default()
    ///     .file(path)
    ///     .has_header(true)
    ///     .target_feature("class")
    ///     .read();
    /// let err = result.err().unwrap();
    /// assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    /// ```
    pub fn read(self) -> io::Result<Sample> {
        if self.file.is_none() {
            panic!("The file name for csv/svmlight is not set");
        }
        let file = self.file.unwrap();
        let file = file.as_ref();

        let sample = if file.extension().is_some_and(|ext| ext == "csv") {
            if self.target.is_none() {
                panic!(
                    "Target (class) column is not specified. \
                    Use `SampleReader::target_feature`."
                );
            }
            let target = self.target.unwrap();
            Sample::from_csv(file, self.has_header)?.set_target(target.as_ref())
        } else {
            Sample::from_svmlight(file)?
        };
        Ok(sample)
    }
}

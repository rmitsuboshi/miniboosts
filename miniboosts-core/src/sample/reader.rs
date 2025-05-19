use std::path::Path;
use std::io;

use super::sample_struct::Sample;

#[derive(Default)]
pub struct SampleReader<P, S> {
    file: Option<P>,
    has_header: bool,
    target: Option<S>,
}

impl<P, S> SampleReader<P, S> {
    /// Set the flag whether the file has the header row or not.
    /// Default is `false.`
    pub fn has_header(mut self, flag: bool) -> Self {
        self.has_header = flag;
        self
    }
}

impl<P, S> SampleReader<P, S>
    where P: AsRef<Path>
{
    /// Set the file name.
    pub fn file(mut self, file: P) -> Self {
        self.file = Some(file);
        self
    }
}

impl<P, S> SampleReader<P, S>
    where S: AsRef<str>
{
    /// Set the column name that is used for target label.
    /// The each item of the column takes value in `{-1, +1}.`
    pub fn target_feature(mut self, column: S) -> Self {
        self.target = Some(column);
        self
    }
}

impl<P, S> SampleReader<P, S>
    where P: AsRef<Path>,
          S: AsRef<str>
{
    /// Reads the file based on the arguments, 
    /// and returns `std::io::Result<Sample>`.
    /// This method consumes `self.`
    /// If you read a CSV file, the extension should be `.csv`.
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
                    Use `SampleReader::target`."
                );
            }
            let target = self.target.unwrap();
            Sample::from_csv(file, self.has_header)?
                .set_target(target.as_ref())
        } else {
            Sample::from_svmlight(file)?
        };
        Ok(sample)
    }
}



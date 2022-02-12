//! Defines some data structure used in this crate.
use std::convert::From;
use std::collections::HashMap;
use std::ops::Index;


/// The trait `Data` defines the desired property of data.
pub trait Data {
    /// The value type of the specified index.
    type Output;

    /// Returns the value of the specified index.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// use lycaon::Data;
    /// let data = vec![1.0, 2.0, 3.0];
    /// assert_eq!(data.value_at(1), 2.0);
    /// ```
    /// 
    fn value_at(&self, index: usize) -> Self::Output;


    /// Returns the dimension of `self`.
    /// # Example
    /// 
    /// ```rust
    /// use lycaon::Data;
    /// let data = vec![1.0, 2.0, 3.0];
    /// assert_eq!(data.dim(), 3);
    /// ```
    fn dim(&self) -> usize;
}


impl<T> Data for HashMap<usize, T>
    where T: Default + Clone
{
    type Output = T;
    fn value_at(&self, index: usize) -> Self::Output {
        match self.get(&index) {
            Some(value) => value.clone(),
            None        => Default::default()
        }
    }


    fn dim(&self) -> usize {
        match self.keys().max() {
            Some(&k) => k + 1,
            None     => 0_usize,
        }
    }
}


impl<T> Data for Vec<T>
    where T: Clone
{
    type Output = T;
    fn value_at(&self, index: usize) -> Self::Output {
        self[index].clone()
    }


    fn dim(&self) -> usize {
        self.len()
    }
}

/// Just an alias to `f64`.
pub type Label = f64;



/// A sequence of the `LabeledData`.
/// We assume that all the example in `sample` has the same format.
#[derive(Debug)]
pub struct Sample<T: Data> {

    /// Holds the pair of data and label.
    inner: Vec<(T, Label)>,

    /// The number of examples. This value is equivalent to
    /// `dat_set.len()` and `lab_set.len()`.
    size:      usize,


    /// The number of features of `Sample<T>`.
    dimension: usize,
}


impl<T: Data> Sample<T> {

    /// Returns the number of training examples.
    /// # Example
    /// 
    /// ```rust
    /// use lycaon::Sample;
    /// 
    /// let examples = vec![
    ///     vec![1.0, 2.0, 3.0],
    ///     vec![4.0, 5.0, 6.0],
    /// ];
    /// let labels = vec![1.0, -1.0];
    /// 
    /// let sample = Sample::from((examples, labels));
    /// 
    /// assert_eq!(sample.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns the dimension in `Sample<T>`.
    /// # Example
    /// 
    /// ```rust
    /// use lycaon::Sample;
    /// 
    /// let examples = vec![
    ///     vec![1.0, 2.0, 3.0],
    ///     vec![4.0, 5.0, 6.0],
    /// ];
    /// let labels = vec![1.0, -1.0];
    /// 
    /// let sample = Sample::from((examples, labels));
    /// 
    /// assert_eq!(sample.dim(), 3);
    /// 
    /// // An example for the sparse sample.
    /// use std::collections::HashMap;
    /// let mut examples = vec![
    ///     HashMap::from([ (0,  1.0), (2, 2.0), (7, -2.0) ]),
    ///     HashMap::from([ (1, -1.0), (8, 8.0), (9,  6.1) ]),
    /// ];
    /// let labels = vec![-1.0, -1.0];
    /// let sample = Sample::from((examples, labels));
    /// assert_eq!(sample.dim(), 10);
    /// ```
    pub fn dim(&self) -> usize {
        self.dimension
    }


}


/// A struct for implementing the iterator over `Sample`.
pub struct SampleIter<'a, T> {
    inner: &'a [(T, Label)]
}


impl<T: Data> Sample<T> {
    /// Iterator for `Sample`.
    pub fn iter(&self) -> SampleIter<'_, T> {
        SampleIter { inner: &self.inner[..] }
    }
}


impl<'a, T: Data> Iterator for SampleIter<'a, T> {
    type Item = &'a (T, Label);

    /// Returns a pair `&'a (T, Label)` of `Sample<T>`.
    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.get(0) {
            Some(item) => {
                self.inner = &self.inner[1..];

                Some(item)
            },
            None => None
        }
    }
}


impl<T: Data> From<(Vec<T>, Vec<Label>)> for Sample<T> {
    /// Convert the pair `(Vec<T>, Vec<Label>)` to `Sample<T>`.
    fn from((examples, labels): (Vec<T>, Vec<Label>)) -> Self {
        assert_eq!(examples.len(), labels.len());


        let size = examples.len();

        let mut dimension = 0;

        let mut inner = Vec::with_capacity(size);

        for (dat, lab) in examples.into_iter().zip(labels.into_iter()) {
            dimension = std::cmp::max(dimension, dat.dim());
            inner.push((dat, lab));
        }

        Sample {
            inner,
            dimension,
            size,
        }
    }
}





impl<T: Data> Index<usize> for Sample<T> {
    type Output = (T, Label);

    /// Returns the pair `(T, Label)` at specified `index` of `Sample<T>`.
    fn index(&self, index: usize) -> &Self::Output {
        &self.inner[index]
    }
}



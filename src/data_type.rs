//! Defines some data structure used in this crate.
use std::convert::From;
use std::collections::HashMap;
use std::ops::Index;


/// The trait `Data` defines the desired property of data.
pub trait Data {
    /// The value type of the specified index.
    type Output;

    /// Returns the value of the specified index.
    fn value_at(&self, index: usize) -> Self::Output;

    /// Returns the dimension
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

/// Introduce the `Label` for clarity.
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
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns the maximum dimensions in `Sample<T>`.
    pub fn dim(&self) -> usize {
        self.dimension
    }


}


/// A struct for implementing the iterator over `Sample`.
pub struct SampleIter<'a, T> {
    inner: &'a [(T, Label)]
}


impl<T: Data> Sample<T> {
    /// Iterator for `Sample`
    pub fn iter(&self) -> SampleIter<'_, T> {
        SampleIter { inner: &self.inner[..] }
    }
}


impl<'a, T: Data> Iterator for SampleIter<'a, T> {
    type Item = &'a (T, Label);

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

    fn index(&self, index: usize) -> &Self::Output {
        &self.inner[index]
    }
}



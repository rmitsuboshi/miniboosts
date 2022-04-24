//! Defines some data structure used in this crate.
use std::convert::From;
use std::collections::HashMap;
use std::ops::Index;


/// A trait that guarantee the existence of maximal value.
pub trait DataBounds {
    /// Returns the maximum value.
    /// For example, `f64` returns `f64::MAX`.
    fn max_value() -> Self;


    /// Returns the minimum value.
    /// For example, `f64` returns `f64::MAX`.
    fn min_value() -> Self;
}

macro_rules! impl_databounds_primitive_inner {
    ($t:ty) => (
        impl DataBounds for $t {
            #[inline]
            fn max_value() -> Self {
                Self::MAX
            }


            fn min_value() -> Self {
                Self::MIN
            }
        }
    )
}


macro_rules! impl_databounds_primitive {
    ($($t:ty)*) => ($(
        impl_databounds_primitive_inner! { $t }
    )*)
}


impl_databounds_primitive! { i8 i16 i32 i64 i128 isize }
impl_databounds_primitive! { u8 u16 u32 u64 u128 usize }
impl_databounds_primitive! { f32 f64 }


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


/// A sequence of the `LabeledData`.
/// We assume that all the example in `sample` has the same format.
#[derive(Debug)]
pub struct Sample<D, L> {

    /// Holds the pair of data and label.
    inner: Vec<(D, L)>,

    /// The number of examples. This value is equivalent to
    /// `dat_set.len()` and `lab_set.len()`.
    size:      usize,


    /// The number of features of `Sample<T>`.
    dimension: usize,
}


impl<D, L> Sample<D, L> {

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
pub struct SampleIter<'a, D, L> {
    inner: &'a [(D, L)]
}


impl<D, L> Sample<D, L> {
    /// Iterator for `Sample`.
    pub fn iter(&self) -> SampleIter<'_, D, L> {
        SampleIter { inner: &self.inner[..] }
    }
}


impl<'a, D, L> Iterator for SampleIter<'a, D, L> {
    type Item = &'a (D, L);

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


impl<D, L> From<(Vec<D>, Vec<L>)> for Sample<D, L>
    where D: Data
{
    /// Convert the pair `(Vec<T>, Vec<Label>)` to `Sample<T>`.
    #[inline]
    fn from((examples, labels): (Vec<D>, Vec<L>)) -> Self {
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





impl<D, L> Index<usize> for Sample<D, L> {
    type Output = (D, L);

    /// Returns the pair `(T, Label)` at specified `index` of `Sample<T>`.
    fn index(&self, index: usize) -> &Self::Output {
        &self.inner[index]
    }
}



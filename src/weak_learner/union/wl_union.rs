use polars::prelude::*;

use crate::weak_learner::*;

use crate::{
    Classifier,
};


/// The union of weak learners.
/// One can use unify multiple weak learners by this struct.
/// 
/// See also:
/// - [`DTree`](crate::weak_learner::DTree)
/// 
/// # Example
/// ```no_run
/// use polars::prelude::*;
/// use miniboosts::prelude::*;
/// 
/// // Read the training data from the CSV file.
/// let mut data = CsvReader::from_path(path_to_csv_file)
///     .unwrap()
///     .has_header(true)
///     .finish()
///     .unwrap();
/// 
/// // Split the column corresponding to labels.
/// let target = data.drop_in_place(class_column_name).unwrap();
/// 
/// 
/// let t1 = DTree::init(&data, &target)
///     .max_depth(1)
///     .criterion(Criterion::Edge);
/// let t2 = DTree::init(&data, &target)
///     .max_depth(2)
///     .criterion(Criterion::Edge);
/// 
/// let weak_learner = WLUnion::new()
///     .union(Box::new(t1))
///     .union(Box::new(t2));
/// ```
pub struct WLUnion<F> {
    weak_learners: Vec<Box<dyn WeakLearner<Hypothesis = F>>>,
}

impl<F> WLUnion<F> {
    /// Generates an empty instance of `WLUnion`.
    pub fn new() -> Self {
        let weak_learners = Vec::new();
        Self { weak_learners }
    }


    /// Append a weak learner to the union.
    pub fn union(
        mut self,
        weak_learner: Box<dyn WeakLearner<Hypothesis = F>>
    ) -> Self
    {
        self.weak_learners.push(weak_learner);
        self
    }
}


impl<F> WeakLearner for WLUnion<F>
    where F: Classifier + 'static
{
    type Hypothesis = Box<dyn Classifier>;


    fn name(&self) -> &str {
        "Union of WLs"
    }


    /// Output a boxed hypothesis which maximizes the edge
    /// among the hypotheses returned by weak learners.
    fn produce(
        &self,
        data: &DataFrame,
        target: &Series,
        dist: &[f64],
    ) -> Self::Hypothesis
    {
        self.weak_learners.iter()
            .map(|wl| {
                let h = wl.produce(data, target, dist);
                let edge = target.i64()
                    .expect("The target class is not a dtype i64")
                    .into_iter()
                    .zip(dist)
                    .enumerate()
                    .map(|(i, (y, d))| {
                        let y = y.unwrap() as f64;
                        let p = h.predict(data, i) as f64;

                        d * y * p
                    })
                    .sum::<f64>();
                (edge, Box::new(h))
            })
            .max_by(|(e1, _), (e2, _)| e1.partial_cmp(&e2).unwrap())
            .unwrap().1
    }
}

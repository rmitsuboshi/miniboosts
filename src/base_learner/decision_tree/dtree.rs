// TODO LIST
//  * Implement `best_hypothesis`
//      - Train/test split
//      ? construct_full_tree
//      - prune
//  * Implement `construct_full_tree`
//      ? Compute impurity
//      - Test whether this function works as expected.
//  * Implement `pruing`
//      - Cross-validation
//  * Implement `train_test_split` to find the best pruning
//  * Test code
//      - construct_full_tree
//  * Run boosting

use crate::{DataBounds, Data, Label, Sample};
use crate::BaseLearner;


use super::dtree_classifier::DTreeClassifier;
use super::split_rule::*;
use super::node::*;


use std::marker::PhantomData;



/// Enumerate of split rule.
/// This enumeration will be updated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    /// Split based on feature (decision stump).
    Feature,
}


/// Generates a `DTreeClassifier` for a given distribution
/// over examples.
pub struct DTree<S> {
    criterion: Criterion,
    _phantom:  PhantomData<S>,
}


impl<S> DTree<S> {
    /// Initialize `DTree`.
    #[inline]
    pub fn init<D>(_sample: &Sample<D>) -> Self
        where D: Data
    {
        Self {
            criterion: Criterion::Entropy,
            _phantom:  PhantomData
        }
    }
}


impl<S, O, D> BaseLearner<D> for DTree<S>
    where S: SplitRule<Input = D>,
          D: Data<Output = O>,
          O: PartialOrd + Clone + DataBounds
{
    type Clf = DTreeClassifier<S>;
    fn best_hypothesis(&self, sample: &Sample<D>, distribution: &[f64])
        -> Self::Clf
    {
        let m = sample.len();


        // TODO
        // Modify this line to split the train/test sample.
        let indices = (0..m).into_iter()
            .collect::<Vec<usize>>();


        let full_tree = construct_full_tree(
            sample,
            distribution,
            indices,
            self.criterion
        );


        todo!()
    }
}



/// TODO complete this function
/// Construct a full binary tree
/// that perfectly classify the given examples.
#[inline]
fn construct_full_tree<O, D>(sample:       &Sample<D>,
                             distribution: &[f64],
                             indices:      Vec<usize>,
                             criterion:    Criterion)
    -> Box<Node<StumpSplit<D, O>>>
    where D: Data<Output = O>,
          O: PartialOrd + Clone + DataBounds
{
    // TODO
    // This sentence may be redundant
    if indices.is_empty() {
        panic!("Empty tree is induced");
    }


    // TODO set impurity
    let impurity = f64::MAX;


    // TODO Implement the case where the all nodes have same label.
    // If all the labels are same,
    // returns a node that predicts the label.
    let mut labels = indices.iter()
        .map(|&i| sample[i].1);
    let l = labels.next().unwrap();
    if labels.all(|t| l == t) {
        let node = Node::leaf(l, impurity);
        return Box::new(node);
    }


    let dim = sample.dim();


    let mut best_index    = 0_usize;
    let mut best_split    = O::min_value();
    let mut best_decrease = f64::MAX;
    for j in 0..dim {
        let sub_sample = indices.iter()
            .map(|&i| {
                let (x, y) = &sample[i];
                (x.value_at(j), *y)
            })
            .collect::<Vec<_>>();

        let sub_dist = indices.iter()
            .map(|&i| distribution[i])
            .collect::<Vec<_>>();


        let (split, decrease) = find_best_split(sub_sample, sub_dist);

        if decrease < best_decrease {
            best_index    = j;
            best_split    = split;
            best_decrease = decrease;
        }
    }


    let rule = StumpSplit::from((best_index, best_split));



    let impurity = match criterion {
        Criterion::Entropy => entropic_impurity(&sample, &indices[..])
    };


    let (lidx, ridx) = indices.into_iter()
        .fold((Vec::new(), Vec::new()), |(mut l, mut r), i| {
            match rule.split(&sample[i].0) {
                LR::Left  => { l.push(i); },
                LR::Right => { r.push(i); }
            }
            (l, r)
        });


    // TODO
    // Get the left/right nodes
    let left = construct_full_tree(
        sample, distribution, lidx, criterion
    );
    let right = construct_full_tree(
        sample, distribution, ridx, criterion
    );


    Box::new(Node::branch(rule, left, right, impurity))
}


/// Returns the best split
/// that maximizes the decrease of impurity.
#[inline]
fn find_best_split<T>(sample:       Vec<(T, Label)>,
                      distribution: Vec<f64>)
    -> (T, f64)
    where T: PartialOrd + Clone + DataBounds
{
    // Sort the indices by the values in `sample` of type `T`.
    let m = sample.len();
    let mut indices = (0..m).into_iter().collect::<Vec<_>>();

    indices.sort_by(|&i, &j|
        sample[i].0.partial_cmp(&sample[j].0).unwrap()
    );


    let total_weight = distribution.iter().sum::<f64>();
    let mut left  = TempNodeInfo::new(0.0, total_weight);
    let mut right = {
        let _p = sample.iter()
            .zip(distribution.iter())
            .fold(0.0, |acc, ((_, y), &d)| {
                if *y > 0.0 { d + acc } else { acc }
            });
        TempNodeInfo::new(_p, total_weight)
    };


    // These variables are used for the best splitting rules.
    let mut best_decrease = right.entropic_impurity();
    let mut best_split = T::min_value();


    let mut iter = indices.into_iter().peekable();

    while let Some(i) = iter.next() {
        let (_, y) = &sample[i];
        let weight = distribution[i];
        if *y > 0.0 {
            left.positive_inc_by(weight);
            right.positive_dec_by(weight);
        } else {
            left.negative_inc_by(weight);
            right.negative_dec_by(weight);
        }


        let lp = left.total_weight / total_weight;
        let rp = 1.0 - lp;


        let decrease = lp * left.entropic_impurity()
            + rp * right.entropic_impurity();


        if decrease < best_decrease {
            best_decrease = decrease;
            best_split = match iter.peek() {
                Some(&index) => sample[index].0.clone(),
                None => T::max_value()
            };
        }
    }


    (best_split, best_decrease)
}


/// Some informations that are useful in `best_hypothesis(..)`.
struct TempNodeInfo {
    positive_weight: f64,
    total_weight:    f64,
}


impl TempNodeInfo {
    /// Build an instance of `TempNodeInfo`.
    fn new(positive_weight: f64, total_weight: f64) -> Self {
        Self { positive_weight, total_weight }
    }


    /// Returns the impurity of this node.
    fn entropic_impurity(&self) -> f64 {
        if self.total_weight == 0.0 {
            return 0.0;
        }


        let p = self.positive_weight / self.total_weight;
        let n = 1.0 - p;
        - p * p.ln() - n * n.ln()
    }


    /// Increase the number of positive examples by one.
    fn positive_inc_by(&mut self, weight: f64) {
        self.positive_weight += weight;
        self.total_weight    += weight;
    }


    /// Decrease the number of positive examples by one.
    fn positive_dec_by(&mut self, weight: f64) {
        self.positive_weight -= weight;
        self.total_weight    -= weight;
    }


    /// Increase the number of negative examples by one.
    fn negative_inc_by(&mut self, weight: f64) {
        self.total_weight += weight;
    }


    /// Decrease the number of negative examples by one.
    fn negative_dec_by(&mut self, weight: f64) {
        self.total_weight -= weight;
    }
}



// TODO: Implement the code that handle the multi-class case.
/// Compute the binary entropy of the given subsample.
#[inline]
fn entropic_impurity<D>(sample: &Sample<D>, indices: &[usize])
    -> f64
{
    let sample_size = indices.len() as f64;

    let pos_cnt = indices.into_iter()
        .filter(|&&i| {
            let (_, l) = sample[i];
            l > 0.0
        })
        .count() as f64;


    let p = pos_cnt / sample_size;


    - p * p.ln() - (1.0 - p) * (1.0 - p).ln()
}

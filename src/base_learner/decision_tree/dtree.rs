// TODO LIST
//  * Implement `best_hypothesis`
//      - Train/test split
//      o construct_full_tree
//      - prune
//  * Implement `construct_full_tree`
//      ? Compute impurity
//          o entropic impurity
//          - gini impurity
//      - Test whether this function works as expected.
//  * Implement `pruing`
//      - Cross-validation
// 
//  * Implement `train_test_split` to find the best pruning
//  * Test code
//      - construct_full_tree
//      - pruning
//  * Run boosting

use crate::{DataBounds, Data, Sample};
use crate::BaseLearner;


use super::dtree_classifier::DTreeClassifier;
use super::split_rule::*;
use super::node::*;
use super::measure::*;


use std::marker::PhantomData;
use std::collections::HashMap;



/// Enumerate of split rule.
/// This enumeration will be updated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    /// Split based on feature (decision stump).
    Feature,
}


/// Generates a `DTreeClassifier` for a given distribution
/// over examples.
pub struct DTree<S, L> {
    criterion: Criterion,
    _phantom:  PhantomData<(S, L)>,
}


impl<S, L> DTree<S, L> {
    /// Initialize `DTree`.
    #[inline]
    pub fn init<D>(_sample: &Sample<D, L>) -> Self
    {
        Self {
            criterion: Criterion::Entropy,
            _phantom:  PhantomData
        }
    }
}


// TODO remove `std::fmt::Debug` trait bound
impl<S, O, D, L> BaseLearner<D, L> for DTree<S, L>
    // where S: SplitRule<Input = D> + std::fmt::Debug,
    where S: SplitRule<D> + std::fmt::Debug,
          D: Data<Output = O> + std::fmt::Debug,
          L: PartialEq + Eq + std::hash::Hash + Clone,
          O: PartialOrd + Clone + DataBounds + std::fmt::Debug
{
    type Clf = DTreeClassifier<S, L>;
    fn best_hypothesis(&self, sample: &Sample<D, L>, distribution: &[f64])
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
fn construct_full_tree<O, D, L>(sample:       &Sample<D, L>,
                                distribution: &[f64],
                                indices:      Vec<usize>,
                                criterion:    Criterion)
    -> Box<Node<StumpSplit<D, O>, L>>
    where D: Data<Output = O> + std::fmt::Debug,
          L: PartialEq + Eq + std::hash::Hash + Clone,
          O: PartialOrd + Clone + DataBounds + std::fmt::Debug,
{
    // TODO
    // This sentence may be redundant
    if indices.is_empty() {
        panic!("Empty tree is induced");
    }


    // TODO set impurity
    let impurity = f64::MAX;


    let mut labels = indices.iter()
        .map(|&i| sample[i].1.clone());
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

        let (split, decrease) = find_best_split(
            sample,
            &distribution[..],
            &indices[..],
            j
        );

        if decrease < best_decrease {
            best_index    = j;
            best_split    = split;
            best_decrease = decrease;
        }
    }


    let rule = StumpSplit::from((best_index, best_split));

    println!("rule: {rule:?}");



    let impurity = match criterion {
        Criterion::Entropy
            => entropic_impurity(&sample, &distribution[..], &indices[..])
    };


    let (lidx, ridx) = indices.into_iter()
        .fold((Vec::new(), Vec::new()), |(mut l, mut r), i| {
            let (x, _) = &sample[i];
            match rule.split(x) {
                LR::Left  => { l.push(i); },
                LR::Right => { r.push(i); }
            }
            (l, r)
        });


    // DEBUG
    println!("left: {lidx:?}, right: {ridx:?}");


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
fn find_best_split<D, O, L>(sample:  &Sample<D, L>,
                            dist:    &[f64],
                            indices: &[usize],
                            index:    usize)
    -> (O, f64)
    where D: Data<Output = O>,
          O: PartialOrd + DataBounds,
          L: PartialEq + Eq + std::hash::Hash + Clone,
{
    // Sort the indices by the values in `sample` of type `D`.
    let mut indices = indices.into_iter()
        .map(|&i| i)
        .collect::<Vec<_>>();

    indices.sort_by(|&i, &j| {
        let (xi, _) = &sample[i];
        let (xj, _) = &sample[j];
        xi.value_at(index)
            .partial_cmp(&xj.value_at(index))
            .unwrap()
    });


    let total_weight = dist.iter().sum::<f64>();
    let mut left  = TempNodeInfo::empty();
    let mut right = TempNodeInfo::new(sample, dist, &indices[..]);


    // These variables are used for the best splitting rules.
    let mut best_decrease = right.entropic_impurity();
    let mut best_split = O::min_value();


    let mut iter = indices.into_iter().peekable();

    while let Some(i) = iter.next() {
        let (_, y) = &sample[i];
        let weight = dist[i];
        left.insert(y.clone(), weight);
        right.delete(y.clone(), weight);


        let lp = left.total / total_weight;
        let rp = 1.0 - lp;


        let decrease = lp * left.entropic_impurity()
            + rp * right.entropic_impurity();


        if decrease < best_decrease {
            best_decrease = decrease;
            best_split = match iter.peek() {
                Some(&j) => {
                    let (data, _) = &sample[j];
                    data.value_at(index)
                },
                None => O::max_value()
            };
        }
    }


    (best_split, best_decrease)
}


/// Some informations that are useful in `best_hypothesis(..)`.
struct TempNodeInfo<L>
    where L: std::hash::Hash + PartialEq + Eq
{
    map:   HashMap<L, f64>,
    total: f64,
}


impl<L> TempNodeInfo<L>
    where L: std::hash::Hash + PartialEq + Eq + Clone
{
    /// Build an empty instance of `TempNodeInfo`.
    #[inline(always)]
    pub(self) fn empty() -> Self {
        Self {
            map:   HashMap::new(),
            total: 0.0_f64,
        }
    }


    /// Build an instance of `TempNodeInfo`.
    #[inline(always)]
    pub(self) fn new<D>(sample:  &Sample<D, L>,
                        dist:    &[f64],
                        indices: &[usize])
        -> Self
    {
        let map = indices.iter()
            .fold(HashMap::new(), |mut mp, &i| {
                let (_, l) = &sample[i];

                if let Some(cnt) = mp.get_mut(l) {
                    *cnt += dist[i];
                } else {
                    mp.insert(l.clone(), dist[i]);
                }
                mp
            });

        let total = indices.iter()
            .fold(0.0_f64, |acc, &i| acc + dist[i]);
        Self { map, total }
    }


    /// Returns the impurity of this node.
    #[inline(always)]
    pub(self) fn entropic_impurity(&self) -> f64 {
        if self.total == 0.0 || self.map.is_empty() { return 0.0; }

        self.map.iter()
            .map(|(_, &p)| {
                let r = p / self.total;
                -r * r.ln()
            })
            .sum::<f64>()
    }


    /// Increase the number of positive examples by one.
    pub(self) fn insert(&mut self, label: L, weight: f64) {
        let cnt = self.map.entry(label).or_insert(0.0);
        *cnt += weight;
        self.total += weight;

    }


    /// Decrease the number of positive examples by one.
    pub(self) fn delete(&mut self, label: L, weight: f64) {
        *self.map.get_mut(&label).unwrap() -= weight;


        if *self.map.get(&label).unwrap() == 0.0 {
            self.map.remove(&label);
        }


        self.total -= weight;
    }
}



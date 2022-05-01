// TODO LIST
//  * Implement `best_hypothesis`
//      x Train/test split
//      x construct_full_tree
//      - prune
//  * Implement `construct_full_tree`
//      ? Compute impurity
//          x entropic impurity
//          - gini impurity
//      - Test whether this function works as expected.
//  * Implement `pruning`
//      - Cross-validation
// 
//  * Test code
//      - construct_full_tree
//      - pruning
//  * Run boosting
//  * Remove `print` for debugging
//  * Add a member `mistake_ratio` to each branch/leaf node.
//  * Each node has `impurity` member, but it may be redundant


use rand::prelude::*;

use crate::{DataBounds, Data, Sample};
use crate::BaseLearner;


use super::dtree_classifier::DTreeClassifier;
use super::split_rule::*;
use super::node::*;
use super::measure::*;


use std::rc::Rc;
use std::cell::RefCell;
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
    rng:         RefCell<ThreadRng>,
    criterion:   Criterion,
    train_ratio: f64,
    _phantom:    PhantomData<(S, L)>,
}


impl<S, L> DTree<S, L> {
    /// Initialize `DTree`.
    #[inline]
    pub fn init<D>(_sample: &Sample<D, L>) -> Self
    {
        let rng         = RefCell::new(rand::thread_rng());
        let criterion   = Criterion::Entropy;
        let train_ratio = 0.8_f64;
        Self {
            rng,
            criterion,
            train_ratio,
            _phantom: PhantomData
        }
    }


    /// Set the training ratio.
    /// Default ratio is `0.8`.
    #[inline]
    pub fn with_train_ratio(mut self, ratio: f64) -> Self {
        assert!(0.0 <= ratio && ratio <= 1.0);
        self.train_ratio = ratio;

        self
    }



    /// Set the criterion that measures a node impurity.
    /// Default criterion is `Criterion::Entropy`.
    #[inline]
    pub fn with_criterion(mut self, criterion: Criterion) -> Self {
        self.criterion = criterion;

        self
    }
}


// TODO remove `std::fmt::Debug` trait bound
impl<S, O, D, L> BaseLearner<D, L> for DTree<S, L>
    where S: SplitRule<D> + std::fmt::Debug,
          D: Data<Output = O> + std::fmt::Debug,
          L: PartialEq + Eq + std::hash::Hash + Clone + std::fmt::Debug,
          O: PartialOrd + Clone + DataBounds + std::fmt::Debug
{
    type Clf = DTreeClassifier<S, L>;
    fn best_hypothesis(&self,
                       sample: &Sample<D, L>,
                       distribution: &[f64])
        -> Self::Clf
    {
        let m = sample.len();


        // TODO
        // Modify this line to split the train/test sample.
        let mut indices = (0..m).into_iter()
            .collect::<Vec<usize>>();


        // Shuffle indices
        let rng: &mut ThreadRng = &mut self.rng.borrow_mut();
        indices.shuffle(rng);


        let train_size = (self.train_ratio * m as f64).floor() as usize;
        let train_indices = indices.drain(..train_size)
            .collect::<Vec<_>>();
        let test_indices = indices;


        println!("Train indices: {train_indices:?}");


        let mut tree = construct_full_tree(
            sample,
            distribution,
            train_indices,
            self.criterion
        );


        // TODO train_err is not proper one.
        // Maybe I should replace train_err to test_err.


        prune(&mut tree, sample, distribution, test_indices, self.criterion);

        println!("# of leaves: {}", tree.leaves());


        println!("Resulting tree:\n{tree:?}");



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
    -> Rc<Node<StumpSplit<D, O>, L>>
    where D: Data<Output = O> + std::fmt::Debug,
          L: PartialEq + Eq + std::hash::Hash + Clone,
          O: PartialOrd + Clone + DataBounds + std::fmt::Debug,
{
    // TODO
    // This sentence may be redundant
    if indices.is_empty() {
        panic!("Empty tree is induced");
    }


    let (label, train_err) = calc_train_err(
        sample, distribution, &indices[..]
    );


    // Compute the node impurity
    let (split, decrease) = find_best_split(
        sample, &distribution[..], &indices[..], 0
    );


    let mut best_index    = 0_usize;
    let mut best_split    = split;
    let mut best_decrease = decrease;


    let dim = sample.dim();
    for j in 1..dim {

        let (split, decrease) = find_best_split(
            sample, &distribution[..], &indices[..], j
        );


        if decrease <= best_decrease {
            best_index    = j;
            best_split    = split;
            best_decrease = decrease;
        }
    }


    let rule = StumpSplit::from((best_index, best_split));


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


    // If the split has no meaning, construct a leaf node.
    if lidx.is_empty() || ridx.is_empty() {
        let leaf = Node::leaf(label, train_err, impurity);
        return Rc::new(leaf);
    }


    // DEBUG
    println!("rule: {rule:?}");
    println!("left: {lidx:?}, right: {ridx:?}");


    // TODO
    // Get the left/right nodes
    let left = construct_full_tree(
        sample, distribution, lidx, criterion
    );
    let right = construct_full_tree(
        sample, distribution, ridx, criterion
    );


    Rc::new(Node::branch(rule, left, right, label, train_err, impurity))
}


/// Returns the best split
/// that maximizes the decrease of impurity.
#[inline]
fn find_best_split<D, O, L>(sample:  &Sample<D, L>,
                            dist:    &[f64],
                            indices: &[usize],
                            index:    usize)
    -> (O, Impurity)
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


    let total_weight = indices.iter()
        .fold(0.0, |acc, &i| acc + dist[i]);
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


        let decrease = Impurity::from(lp) * left.entropic_impurity()
            + Impurity::from(rp) * right.entropic_impurity();


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
                let l = l.clone();
                let cnt = mp.entry(l).or_insert(0.0);
                *cnt += dist[i];
                mp
            });

        let total = indices.iter()
            .fold(0.0_f64, |acc, &i| acc + dist[i]);
        Self { map, total }
    }


    /// Returns the impurity of this node.
    #[inline(always)]
    pub(self) fn entropic_impurity(&self) -> Impurity {
        if self.total == 0.0 || self.map.is_empty() { return 0.0.into(); }

        self.map.iter()
            .map(|(_, &p)| {
                let r = p / self.total;
                -r * r.ln()
            })
            .sum::<f64>()
            .into()
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


        if *self.map.get(&label).unwrap() <= 0.0 {
            self.map.remove(&label);
        }


        self.total -= weight;
    }
}



/// Returns the training error
#[inline]
fn calc_train_err<D, L>(sample:  &Sample<D, L>,
                        dist:    &[f64],
                        indices: &[usize])
    -> (L, NodeError)
    where L: Eq + std::hash::Hash + Clone
{
    let mut counter = HashMap::new();
    for &i in indices {
        let (_, l) = &sample[i];
        let l = l.clone();
        let cnt = counter.entry(l).or_insert(0.0);
        *cnt += dist[i];
    }


    let total = counter.values().sum::<f64>();


    // Compute the max (key, val) that has maximal p(j, t)
    let (l, p) = counter.into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();


    let node_err = total * (1.0 - (p / total));


    (l, node_err.into())
}


/// Prune the full-binary tree.
#[inline]
fn prune<D, S, L>(root: &mut Rc<Node<S, L>>,
                  sample: &Sample<D, L>,
                  dist: &[f64],
                  indices: Vec<usize>,
                  criterion: Criterion)
    where L: Clone
{
    Rc::get_mut(root).unwrap().pre_process();
    todo!()
}




use polars::prelude::*;
use rayon::prelude::*;

use rand::prelude::*;
use rand::rngs::StdRng;


use crate::BaseLearner;


use super::node::*;
use super::split_rule::*;
use super::train_node::*;
use super::dtree_classifier::DTreeClassifier;


use std::rc::Rc;
use std::ops::Deref;
use std::cell::{Ref, RefCell};
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
pub struct DTree {
    rng: RefCell<StdRng>,
    criterion: Criterion,
    split_rule: Split,
    train_ratio: f64,
    max_depth: Option<usize>,
}


impl DTree {
    /// Initialize `DTree`.
    #[inline]
    pub fn init(_df: &DataFrame) -> Self {
        let seed: u64 = 0;
        let rng = RefCell::new(SeedableRng::seed_from_u64(seed));
        let criterion   = Criterion::Entropy;
        let split_rule  = Split::Feature;
        let train_ratio = 0.8_f64;
        let max_depth   = None;
        Self {
            rng,
            criterion,
            split_rule,
            train_ratio,
            max_depth,
        }
    }


    /// Initialize the RNG by `seed`.
    /// If you don't use this method, 
    /// `DTree` initializes RNG by `0_u64`.
    pub fn seed(self, seed: u64) -> Self {
        let rng: StdRng = SeedableRng::seed_from_u64(seed);
        *self.rng.borrow_mut() = rng;

        self
    }


    /// Specify the maximal depth of the tree.
    /// Default maximul depth is `None`.
    pub fn max_depth(mut self, depth: usize) -> Self {
        assert!(depth > 0);
        self.max_depth = Some(depth);

        self
    }


    /// Set the ratio used for growing a tree.
    /// The rest examples are for pruning.
    /// Default ratio is `0.8`.
    #[inline]
    pub fn with_grow_ratio(mut self, ratio: f64) -> Self {
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


impl BaseLearner for DTree {
    type Clf = DTreeClassifier;
    fn produce(&self,
               data: &DataFrame,
               target: &Series,
               distribution: &[f64])
        -> Self::Clf
    {
        let (m, _) = data.shape();


        let mut indices = (0..m).into_iter()
            .collect::<Vec<usize>>();


        // Shuffle indices
        let rng: &mut StdRng = &mut self.rng.borrow_mut();
        indices.shuffle(rng);


        let train_size = (self.train_ratio * m as f64).floor() as usize;
        let train_indices = indices.drain(..train_size)
            .collect::<Vec<_>>();
        let test_indices = indices;


        let mut tree = match self.split_rule {
            Split::Feature =>
                stump_fulltree(
                    data,
                    target,
                    distribution,
                    train_indices,
                    test_indices,
                    self.criterion,
                    self.max_depth.clone(),
                ),
        };


        prune(&mut tree);


        let root = Node::from(
            Rc::try_unwrap(tree)
                .expect("Root node has reference counter >= 1")
                .into_inner()
        );

        DTreeClassifier::from(root)
    }
}



/// Construct a full binary tree
/// that perfectly classify the given examples.
#[inline]
fn stump_fulltree(data: &DataFrame,
                  target: &Series,
                  dist: &[f64],
                  train: Vec<usize>,
                  test: Vec<usize>,
                  criterion: Criterion,
                  max_depth: Option<usize>)
    -> Rc<RefCell<TrainNode>>
{
    // Compute the best prediction that minimizes the training error
    // on this node.
    let (pred, train_err) = calc_train_err(target, dist, &train[..]);


    let test_err = calc_test_err(target, dist, &test[..], pred);


    let node_err = NodeError::from((train_err, test_err));


    // If sum of `dist` over `train` is zero, construct a leaf node.
    if train_err == 0.0 {
        let leaf = TrainNode::leaf(pred, node_err);
        return Rc::new(RefCell::new(leaf));
    }


    let (_, feature, threshold) = data.get_columns()
        .into_par_iter()
        .map(|column| {
            let (thr, dec) = find_best_split(
                column, target, &dist[..], &train[..]
            );
            (dec, column.name(), thr)
        })
        .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
        .expect("No feature that descrease impurity");


    let rule = SplitRule::create_stump(feature, threshold);


    let mut ltrain = Vec::new();
    let mut rtrain = Vec::new();
    for i in train.into_iter() {
        match rule.split(data, i) {
            LR::Left  => { ltrain.push(i); },
            LR::Right => { rtrain.push(i); },
        }
    }


    let mut ltest = Vec::new();
    let mut rtest = Vec::new();
    for i in test.into_iter() {
        match rule.split(data, i) {
            LR::Left  => { ltest.push(i); },
            LR::Right => { rtest.push(i); },
        }
    }


    // If the split has no meaning, construct a leaf node.
    if ltrain.is_empty() || rtrain.is_empty() {
        let leaf = TrainNode::leaf(pred, node_err);
        return Rc::new(RefCell::new(leaf));
    }


    let left;
    let right;
    match max_depth {
        Some(depth) => {
            if depth == 1 {
                left = construct_leaf(target, dist, ltrain, ltest);
                right = construct_leaf(target, dist, rtrain, rtest);
            } else {
                let d = Some(depth - 1);
                left = stump_fulltree(
                    data, target, dist, ltrain, ltest, criterion, d
                );
                right = stump_fulltree(
                    data, target, dist, rtrain, rtest, criterion, d
                );
            }
        },
        None => {
                left = stump_fulltree(
                    data, target, dist, ltrain, ltest, criterion, None
                );
                right = stump_fulltree(
                    data, target, dist, rtrain, rtest, criterion, None
                );
        }
    }


    // let left = stump_fulltree(data, target, dist, ltrain, ltest, criterion);
    // let right = stump_fulltree(data, target, dist, rtrain, rtest, criterion);


    Rc::new(RefCell::new(TrainNode::branch(
        rule, left, right, pred, node_err
    )))
}


#[inline]
fn construct_leaf(target: &Series,
                  dist: &[f64],
                  train: Vec<usize>,
                  test: Vec<usize>)
    -> Rc<RefCell<TrainNode>>
{
    // Compute the best prediction that minimizes the training error
    // on this node.
    let (pred, train_err) = calc_train_err(target, dist, &train[..]);


    let test_err = calc_test_err(target, dist, &test[..], pred);


    let node_err = NodeError::from((train_err, test_err));


    let leaf = TrainNode::leaf(pred, node_err);
    Rc::new(RefCell::new(leaf))
}


/// Returns the best split
/// that maximizes the decrease of impurity.
#[inline]
fn find_best_split(data: &Series,
                   target: &Series,
                   dist: &[f64],
                   indices: &[usize])
    -> (f64, Impurity)
{
    let target = target.i64()
        .expect("The target class is not a dtype i64");


    let data = data.f64()
        .expect("The data is not a dtype f64");


    let mut triplets = indices.into_iter()
        .copied()
        .map(|i| {
            let val = data.get(i).unwrap();
            let lab = target.get(i).unwrap();
            (val, dist[i], lab)
        })
        .collect::<Vec<(f64, f64, i64)>>();
    triplets.sort_by(|(x, _, _), (y, _, _)| x.partial_cmp(&y).unwrap());


    let total_weight = triplets.par_iter()
        .map(|(_, d, _)| d)
        .sum::<f64>();
    let mut left = TempNodeInfo::empty();
    let mut right = TempNodeInfo::new(&triplets[..]);


    let mut iter = triplets.into_iter().peekable();


    // These variables are used for the best splitting rules.
    let mut best_decrease = right.entropic_impurity();
    let mut best_threshold = iter.peek()
        .map(|(v, _, _)| *v - 2.0_f64)
        .unwrap_or(f64::MIN);

    while let Some((old_val, d, y)) = iter.next() {
        left.insert(y, d);
        right.delete(y, d);


        while let Some(&(xx, dd, yy)) = iter.peek() {
            if xx != old_val { break; }

            left.insert(yy, dd);
            right.delete(yy, dd);

            iter.next();
        }

        let new_val = iter.peek()
            .map(|(xx, _, _)| *xx)
            .unwrap_or(old_val + 2.0_f64);

        let threshold = (old_val + new_val) / 2.0;


        let lp = left.total / total_weight;
        let rp = 1.0 - lp;


        let decrease = Impurity::from(lp) * left.entropic_impurity()
            + Impurity::from(rp) * right.entropic_impurity();


        if decrease < best_decrease {
            best_decrease = decrease;
            best_threshold = threshold;
        }
    }



    (best_threshold, best_decrease)
}


/// Some information that are useful in `produce(..)`.
struct TempNodeInfo {
    map: HashMap<i64, f64>,
    total: f64,
}


impl TempNodeInfo {
    /// Build an empty instance of `TempNodeInfo`.
    #[inline(always)]
    pub(self) fn empty() -> Self {
        Self {
            map: HashMap::new(),
            total: 0.0_f64,
        }
    }


    /// Build an instance of `TempNodeInfo`.
    #[inline(always)]
    pub(self) fn new(triplets: &[(f64, f64, i64)]) -> Self {
        let mut total = 0.0_f64;
        let mut map = HashMap::new();
        for (_, d, l) in triplets {
            total += *d;
            let cnt = map.entry(*l).or_insert(0.0);
            *cnt += *d;
        }
        Self { map, total }
    }


    /// Returns the impurity of this node.
    #[inline(always)]
    pub(self) fn entropic_impurity(&self) -> Impurity {
        if self.total == 0.0 || self.map.is_empty() { return 0.0.into(); }

        self.map.par_iter()
            .map(|(_, &p)| {
                let r = p / self.total;
                if p == 0.0 { 0.0 } else { -r * r.ln() }
            })
            .sum::<f64>()
            .into()
    }


    /// Increase the number of positive examples by one.
    pub(self) fn insert(&mut self, label: i64, weight: f64) {
        let cnt = self.map.entry(label).or_insert(0.0);
        *cnt += weight;
        self.total += weight;
    }


    /// Decrease the number of positive examples by one.
    pub(self) fn delete(&mut self, label: i64, weight: f64) {
        if let Some(key) = self.map.get_mut(&label) {
            *key -= weight;
            if *self.map.get(&label).unwrap() == 0.0 {
                self.map.remove(&label);
            }
            self.total -= weight;
        }
    }
}



/// Returns
/// - a label that minimizes the training error,
/// - the error.
#[inline]
fn calc_train_err(target: &Series, dist: &[f64], indices: &[usize])
    -> (i64, f64)
{
    let target = target.i64()
        .expect("The target class df[{s}] is not a dtype i64");
    let mut counter = HashMap::new();
    for &i in indices {
        let l = target.get(i).unwrap();
        let cnt = counter.entry(l).or_insert(0.0);
        *cnt += dist[i];
    }


    let total = counter.values().sum::<f64>();


    // Compute the max (key, val) that has maximal p(j, t)
    let (l, p) = counter.into_par_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();


    // From the update rule of boosting algorithm,
    // the sum of `dist` over `indices` may become zero,
    let node_err = if total > 0.0 {
            total * (1.0 - (p / total))
        } else {
            0.0
        };


    (l, node_err)
}


#[inline]
fn calc_test_err(target: &Series, dist: &[f64], indices: &[usize], pred: i64)
    -> f64
{
    let target = target.i64()
        .expect("The target class df[{s}] is not a dtype i64");
    let total = indices.par_iter().map(|&i| dist[i]).sum::<f64>();


    if total == 0.0 {
        return 0.0;
    }


    let wrong = indices.iter()
        .filter_map(|&i| {
            let l = target.get(i).unwrap();
            if l != pred { Some(dist[i]) } else { None }
        })
        .sum::<f64>();


    wrong / total
}


/// Prune the full-binary tree.
#[inline]
fn prune(root: &mut Rc<RefCell<TrainNode>>) {
    // Construct the sub-tree that achieves the same training error
    // with fewer leaves.
    Rc::get_mut(root).unwrap()
        .borrow_mut()
        .pre_process();


    // Sort the nodes in the tree by the `alpha` value.
    let links = weak_links(root);


    // Prune the nodes 
    // while node error is lower than or equals to tree error
    for node in links {
        let node_err = node.borrow().node_error();
        let tree_err = node.borrow().tree_error();

        if node_err.test >= tree_err.test {
            break;
        }


        node.borrow_mut().prune();
    }
}



#[inline]
fn weak_links(root: &Rc<RefCell<TrainNode>>)
    -> Vec<Rc<RefCell<TrainNode>>>
{
    let mut links = Vec::new();
    let mut stack = Vec::from([Rc::clone(root)]);
    while let Some(node) = stack.pop() {
        match get_ref(&node).deref() {
            TrainNode::Leaf(_) => {
                continue;
            },
            TrainNode::Branch(ref branch) => {
                stack.push(Rc::clone(&branch.left));
                stack.push(Rc::clone(&branch.right));
            },
        }


        links.push(node);
    }


    links.sort_by(|u, v|
        u.borrow().alpha().partial_cmp(&v.borrow().alpha()).unwrap()
    );

    links
}



struct TrainNodeGuard<'a> {
    guard: Ref<'a, TrainNode>
}


impl<'b> Deref for TrainNodeGuard<'b> {
    type Target = TrainNode;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}


pub(self) fn get_ref(node: &Rc<RefCell<TrainNode>>) -> TrainNodeGuard
{
    TrainNodeGuard {
        guard: node.borrow()
    }
}



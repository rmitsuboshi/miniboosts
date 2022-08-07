// TODO
// set_tree_error_train ?
// test/tree errors are redundant?
use polars::prelude::*;
use rayon::prelude::*;

use rand::prelude::*;
use rand::rngs::StdRng;


use crate::BaseLearner;


use super::node::*;
use super::criterion::*;
use super::split_rule::*;
use super::train_node::*;
use super::dtree_classifier::DTreeClassifier;


use std::rc::Rc;
use std::ops::Deref;
use std::cell::RefCell;
use std::collections::HashMap;



/// Generates a `DTreeClassifier` for a given distribution
/// over examples.
pub struct DTree {
    rng: RefCell<StdRng>,
    criterion: Criterion,
    train_ratio: f64,
    max_depth: Option<usize>,
    prune: bool,
}


impl DTree {
    /// Initialize `DTree`.
    #[inline]
    pub fn init(_df: &DataFrame) -> Self {
        let seed: u64 = 0;
        let rng = RefCell::new(SeedableRng::seed_from_u64(seed));
        let criterion   = Criterion::Entropy;
        let train_ratio = 1.0_f64;
        let max_depth   = None;
        let prune = false;

        Self {
            rng,
            criterion,
            train_ratio,
            max_depth,
            prune,
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
    /// Default ratio is `1.0`.
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


    /// Set the pruning parameter.
    /// If `true`, the resulting tree is the pruned one.
    /// Default value is `false.`
    #[inline]
    pub fn prune(mut self, p: bool) -> Self {
        self.prune = p;
        self
    }


    /// Set criterion for node splitting.
    #[inline]
    pub fn criterion(mut self, criterion: Criterion) -> Self {
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


        let train_size = (self.train_ratio * m as f64).ceil() as usize;
        let train_indices = indices.drain(..train_size)
            .into_iter()
            .filter(|&i| distribution[i] > 0.0)
            .collect::<Vec<_>>();
        let test_indices = indices;


        // Construct a large binary tree
        let mut tree = full_tree(
            data,
            target,
            distribution,
            train_indices,
            test_indices,
            self.criterion,
            self.max_depth.clone(),
        );


        // Prune the nodes
        if self.prune {
            prune(&mut tree);
        }


        tree.borrow_mut().remove_parent();


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
fn full_tree(data: &DataFrame,
             target: &Series,
             dist: &[f64],
             train: Vec<usize>,
             test: Vec<usize>,
             criterion: Criterion,
             max_depth: Option<usize>)
    -> Rc<RefCell<TrainNode>>
{
    let train_total_weight = train.iter()
        .copied()
        .map(|i| dist[i])
        .sum::<f64>();

    let test_total_weight = test.iter()
        .copied()
        .map(|i| dist[i])
        .sum::<f64>();
    // Compute the best prediction that minimizes the training error
    // on this node.
    let (pred, train_err) = calc_train_error_as_leaf(
        target, dist, &train[..]
    );


    let test_err = calc_test_error_as_leaf(target, dist, &test[..], pred);


    // If sum of `dist` over `train` is zero, construct a leaf node.
    if train_err == 0.0 {
        println!("zero train err");
        return TrainNode::leaf(
            pred, train_total_weight, train_err, test_total_weight, test_err
        );
    }


    // Find the best splitting rule.
    let (feature, threshold) = match criterion {
        // FInd a split that minimizes the entropic impurity
        Criterion::Entropy => {
            let (_, feature, threshold) = data.get_columns()
                .into_par_iter()
                .map(|column| {
                    let (thr, dec) = find_best_split_entropy(
                        column, target, &dist[..], &train[..]
                    );
                    (dec, column.name(), thr)
                })
                .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
                .expect("No feature that descrease impurity");
            (feature, threshold)
        },
        // Find a split that maximizes the edge
        Criterion::Edge => {
            let (_, feature, threshold) = data.get_columns()
                .into_par_iter()
                .map(|column| {
                    let (thr, edge) = find_best_split_edge(
                        column, target, &dist[..], &train[..]
                    );
                    (edge, column.name(), thr)
                })
                .max_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
                .expect("No feature that that maximizes the edge");
            (feature, threshold)
        },
    };
    let rule = Splitter::new(feature, threshold);


    // Split the train data for left/right childrens
    let mut ltrain = Vec::new();
    let mut rtrain = Vec::new();
    for i in train.into_iter() {
        match rule.split(data, i) {
            LR::Left  => { ltrain.push(i); },
            LR::Right => { rtrain.push(i); },
        }
    }



    // Split the test data for left/right childrens
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
        return TrainNode::leaf(
            pred, train_total_weight, train_err, test_total_weight, test_err
        );
    }


    // grow the tree.
    let left;
    let right;
    match max_depth {
        Some(depth) => {
            if depth == 1 {
                // If `depth == 1`,
                // the childs from this node must be leaves.
                left = construct_leaf(target, dist, ltrain, ltest);
                right = construct_leaf(target, dist, rtrain, rtest);
            } else {
                // If `depth > 1`,
                // the childs from this node might be branches.
                let d = Some(depth - 1);
                left = full_tree(
                    data, target, dist, ltrain, ltest, criterion, d
                );
                right = full_tree(
                    data, target, dist, rtrain, rtest, criterion, d
                );
            }
        },
        None => {
            left = full_tree(
                data, target, dist, ltrain, ltest, criterion, None
            );
            right = full_tree(
                data, target, dist, rtrain, rtest, criterion, None
            );
        }
    }


    TrainNode::branch(
        rule, left, right, pred,
        train_total_weight, train_err, test_total_weight, test_err
    )
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
    let (pred, train_err) = calc_train_error_as_leaf(target, dist, &train[..]);


    let test_err = calc_test_error_as_leaf(target, dist, &test[..], pred);


    let train_total_weight = train.iter()
        .copied()
        .map(|i| dist[i])
        .sum::<f64>();

    let test_total_weight = test.iter()
        .copied()
        .map(|i| dist[i])
        .sum::<f64>();

    TrainNode::leaf(
        pred, train_total_weight, train_err, test_total_weight, test_err
    )
}


/// Returns the best split
/// that maximizes the decrease of impurity.
/// Here, the impurity is
/// `- \sum_{l} p(l) \ln [ p(l) ]`,
/// where `p(l)` is the total weight of class `l`.
#[inline]
fn find_best_split_entropy(data: &Series,
                           target: &Series,
                           dist: &[f64],
                           indices: &[usize])
    -> (f64, Impurity)
{
    let target = target.i64()
        .expect("The target class is not a dtype i64");


    let data = data.f64()
        .expect("The data is not a dtype f64");


    let mut triplets = indices.into_par_iter()
        .copied()
        .map(|i| {
            let x = data.get(i).unwrap();
            let y = target.get(i).unwrap();
            (x, dist[i], y)
        })
        .collect::<Vec<(f64, f64, i64)>>();
    triplets.sort_by(|(x1, _, _), (x2, _, _)| x1.partial_cmp(&x2).unwrap());


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

        assert!(total_weight > 0.0);

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


/// Returns the best split
/// that maximizes the edge.
/// Here, edge is the weighted accuracy.
/// Given a distribution `dist[..]` over the training examples,
/// the edge is `\sum_{i} dist[i] y[i] h(x[i])`
/// where `(x[i], y[i])` is the `i`th training example.
#[inline]
fn find_best_split_edge(data: &Series,
                        target: &Series,
                        dist: &[f64],
                        indices: &[usize])
    -> (f64, Edge)
{
    let target = target.i64()
        .expect("The target class is not a dtype i64");


    let data = data.f64()
        .expect("The data is not a dtype f64");


    let mut triplets = indices.into_par_iter()
        .copied()
        .map(|i| {
            let x = data.get(i).unwrap();
            let y = target.get(i).unwrap() as f64;
            (x, dist[i], y)
        })
        .collect::<Vec<(f64, f64, f64)>>();
    triplets.sort_by(|(x1, _, _), (x2, _, _)| x1.partial_cmp(&x2).unwrap());


    // Compute the edge of the hypothesis that predicts `+1`
    // for all instances.
    let mut edge = triplets.iter()
        .map(|(_, d, y)| *d * *y)
        .sum::<f64>();




    let mut iter = triplets.into_iter().peekable();


    // best threshold is the smallest value.
    // we define the initial threshold as the smallest value minus 2.0
    let mut best_threshold = iter.peek()
        .map(|(v, _, _)| *v - 2.0_f64)
        .unwrap_or(f64::MIN);

    // best edge
    let mut best_edge = edge.abs();

    while let Some((left, d, y)) = iter.next() {
        edge -= 2.0 * d * y;


        while let Some(&(xx, dd, yy)) = iter.peek() {
            if xx != left { break; }

            edge -= 2.0 * dd * yy;

            iter.next();
        }

        let right = iter.peek()
            .map(|(xx, _, _)| *xx)
            .unwrap_or(left + 2.0_f64);

        let threshold = (left + right) / 2.0;


        if best_edge < edge.abs() {
            best_edge = edge.abs();
            best_threshold = threshold;
        }
    }


    (best_threshold, Edge::from(best_edge))
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
                if r == 0.0 { 0.0 } else { -r * r.ln() }
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



/// This function returns a tuple `(y, e)` where
/// - `y` is the prediction label that minimizes the training error.
/// - `e` is the training error when the prediction is `y`.
#[inline]
fn calc_train_error_as_leaf(target: &Series, dist: &[f64], indices: &[usize])
    -> (i64, f64)
{
    let target = target.i64()
        .expect("The target class is not a dtype i64");
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
fn calc_test_error_as_leaf(target: &Series,
                           dist: &[f64],
                           indices: &[usize],
                           pred: i64)
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
    root.borrow_mut().pre_process();

    let mut front = frontier(root);
    // 1. Get the weakest link node.
    while let Some(node) = front.pop() {
        let node_err = node.borrow().train_node_misclassification_cost();
        let tree_err = node.borrow().test_tree_misclassification_cost();

        // 2. If the test error increase, then break.
        // The explession `node_err >= tree_err` implies
        // the test test error become increasing.
        if node_err > tree_err {
            break;
        }

        // 3. Otherwise, prune the node.
        let parent = node.borrow_mut().prune();


        // If the pruned node have a parent node, push it to `front`.
        if let Some(p) = parent {
            front.push(p);
        }


        // Sort them in increasing order.
        front.sort_by(|x, y| 
            y.borrow().alpha().partial_cmp(&x.borrow().alpha()).unwrap()
        );
    }
}



#[inline]
fn frontier(root: &Rc<RefCell<TrainNode>>) -> Vec<Rc<RefCell<TrainNode>>> {
    let mut front: Vec<Rc<RefCell<TrainNode>>> = Vec::new();
    let mut stack = vec![Rc::clone(root)];

    while let Some(node) = stack.pop() {
        let mut is_frontier = false;
        if let TrainNode::Branch(branch) = node.borrow().deref() {
            if !branch.left.borrow().is_leaf() {
                stack.push(Rc::clone(&branch.left));
            } else if !branch.right.borrow().is_leaf() {
                stack.push(Rc::clone(&branch.right));
            } else {
                is_frontier = true;
            }
        }

        if is_frontier {
            front.push(node);
        }
    }
    front.sort_by(|x, y| 
        y.borrow().alpha().partial_cmp(&x.borrow().alpha()).unwrap()
    );

    front
}



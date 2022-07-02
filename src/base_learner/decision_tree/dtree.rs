use polars::prelude::*;

use rand::prelude::*;
use rand::rngs::StdRng;

use crate::BaseLearner;


use super::dtree_classifier::DTreeClassifier;
use super::split_rule::*;
use super::node::*;
use super::train_node::*;


use std::rc::Rc;
use std::cell::RefCell;
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
}


impl DTree {
    /// Initialize `DTree`.
    #[inline]
    pub fn init(_df: &DataFrame) -> Self {
        let seed: u64 = 0;
        let rng = RefCell::new(SeedableRng::seed_from_u64(seed));
        let criterion = Criterion::Entropy;
        let split_rule = Split::Feature;
        let train_ratio = 0.8_f64;
        Self {
            rng,
            criterion,
            split_rule,
            train_ratio,
        }
    }


    /// Initialize the RNG by `seed`.
    /// If you don't use this method, 
    /// `DTree` initializes RNG by `0_u64`.
    pub fn seed(&self, seed: u64) {
        let rng: StdRng = SeedableRng::seed_from_u64(seed);
        *self.rng.borrow_mut() = rng;
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
                    self.criterion
                ),
        };


        prune(&mut tree);


        let root = match Rc::try_unwrap(tree) {
            Ok(train_node) => Node::from(train_node.into_inner()),
            Err(_) => panic!("Root node has reference counter >= 1")
        };

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
                  criterion: Criterion)
    -> Rc<RefCell<TrainNode>>
{
    // Compute the best prediction that minimizes the training error
    // on this node.
    let (pred, train_err) = calc_train_err(target, dist, &train[..]);


    let test_err = calc_test_err(target, dist, &test[..], pred);


    let node_err = NodeError::from((train_err, test_err));


    let (_, feature, threshold) = data.get_columns()
        .into_iter()
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


    let left = stump_fulltree(data, target, dist, ltrain, ltest, criterion);
    let right = stump_fulltree(data, target, dist, rtrain, rtest, criterion);


    Rc::new(RefCell::new(TrainNode::branch(
        rule, left, right, pred, node_err
    )))
}


/// Returns the best split
/// that maximizes the decrease of impurity.
#[inline]
fn find_best_split(data: &Series,
                   label: &Series,
                   dist: &[f64],
                   indices: &[usize])
    -> (f64, Impurity)
{
    let label = label.i64()
        .expect("The target class is not an dtype i64");


    let data = data.f64().unwrap();


    let mut indices = indices.to_vec();
    indices.sort_by(|&i, &j| {
        let xi = data.get(i).unwrap();
        let xj = data.get(j).unwrap();
        xi.partial_cmp(&xj).unwrap()
    });


    let total_weight = indices.iter()
        .fold(0.0, |acc, &i| acc + dist[i]);
    let mut left = TempNodeInfo::empty();
    let mut right = TempNodeInfo::new(label, dist, &indices[..]);


    // These variables are used for the best splitting rules.
    let mut best_decrease = right.entropic_impurity();
    let mut best_split = f64::MIN;


    let mut iter = indices.into_iter().peekable();

    while let Some(i) = iter.next() {
        let old_val = data.get(i).unwrap();
        let y = label.get(i).unwrap();
        let weight = dist[i];
        left.insert(y, weight);
        right.delete(y, weight);


        let mut new_val = f64::MAX;
        while let Some(&j) = iter.peek() {
            new_val = data.get(j).unwrap();
            if new_val != old_val { break; }

            let yj = label.get(j).unwrap();
            let wj = dist[j];
            left.insert(yj, wj);
            right.delete(yj, wj);

            iter.next();
        }


        let threshold = old_val * 0.5 + new_val * 0.5;


        let lp = left.total / total_weight;
        let rp = 1.0 - lp;


        let decrease = Impurity::from(lp) * left.entropic_impurity()
            + Impurity::from(rp) * right.entropic_impurity();


        if decrease < best_decrease {
            best_decrease = decrease;
            best_split = threshold;
        }
    }


    (best_split, best_decrease)
}


/// Some informations that are useful in `produce(..)`.
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
    pub(self) fn new(label: &ChunkedArray<Int64Type>,
                     dist: &[f64],
                     indices: &[usize])
        -> Self
    {
        let map = indices.iter()
            .fold(HashMap::new(), |mut mp, &i| {
                let l = label.get(i).unwrap();
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
    pub(self) fn insert(&mut self, label: i64, weight: f64) {
        let cnt = self.map.entry(label).or_insert(0.0);
        *cnt += weight;
        self.total += weight;

    }


    /// Decrease the number of positive examples by one.
    pub(self) fn delete(&mut self, label: i64, weight: f64) {
        match self.map.get_mut(&label) {
            Some(key) => { *key -= weight; },
            None => { return; }
        }
        if *self.map.get(&label).unwrap() <= 0.0 {
            self.map.remove(&label);
        }


        self.total -= weight;
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
        .expect("The target class df[{s}] is not an dtype i64");
    let mut counter = HashMap::new();
    for &i in indices {
        let l = target.get(i).unwrap();
        let cnt = counter.entry(l).or_insert(0.0);
        *cnt += dist[i];
    }


    let total = counter.values().sum::<f64>();


    // Compute the max (key, val) that has maximal p(j, t)
    let (l, p) = counter.into_iter()
        // .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .max_by(|a, b| {
            match a.1.partial_cmp(&b.1) {
                Some(res) => res,
                None => {
                    panic!("a.1 is: {}, b.1 is: {}", a.1, b.1);
                }
            }
        })
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
        .expect("The target class df[{s}] is not an dtype i64");
    let total = indices.iter().map(|&i| dist[i]).sum::<f64>();


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


use std::cell::Ref;
use std::ops::Deref;

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



use rand::prelude::*;
use rand::rngs::StdRng;

use crate::{DataBounds, Data, Sample};
use crate::BaseLearner;


use super::dtree_classifier::DTreeClassifier;
use super::split_rule::*;
use super::node::*;
use super::train_node::*;


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
pub struct DTree<L> {
    rng: RefCell<StdRng>,
    criterion: Criterion,
    split_rule: Split,
    train_ratio: f64,
    _phantom: PhantomData<L>,
}


impl<L> DTree<L> {
    /// Initialize `DTree`.
    #[inline]
    pub fn init<D>(_sample: &Sample<D, L>) -> Self
    {
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
            _phantom: PhantomData
        }
    }


    /// Initialize the RNG by `seed`.
    /// If you don't use this method, 
    /// `DTree` initializes RNG by `0_u64`.
    pub fn seed(&self, seed: u64)
    {
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


impl<O, D, L> BaseLearner<D, L> for DTree<L>
    where D: Data<Output = O>,
          L: PartialEq + Eq + std::hash::Hash + Clone,
          O: PartialOrd + Clone + DataBounds
{
    type Clf = DTreeClassifier<O, L>;
    fn produce(&self,
               sample: &Sample<D, L>,
               distribution: &[f64])
        -> Self::Clf
    {
        let m = sample.len();


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
                    sample,
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
fn stump_fulltree<O, D, L>(sample:     &Sample<D, L>,
                           dist:       &[f64],
                           train:       Vec<usize>,
                           test:        Vec<usize>,
                           criterion:   Criterion)
    -> Rc<RefCell<TrainNode<O, L>>>
    where D: Data<Output = O>,
          L: PartialEq + Eq + std::hash::Hash + Clone,
          O: PartialOrd + Clone + DataBounds,
{
    let (label, train_node_err) = calc_train_err(sample, dist, &train[..]);


    let test_node_err = calc_test_err(sample, dist, &test[..], &label);


    let node_err = NodeError::from((train_node_err, test_node_err));


    // Compute the node impurity
    let (mut best_split, mut best_decrease) = find_best_split(
        sample, &dist[..], &train[..], 0
    );


    let mut best_index = 0_usize;


    let dim = sample.dim();
    for j in 1..dim {

        let (split, decrease) = find_best_split(
            sample, &dist[..], &train[..], j
        );


        if decrease <= best_decrease {
            best_index    = j;
            best_split    = split;
            best_decrease = decrease;
        }
    }


    let rule = SplitRule::Stump(StumpSplit::from((best_index, best_split)));


    let (ltrain, rtrain) = train.into_iter()
        .fold((Vec::new(), Vec::new()), |(mut l, mut r), i| {
            let (x, _) = &sample[i];
            match rule.split(x) {
                LR::Left  => { l.push(i); },
                LR::Right => { r.push(i); }
            }
            (l, r)
        });
    let (ltest, rtest) = test.into_iter()
        .fold((Vec::new(), Vec::new()), |(mut l, mut r), i| {
            let (x, _) = &sample[i];
            match rule.split(x) {
                LR::Left  => { l.push(i); },
                LR::Right => { r.push(i); }
            }
            (l, r)
        });


    // If the split has no meaning, construct a leaf node.
    if ltrain.is_empty() || rtrain.is_empty() {
        let leaf = TrainNode::leaf(label, node_err);
        return Rc::new(RefCell::new(leaf));
    }


    let left = stump_fulltree(sample, dist, ltrain, ltest, criterion);
    let right = stump_fulltree(sample, dist, rtrain, rtest, criterion);


    Rc::new(RefCell::new(TrainNode::branch(
        rule, left, right, label, node_err
    )))
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


/// Some informations that are useful in `produce(..)`.
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



/// Returns the training error
#[inline]
fn calc_train_err<D, L>(sample:  &Sample<D, L>,
                        dist:    &[f64],
                        indices: &[usize])
    -> (L, f64)
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
fn calc_test_err<D, L>(sample:  &Sample<D, L>,
                       dist:    &[f64],
                       indices: &[usize],
                       pred:    &L)
    -> f64
    where L: Eq
{
    let total = indices.iter().map(|&i| dist[i]).sum::<f64>();


    if total == 0.0 {
        return 0.0;
    }


    let wrong = indices.iter()
        .filter_map(|&i| {
            let (_, l) = &sample[i];
            if l != pred { Some(dist[i]) } else { None }
        })
        .sum::<f64>();


    wrong / total
}


/// Prune the full-binary tree.
#[inline]
fn prune<O, L>(root: &mut Rc<RefCell<TrainNode<O, L>>>)
    where L: Clone
{
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
fn weak_links<O, L>(root: &Rc<RefCell<TrainNode<O, L>>>)
    -> Vec<Rc<RefCell<TrainNode<O, L>>>>
    where L: Clone
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

struct TrainNodeGuard<'a, O, L> {
    guard: Ref<'a, TrainNode<O, L>>
}


impl<'b, O, L> Deref for TrainNodeGuard<'b, O, L> {
    type Target = TrainNode<O, L>;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}


pub(self) fn get_ref<O, L>(node: &Rc<RefCell<TrainNode<O, L>>>)
    -> TrainNodeGuard<O, L>
{
    TrainNodeGuard {
        guard: node.borrow()
    }
}

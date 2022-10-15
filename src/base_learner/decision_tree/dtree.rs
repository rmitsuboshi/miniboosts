use polars::prelude::*;
use rayon::prelude::*;


use crate::BaseLearner;


use super::node::*;
use super::criterion::*;
use super::split_rule::*;
use super::train_node::*;
use super::dtree_classifier::DTreeClassifier;


use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;



/// Generates a `DTreeClassifier` for a given distribution
/// over examples.
pub struct DTree {
    criterion: Criterion,
    max_depth: Option<usize>,
    size: usize,
}


impl DTree {
    /// Initialize `DTree`.
    #[inline]
    pub fn init(df: &DataFrame) -> Self {
        let criterion = Criterion::Entropy;
        let max_depth = None;
        let size = df.shape().0;

        Self {
            criterion,
            max_depth,
            size,
        }
    }


    /// Specify the maximal depth of the tree.
    /// Default maximul depth is `None`.
    pub fn max_depth(mut self, depth: usize) -> Self {
        assert!(depth > 0);
        self.max_depth = Some(depth);

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
    fn produce(&self, data: &DataFrame, target: &Series, dist: &[f64])
        -> Self::Clf
    {
        let mut indices = (0..self.size).into_iter()
            // .filter(|&i| dist[i] > 0.0)
            .collect::<Vec<usize>>();

        indices.sort_by(|&i, &j| dist[i].partial_cmp(&dist[j]).unwrap());

        let criterion = self.criterion;
        let depth = self.max_depth;

        // Construct a large binary tree
        let tree = full_tree(data, target, dist, indices, criterion, depth);


        tree.borrow_mut().remove_redundant_nodes();


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
             indices: Vec<usize>,
             criterion: Criterion,
             max_depth: Option<usize>)
    -> Rc<RefCell<TrainNode>>
{

    let total_weight = indices.par_iter()
        .copied()
        .map(|i| dist[i])
        .sum::<f64>();


    // Compute the best prediction that minimizes the training error
    // on this node.
    let (pred, loss) = calc_loss_as_leaf(target, dist, &indices[..]);


    // If sum of `dist` over `train` is zero, construct a leaf node.
    if loss == 0.0 {
        return TrainNode::leaf(pred, total_weight, loss);
    }


    // Find the best splitting rule.
    let (feature, threshold) = match criterion {
        // FInd a split that minimizes the entropic impurity
        Criterion::Entropy => {
            let (_, feature, threshold) = data.get_columns()
                .into_par_iter()
                .map(|column| {
                    let (thr, dec) = find_best_split_entropy(
                        column, target, dist, &indices[..]
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
                        column, target, dist, &indices[..]
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
    let mut lindices = Vec::new();
    let mut rindices = Vec::new();
    for i in indices.into_iter() {
        match rule.split(data, i) {
            LR::Left  => { lindices.push(i); },
            LR::Right => { rindices.push(i); },
        }
    }


    // If the split has no meaning, construct a leaf node.
    if lindices.is_empty() || rindices.is_empty() {
        return TrainNode::leaf(pred, total_weight, loss);
    }


    // grow the tree.
    let ltree; // Left child
    let rtree; // Right child
    match max_depth {
        Some(depth) => {
            if depth == 1 {
                // If `depth == 1`,
                // the childs from this node must be leaves.
                ltree = construct_leaf(target, dist, lindices);
                rtree = construct_leaf(target, dist, rindices);
            } else {
                // If `depth > 1`,
                // the childs from this node might be branches.
                let d = Some(depth - 1);
                ltree = full_tree(data, target, dist, lindices, criterion, d);
                rtree = full_tree(data, target, dist, rindices, criterion, d);
            }
        },
        None => {
            ltree = full_tree(data, target, dist, lindices, criterion, None);
            rtree = full_tree(data, target, dist, rindices, criterion, None);
        }
    }


    TrainNode::branch(rule, ltree, rtree, pred, total_weight, loss)
}


#[inline]
fn construct_leaf(target: &Series,
                  dist: &[f64],
                  indices: Vec<usize>)
    -> Rc<RefCell<TrainNode>>
{
    // Compute the best prediction that minimizes the training error
    // on this node.
    let (pred, loss) = calc_loss_as_leaf(target, dist, &indices[..]);


    let total_weight = indices.iter()
            .copied()
            .map(|i| dist[i])
            .sum::<f64>();


    TrainNode::leaf(pred, total_weight, loss)
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
    triplets.sort_by(|(x1, _, _), (x2, _, _)| x1.partial_cmp(x2).unwrap());


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
    triplets.sort_by(|(x1, _, _), (x2, _, _)| x1.partial_cmp(x2).unwrap());


    let mut best_edge;
    let mut best_threshold;
    // Compute the edge of the hypothesis that predicts `+1`
    // for all instances.
    let mut edge = triplets.iter()
        .map(|(_, d, y)| *d * *y)
        .sum::<f64>();


    let mut iter = triplets.into_iter().peekable();


    // best threshold is the smallest value.
    // we define the initial threshold as the smallest value minus 1.0
    best_threshold = iter.peek()
        .map(|(v, _, _)| *v - 1.0_f64)
        .unwrap_or(f64::MIN);

    // best edge
    best_edge = edge.abs();

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
        let mut map: HashMap<i64, f64> = HashMap::new();
        triplets.iter()
            .for_each(|(_, d, y)| {
                total += *d;
                let cnt = map.entry(*y).or_insert(0.0);
                *cnt += *d;
            });

        Self { map, total }
    }


    /// Returns the impurity of this node.
    #[inline(always)]
    pub(self) fn entropic_impurity(&self) -> Impurity {
        if self.total == 0.0 || self.map.is_empty() {
            return 0.0.into();
        }

        self.map.par_iter()
            .map(|(_, &p)| {
                let r = p / self.total;
                if r == 0.0 { 0.0 } else { -r * r.ln() }
            })
            .sum::<f64>()
            .into()
    }


    /// Increase the number of positive examples by one.
    pub(self) fn insert(&mut self, y: i64, weight: f64) {
        let cnt = self.map.entry(y).or_insert(0.0);
        *cnt += weight;
        self.total += weight;
    }


    /// Decrease the number of positive examples by one.
    pub(self) fn delete(&mut self, y: i64, weight: f64) {
        if let Some(key) = self.map.get_mut(&y) {
            *key -= weight;
            self.total -= weight;
        }
    }
}



/// This function returns a tuple `(y, e)` where
/// - `y` is the prediction label that minimizes the training loss.
/// - `e` is the training loss when the prediction is `y`.
#[inline]
fn calc_loss_as_leaf(target: &Series, dist: &[f64], indices: &[usize])
    -> (i64, f64)
{
    let target = target.i64()
        .expect("The target class is not a dtype i64");
    let mut counter: HashMap<i64, f64> = HashMap::new();

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



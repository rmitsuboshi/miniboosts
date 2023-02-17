use polars::prelude::*;
use rayon::prelude::*;


use crate::WeakLearner;


use crate::weak_learner::common::{
    type_and_struct::*,
    split_rule::*,
};
use super::{
    node::*,
    criterion::*,
    train_node::*,
    dtree_classifier::DTreeClassifier,
};


use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;


/// `DTree` is the factory that
/// generates a `DTreeClassifier` for a given distribution over examples.
/// 
/// See also:
/// - [`DTree::max_depth`](DTree::max_depth)
/// - [`DTree::criterion`](DTree::criterion)
/// - [`Criterion`](Criterion)
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
/// // Get an instance of decision tree weak learner.
/// // In this example,
/// // the output hypothesis is at most depth 2.
/// // Further, this example uses `Criterion::Edge` for splitting rule.
/// let weak_learner = DTree::init(&data, &target)
///     .max_depth(2)
///     .criterion(Criterion::Edge);
/// ```
pub struct DTree {
    criterion: Criterion,
    max_depth: Depth,
}


impl DTree {
    /// Initialize [`DTree`](DTree).
    #[inline]
    pub fn init(data: &DataFrame, _target: &Series) -> Self {
        let criterion = Criterion::Entropy;
        let size = data.shape().0;
        let depth = ((size as f64).log2() + 1.0).ceil() as usize;

        Self {
            criterion,
            max_depth: Depth::from(depth),
        }
    }


    /// Specify the maximal depth of the tree.
    /// Default maximal depth is `log2` of number of training examples.
    pub fn max_depth(mut self, depth: usize) -> Self {
        assert!(depth > 0);
        self.max_depth = Depth::from(depth);

        self
    }


    /// Set criterion for node splitting.
    /// Default value is `Criterion::Entropy`.
    /// See [`Criterion`](Criterion).
    #[inline]
    pub fn criterion(mut self, criterion: Criterion) -> Self {
        self.criterion = criterion;
        self
    }
}


impl WeakLearner for DTree {
    type Hypothesis = DTreeClassifier;
    /// This method computes as follows;
    /// 1. construct a `TrainNode` which contains some information
    ///     to grow a tree (e.g., impurity, total distribution mass, etc.)
    /// 2. Convert `TrainNode` to `Node` that pares redundant information
    #[inline]
    fn produce(&self, data: &DataFrame, target: &Series, dist: &[f64])
        -> Self::Hypothesis
    {
        let n_sample = data.shape().0;

        let mut indices = (0..n_sample).into_iter()
            .filter(|&i| dist[i] > 0.0)
            .collect::<Vec<usize>>();

        indices.sort_by(|&i, &j| dist[i].partial_cmp(&dist[j]).unwrap());

        let criterion = self.criterion;

        // Construct a large binary tree
        let tree = full_tree(
            data, target, dist, indices, criterion, self.max_depth
        );


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
fn full_tree(
    data: &DataFrame,
    target: &Series,
    dist: &[f64],
    indices: Vec<usize>,
    criterion: Criterion,
    depth: Depth,
) -> Rc<RefCell<TrainNode>>
{

    let total_weight = indices.par_iter()
        .copied()
        .map(|i| dist[i])
        .sum::<f64>();


    // Compute the best prediction that minimizes the training error
    // on this node.
    let (pred, loss) = prediction_and_loss(target, dist, &indices[..]);


    // If sum of `dist` over `train` is zero, construct a leaf node.
    if loss == 0.0 {
        return TrainNode::leaf(pred, total_weight, loss);
    }


    // Find the best pair of feature name and threshold
    // based on the `criterion`.
    let (feature, threshold) = criterion.best_split(
        data, target, dist, &indices[..]
    );


    // Construct the splitting rule
    // from the best feature and threshold.
    let rule = Splitter::new(feature, Threshold::from(threshold));


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


    // Grow the tree.
    let ltree; // Left child
    let rtree; // Right child

    if depth <= 1 {
        // If `depth == 1`,
        // the childs from this node must be leaves.
        ltree = construct_leaf(target, dist, lindices);
        rtree = construct_leaf(target, dist, rindices);
    } else {
        // If `depth > 1`,
        // the childs from this node might be branches.
        let depth = depth - 1;
        ltree = full_tree(data, target, dist, lindices, criterion, depth);
        rtree = full_tree(data, target, dist, rindices, criterion, depth);
    }


    TrainNode::branch(rule, ltree, rtree, pred, total_weight, loss)
}


#[inline]
fn construct_leaf(
    target: &Series,
    dist: &[f64],
    indices: Vec<usize>
) -> Rc<RefCell<TrainNode>>
{
    // Compute the best prediction that minimizes the training error
    // on this node.
    let (pred, loss) = prediction_and_loss(target, dist, &indices[..]);


    let total_weight = indices.iter()
            .copied()
            .map(|i| dist[i])
            .sum::<f64>();


    TrainNode::leaf(pred, total_weight, loss)
}



/// This function returns a tuple `(y, e)` where
/// - `y` is the prediction label that minimizes the training loss.
/// - `e` is the training loss when the prediction is `y`.
#[inline]
fn prediction_and_loss(target: &Series, dist: &[f64], indices: &[usize])
    -> (Prediction<i64>, LossValue)
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

    let prediction = Prediction::from(l);
    let loss = LossValue::from(node_err);
    (prediction, loss)
}



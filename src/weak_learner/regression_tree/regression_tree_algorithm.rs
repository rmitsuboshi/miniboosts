use crate::{Sample, WeakLearner};
use super::bin::*;


use crate::weak_learner::common::{
    split_rule::*,
};

use super::{
    node::*,
    train_node::*,
    loss::LossType,
    regression_tree_regressor::RegressionTreeRegressor,
};


use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;


/// `RegressionTree` is the factory that generates
/// a `RegressionTreeClassifier` for a given distribution over examples.
/// 
/// # Example
/// ```no_run
/// use miniboosts::prelude::*;
/// 
/// // Read the training data from the CSV file.
/// let file = "/path/to/data/file.csv";
/// let has_header = true;
/// let sample = Sample::from_csv(file, has_header)
///     .unwrap()
///     .set_target("class");
/// 
/// 
/// // Get an instance of decision tree weak learner.
/// // In this example,
/// // the output hypothesis is at most depth 2.
/// // Further, this example uses `Criterion::Edge` for splitting rule.
/// let tree = RegressionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .loss(LossType::L2)
///     .build();
/// 
/// let n_sample = sample.shape().0;
/// let dist = vec![1/n_sample as f64; n_sample];
/// let f = tree.produce(&sample, &dist);
/// 
/// let predictions = f.predict_all(&sample);
/// 
/// let loss = sample.target()
///     .into_iter()
///     .zip(predictions)
///     .map(|(ty, py)| (ty as f64 - py).powi(2))
///     .sum::<f64>()
///     / n_sample as f64;
/// println!("loss (train) is: {loss}");
/// ```
pub struct RegressionTree<'a> {
    bins: HashMap<&'a str, Bins>,
    // The maximal depth of the output trees
    max_depth: usize,

    // The number of training instances
    n_sample: usize,

    // Regularization parameter
    lambda_l2: f64,


    // LossType function
    loss_type: LossType,
}


impl<'a> RegressionTree<'a> {
    #[inline]
    pub(super) fn from_components(
        bins: HashMap<&'a str, Bins>,
        n_sample: usize,
        max_depth: usize,
        lambda_l2: f64,
        loss_type: LossType,
    ) -> Self
    {
        Self { bins, n_sample, max_depth, lambda_l2, loss_type, }
    }


    #[inline]
    fn full_tree(
        &self,
        sample: &Sample,
        gh: &[GradientHessian],
        indices: Vec<usize>,
        max_depth: usize,
    ) -> Rc<RefCell<TrainNode>>
    {
        // Compute the best prediction that minimizes the training error
        // on this node.
        let (pred, loss) = self.loss_type.prediction_and_loss(
            &indices, gh, self.lambda_l2,
        );


        // If sum of `dist` over `train` is zero, construct a leaf node.
        if loss == 0.0 || max_depth <= 1 {
            return TrainNode::leaf(pred, loss);
        }


        // Find the best splitting rule.
        let (feature, threshold) = self.loss_type.best_split(
            &self.bins, &sample, gh, &indices[..], self.lambda_l2,
        );

        let rule = Splitter::new(feature, threshold);


        // Split the train data for left/right childrens
        let mut lindices = Vec::new();
        let mut rindices = Vec::new();
        for i in indices.into_iter() {
            match rule.split(sample, i) {
                LR::Left  => { lindices.push(i); },
                LR::Right => { rindices.push(i); },
            }
        }


        // If the split has no meaning, construct a leaf node.
        if lindices.is_empty() || rindices.is_empty() {
            return TrainNode::leaf(pred, loss);
        }

        // -----
        // At this point, `max_depth > 1` is guaranteed
        // so that one can grow the tree.
        let ltree = self.full_tree(sample, gh, lindices, max_depth-1);
        let rtree = self.full_tree(sample, gh, rindices, max_depth-1);


        TrainNode::branch(rule, ltree, rtree, pred, loss)
    }
}


impl<'a> WeakLearner for RegressionTree<'a> {
    type Hypothesis = RegressionTreeRegressor;


    fn name(&self) -> &str {
        "Regression Tree"
    }


    fn info(&self) -> Option<Vec<(&str, String)>> {
        let n_bins = self.bins.values()
            .map(|bin| bin.len())
            .reduce(usize::max)
            .unwrap_or(0);
        let info = Vec::from([
            ("# of bins (max)", format!("{n_bins}")),
            ("Max depth", format!("{}", self.max_depth)),
            ("Split criterion", format!("{}", self.loss_type)),
            ("Regularization param.", format!("{}", self.lambda_l2)),
        ]);
        Some(info)
    }


    fn produce(&self, sample: &Sample, predictions: &[f64])
        -> Self::Hypothesis
    {
        let gh = self.loss_type.gradient_and_hessian(
            sample.target(),
            predictions,
        );


        let indices = (0..self.n_sample).filter(|&i| {
                gh[i].grad != 0.0 || gh[i].hess != 0.0
            })
            .collect::<Vec<usize>>();


        let tree = self.full_tree(sample, &gh, indices, self.max_depth);


        let root = Node::from(
            Rc::try_unwrap(tree)
                .expect("Root node has reference counter >= 1")
                .into_inner()
        );

        RegressionTreeRegressor::from(root)
    }
}



impl fmt::Display for RegressionTree<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\
            ----------\n\
            # Decision Tree Weak Learner\n\n\
            - Max depth: {}\n\
            - Loss function: {}\n\
            - Bins:\
            ",
            self.max_depth,
            self.loss_type,
        )?;


        let width = self.bins.keys()
            .map(|key| key.len())
            .max()
            .expect("Tried to print bins, but no features are found");
        let max_bin_width = self.bins.values()
            .map(|bin| bin.len().ilog10() as usize)
            .max()
            .expect("Tried to print bins, but no features are found")
            + 1;
        for (feat_name, feat_bins) in self.bins.iter() {
            let n_bins = feat_bins.len();
            writeln!(
                f,
                "\
                \t* [{feat_name: <width$} | \
                {n_bins: >max_bin_width$} bins]  \
                {feat_bins}\
                "
            )?;
        }

        write!(f, "----------")
    }
}

use rayon::prelude::*;

use miniboosts_core::{
    tree::*,
    binning::*,
    Sample,
    WeakLearner,
    Regressor,
};

use decision_tree::{
    split_by::SplitBy,
    node::*,
};
use crate::regressor::RegressionTreeRegressor;
use crate::loss::RegressionTreeLoss;

use std::fmt;
use std::collections::HashMap;

type Gradient = f64;
type Hessian  = f64;

/// `RegressionTree` is the factory that generates
/// a `RegressionTreeClassifier` for a given distribution over examples.
/// 
/// # Example
/// ```no_run
/// use miniboosts::prelude::*;
/// 
/// // Read the training data from the CSV file.
/// let sample = SampleReader::new()
///     .file(file)
///     .has_header(true)
///     .target_feature("class")
///     .read()
///     .unwrap();
/// 
/// // Get an instance of decision tree weak learner.
/// // In this example,
/// // the output hypothesis is at most depth 2.
/// // Further, this example uses `Criterion::Edge` for splitting rule.
/// let tree = RegressionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .loss(GBMLoss::L2)
///     .build();
/// 
/// let n_sample = sample.shape().0;
/// let dist = vec![1f64 / n_sample as f64; n_sample];
/// let f = tree.produce(&sample, &dist);
/// 
/// let predictions = f.predict_all(&sample);
/// 
/// let loss = sample.target()
///     .into_iter()
///     .zip(predictions)
///     .map(|(ty, py)| (*ty as f64 - py).powi(2))
///     .sum::<f64>()
///     / n_sample as f64;
/// println!("loss (train) is: {loss}");
/// ```
pub struct RegressionTree<'a, L> {
    bins: HashMap<&'a str, Bins>,
    // The maximal depth of the output trees
    max_depth: usize,

    // The number of training instances
    n_sample: usize,

    // Regularization parameter
    lambda_l2: f64,

    // Loss function
    loss_func: L,
}

impl<'a, L> RegressionTree<'a, L> {
    #[inline]
    pub(super) fn from_components(
        bins: HashMap<&'a str, Bins>,
        n_sample:  usize,
        max_depth: usize,
        lambda_l2: f64,
        loss_func: L,
    ) -> Self
    {
        Self { bins, n_sample, max_depth, lambda_l2, loss_func, }
    }

    #[inline]
    fn grow(
        &self,
        sample: &Sample,
        gradient: &[Gradient],
        hessian: &[Hessian],
        indices: Vec<usize>,
        max_depth: usize,
    ) -> Box<Node>
    {
        // Compute the best prediction that minimizes the training error
        // on this node.
        let (pred, loss) = prediction_and_loss(
            &indices[..], gradient, hessian, self.lambda_l2,
        );

        // If sum of `dist` over `train` is zero, construct a leaf node.
        if loss == 0f64 || max_depth < 1 {
            return Box::new(Node::leaf(pred));
        }

        // Find the best splitting rule.
        let (feature, threshold) = best_split(
            &self.bins,
            sample,
            gradient,
            hessian,
            &indices[..],
            self.lambda_l2,
        );

        let rule = Splitter::new(feature, threshold);

        // Split the train data for left/right childrens
        let mut lindices = Vec::new();
        let mut rindices = Vec::new();
        for i in indices.into_iter() {
            match rule.split(sample, i) {
                LeftRight::Left  => { lindices.push(i); },
                LeftRight::Right => { rindices.push(i); },
            }
        }

        // If the split has no meaning, construct a leaf node.
        if lindices.is_empty() || rindices.is_empty() {
            return Box::new(Node::leaf(pred));
        }

        // -----
        // At this point, `max_depth > 1` is guaranteed
        // so that one can grow the tree.
        let ltree = self.grow(sample, gradient, hessian, lindices, max_depth-1);
        let rtree = self.grow(sample, gradient, hessian, rindices, max_depth-1);

        Box::new(Node::branch(rule, ltree, rtree, pred))
    }
}

impl<L> WeakLearner for RegressionTree<'_, L>
    where L: RegressionTreeLoss,
{
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
            ("Split criterion", self.loss_func.name().to_string()),
            ("Regularization param.", format!("{}", self.lambda_l2)),
        ]);
        Some(info)
    }

    fn produce(&self, sample: &Sample, predictions: &[f64])
        -> Self::Hypothesis
    {
        let gradient = self.loss_func.gradient(predictions, sample.target());
        let hessian = self.loss_func.diag_hessian(predictions, sample.target());

        let indices = (0..self.n_sample).collect::<Vec<_>>();

        let root = self.grow(
            sample,
            &gradient[..],
            &hessian[..],
            indices,
            self.max_depth,
        );

        RegressionTreeRegressor::from(root)
    }
}

impl<L> fmt::Display for RegressionTree<'_, L>
    where L: RegressionTreeLoss,
{
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
            self.loss_func.name(),
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

/// Returns the best splitting rule based on the loss function.
fn best_split<'a>(
    bins_map: &HashMap<&'_ str, Bins>,
    sample: &'a Sample,
    gradient: &[Gradient],
    hessian: &[Hessian],
    idx: &[usize],
    lambda_l2: f64,
) -> (&'a str, f64)
{

    sample.features()
        .par_iter()
        .map(|feature| {
            let name = feature.name();
            let bin = bins_map.get(name).unwrap();
            let pack = bin.pack(idx, feature, gradient, hessian);
            let (score, threshold) = best_split_at(pack, lambda_l2);

            (score, name, threshold)
        })
        .max_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
        .map(|(_, name, threshold)| (name, threshold))
        .expect("No feature that maximizes the score.")
}

/// this code is implemented based on Algorithm 3 of the following paper:
/// Tianqi Chen and Carlos Guestrin.
/// XGBoost: A scalable tree boosting system [KDD '16]
fn best_split_at(
    pack: Vec<(Bin, Gradient, Hessian)>,
    lambda_l2: f64,
) -> (LossValue, f64)
{
    let mut right_grad_sum = pack.par_iter()
        .map(|(_, grad, _)| grad)
        .sum::<f64>();
    let mut right_hess_sum = pack.par_iter()
        .map(|(_, _, hess)| hess)
        .sum::<f64>();

    let mut left_grad_sum = 0.0;
    let mut left_hess_sum = 0.0;

    let mut best_score = f64::MIN;
    let mut best_threshold = f64::MIN;

    for (bin, grad, hess) in pack {
        left_grad_sum  += grad;
        left_hess_sum  += hess;
        right_grad_sum -= grad;
        right_hess_sum -= hess;

        let score = {
            let l = left_grad_sum.powi(2) / (left_hess_sum + lambda_l2);
            let r = right_grad_sum.powi(2) / (right_hess_sum + lambda_l2);
            l + r
        };
        if best_score < score {
            best_score = score;
            best_threshold = bin.0.end;
        }
    }

    (best_score.into(), best_threshold.into())
}

/// returns the prediction value and the loss value of a leaf.
/// this function is implemented based on Eqs. (5), (6) of the following paper:
/// Tianqi Chen and Carlos Guestrin.
/// XGBoost: A scalable tree boosting system [KDD '16]
fn prediction_and_loss(
    indices: &[usize],
    gradient: &[Gradient],
    hessian: &[Hessian],
    lambda_l2: f64,
) -> (f64, LossValue)
{
    let grad_sum = indices.par_iter()
        .map(|&i| gradient[i])
        .sum::<f64>();

    let hess_sum = indices.par_iter()
        .map(|&i| hessian[i])
        .sum::<f64>();

    let prediction = - grad_sum / (hess_sum + lambda_l2);
    let loss_value = -0.5 * grad_sum.powi(2) / (hess_sum + lambda_l2);

    (prediction, loss_value.into())
}


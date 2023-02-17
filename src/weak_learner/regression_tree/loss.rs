//! Defines some criterions for regression tree.
use polars::prelude::*;
use rayon::prelude::*;
use serde::{
    Serialize,
    Deserialize
};


use crate::weak_learner::common::type_and_struct::*;

/// The type of loss (error) function.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum LossType {
    /// Least Absolute Error
    L1,
    /// Least Squared Error
    L2,
}


impl LossType {
    /// Returns the best splitting rule based on the loss function.
    pub(super) fn best_split<'a>(
        &self,
        data: &'a DataFrame,
        target: &Series,
        dist: &[f64],
        idx: &[usize],
    ) -> (&'a str, Threshold)
    {
        data.get_columns()
            .into_par_iter()
            .map(|column| {
                let (loss_value, threshold) = self.best_split_at(
                    column, target, dist, &idx[..]
                );

                (loss_value, column.name(), threshold)
            })
            .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
            .map(|(_, name, threshold)| (name, threshold))
            .expect(
                "\
                No feature that minimizes \
                the weighted loss.\
                "
            )
    }


    fn best_split_at(
        &self,
        column: &Series,
        target: &Series,
        dist: &[f64],
        idx: &[usize],
    ) -> (LossValue, Threshold)
    {
        match self {
            LossType::L1 => best_split_l1(column, target, dist, idx),
            LossType::L2 => best_split_l2(column, target, dist, idx),
        }
    }


    pub(super) fn prediction_and_loss(
        &self,
        target: &Series,
        indices: &[usize],
        dist: &[f64],
    ) -> (Prediction<f64>, LossValue)
    {
        let target = target.f64()
            .expect("The target class is not a dtype i64");


        let tuples = indices.into_par_iter()
            .copied()
            .map(|i| {
                let y = target.get(i).unwrap();
                (dist[i], y)
            })
            .collect::<Vec<(f64, f64)>>();

        let sum_dist = tuples.iter()
            .map(|(d, _)| d)
            .sum::<f64>();

        if sum_dist == 0.0 {
            return (Prediction::from(0.0), LossValue::from(0.0));
        }


        let prediction;
        let loss_value;
        match self {
            LossType::L1 => {
                // For `L1`-loss, we use `inner_prediction_and_loss`
                // to get the prediction and its loss.
                // Function `inner_prediction_and_loss` takes the slice
                // `&[(Feature, Mass, Target)]`
                // but does not use the first argument.
                // So I adopt the dummy feature value `0.0`.
                let triplets = tuples.into_iter()
                    .map(|(d, y)| (0.0.into(), d.into(), y.into()))
                    .collect::<Vec<(Feature, Mass, Target)>>();
                // // Sort the values of `tuples`
                // // in the ascending order of `y`.
                // tuples.sort_by(|(_, y1), (_, y2)|
                //     y1.partial_cmp(&y2).unwrap()
                // );

                // let mut d_sum = 0.0;

                // for (d, y) in tuples.iter() {
                //     d_sum += d;
                //     if d_sum >= 0.5 * sum_dist {
                //         prediction = *y;
                //         break;
                //     }
                // }


                // assert_ne!(prediction, 1e9);

                // loss_value = tuples.into_iter()
                //     .map(|(d, y)| d * (y - prediction).abs())
                //     .sum::<f64>()
                //     / sum_dist;
                (prediction, loss_value) = inner_prediction_and_loss(
                    &triplets[..],
                );
            },
            LossType::L2 => {
                prediction = Prediction::from(
                    tuples.iter()
                        .map(|(d, y)| *d * *y)
                        .sum::<f64>()
                        / sum_dist
                );


                loss_value = LossValue::from(
                    tuples.into_iter()
                        .map(|(d, y)| d * (y - prediction.0).powi(2))
                        .sum::<f64>()
                        / sum_dist
                );
            }
        }

        (prediction, loss_value)
    }

}


/// Returns the pair of the best LAE and the best threshold.
fn best_split_l1(
    column: &Series,
    target: &Series,
    dist: &[f64],
    idx: &[usize]
) -> (LossValue, Threshold)
{
    let target = target.f64()
        .expect("The target class is not a dtype i64");


    let column = column.f64()
        .expect("The column is not a dtype f64");


    // Sort the data in the increasing order of x.
    let mut triplets = idx.into_par_iter()
        .copied()
        .map(|i| {
            let x = column.get(i).unwrap();
            let y = target.get(i).unwrap();
            (Feature::from(x), Mass::from(dist[i]), Target::from(y))
        })
        .collect::<Vec<_>>();
    triplets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());


    let mut node = BestSplitFinderL1::new(triplets);
    let mut best_loss_value;
    let mut best_threshold;


    (best_loss_value, best_threshold) = node.lad_and_threshold();


    while let Some((loss_value, threshold)) = node.move_to_left() {
        if loss_value < best_loss_value {
            best_loss_value = loss_value;
            best_threshold = threshold;
        }
    }

    let loss = LossValue::from(best_loss_value);
    let threshold = Threshold::from(best_threshold);
    (loss, threshold)
}


/// Returns the pair of the best variance and the best threshold.
fn best_split_l2(
    column: &Series,
    target: &Series,
    dist: &[f64],
    idx: &[usize]
) -> (LossValue, Threshold)
{
    let target = target.f64()
        .expect("The target class is not a dtype i64");


    let column = column.f64()
        .expect("The column is not a dtype f64");


    // Sort the data in the increasing order of x.
    let mut triplets = idx.into_par_iter()
        .copied()
        .map(|i| {
            let x = column.get(i).unwrap();
            let y = target.get(i).unwrap();
            (x, dist[i], y)
        })
        .collect::<Vec<(f64, f64, f64)>>();
    triplets.sort_by(|(x1, _, _), (x2, _, _)| x1.partial_cmp(x2).unwrap());


    let mut node = BestSplitFinderL2::new(triplets);
    let mut best_variance;
    let mut best_threshold;


    (best_variance, best_threshold) = node.variance_and_threshold();


    while let Some((variance, threshold)) = node.move_to_left() {
        if variance < best_variance {
            best_variance = variance;
            best_threshold = threshold;
        }
    }

    let loss = LossValue::from(best_variance);
    let threshold = Threshold::from(best_threshold);
    (loss, threshold)
}


/// This struct stores the temporary information to find a best split.
struct BestSplitFinderL2 {
    /// The list of target values for the left/right nodes.
    left: Vec<(f64, f64, f64)>,
    right: Vec<(f64, f64, f64)>,

    /// mean of the target values reached to left/right nodes.
    left_sum_target: f64,
    right_sum_target: f64,


    /// Sum of the distribution for the instances reached to left/right nodes.
    left_sum_dist: f64,
    right_sum_dist: f64,
}


impl BestSplitFinderL2 {
    /// Create an instance of `BestSplitFinderL2` that contains
    /// all instances in `triplets`.
    /// Note that `triplets` is sorted in the ascending order
    /// with respect to the first element.
    pub(self) fn new(triplets: Vec<(f64, f64, f64)>) -> Self {
        let m = triplets.len();

        let left = Vec::with_capacity(m);
        let right = triplets;

        let left_sum_target: f64 = 0.0;
        let mut right_sum_target: f64 = 0.0;

        let left_sum_dist: f64 = 0.0;
        let mut right_sum_dist: f64 = 0.0;

        // `targets` reserves the target values in
        // the descending order of the first element.
        right.iter()
            .rev()
            .for_each(|(_, d, target)| {
                right_sum_target += d * target;
                right_sum_dist += d;
            });


        Self {
            left,
            right,

            left_sum_target,
            right_sum_target,

            left_sum_dist,
            right_sum_dist,
        }
    }


    /// Returns the pair of variance and threshold.
    pub(self) fn variance_and_threshold(&self) -> (f64, f64) {
        // Compute the threshold to make the current split.
        let threshold;
        if self.right.is_empty() {
            let left_largest = self.left.last()
                .map(|(x, _, _)| x).unwrap();
            threshold = left_largest + 1.0;
        } else {
            let right_smallest = self.right.last()
                .map(|(x, _, _)| x)
                .unwrap();
            let left_largest = self.left.last()
                .map(|(x, _, _)| *x)
                .unwrap_or(right_smallest - 2.0);
            threshold = (left_largest + right_smallest) / 2.0;
        }


        // Compute the variance.
        let left_variance = calc_variance(
            &self.left[..], self.left_sum_target, self.left_sum_dist,
        );

        let right_variance = calc_variance(
            &self.right[..], self.right_sum_target, self.right_sum_dist,
        );


        let left_prob = self.left_sum_dist
            / (self.left_sum_dist + self.right_sum_dist);
        let right_prob = 1.0_f64 - left_prob;


        let variance = left_prob * left_variance + right_prob * right_variance;

        (variance, threshold)
    }



    /// Move an instance of right node to the left node.
    /// After that, this method returns the pair of
    /// the variance of the new threshold.
    pub(self) fn move_to_left(&mut self) -> Option<(f64, f64)> {
        if let Some(r) = self.right.pop() {
            self.right_sum_dist -= r.1;
            self.right_sum_target -= r.1 * r.2;

            self.left_sum_dist += r.1;
            self.left_sum_target += r.1 * r.2;

            self.left.push(r);

            while let Some(r2) = self.right.pop() {
                if r.0 == r2.0 {
                    self.right_sum_dist -= r2.1;
                    self.right_sum_target -= r2.1 * r2.2;

                    self.left_sum_dist += r2.1;
                    self.left_sum_target += r2.1 * r2.2;

                    self.left.push(r2);
                } else {
                    self.right.push(r2);
                    break;
                }
            }

            let variance_threshold = self.variance_and_threshold();
            Some(variance_threshold)
        } else {
            None
        }
    }
}



fn calc_variance(triplets: &[(f64, f64, f64)], y_sum: f64, sum_dist: f64)
    -> f64
{
    if sum_dist == 0.0 || triplets.is_empty() {
        return 0.0;
    }


    let y_mean = y_sum / sum_dist;
    triplets.into_par_iter()
        .map(|(_, d, y)| d * (y - y_mean).powi(2))
        .sum::<f64>()
        / sum_dist
}


/// This struct stores the temporary information to find a best split.
struct BestSplitFinderL1 {
    /// The list of target values for the left/right nodes.
    left: Vec<(Feature, Mass, Target)>,
    right: Vec<(Feature, Mass, Target)>,
}


impl BestSplitFinderL1 {
    /// Create an instance of `BestSplitFinderL1` that contains
    /// all instances in `triplets`.
    /// Note that `triplets` is sorted in the ascending order
    /// with respect to the first element.
    pub(self) fn new(triplets: Vec<(Feature, Mass, Target)>) -> Self {
        let m = triplets.len();

        let left = Vec::with_capacity(m);
        let right = triplets;

        Self {
            left,
            right,
        }
    }


    /// Returns the pair of variance and threshold.
    pub(self) fn lad_and_threshold(&self)
        -> (LossValue, Threshold)
    {
        // Compute the threshold to make the current split.
        let threshold;
        if self.right.is_empty() {
            let left_largest = self.left.last()
                .map(|(x, _, _)| x.0)
                .unwrap();
            threshold = left_largest + 1.0;
        } else {
            let right_smallest = self.right.last()
                .map(|(x, _, _)| x.0)
                .unwrap();
            let left_largest = self.left.last()
                .map(|(x, _, _)| x.0)
                .unwrap_or(right_smallest - 2.0);
            threshold = (left_largest + right_smallest) / 2.0;
        }

        let threshold = Threshold::from(threshold);


        // Compute the variance.
        let left_lad = inner_prediction_and_loss(&self.left[..]).1;

        let right_lad = inner_prediction_and_loss(&self.right[..]).1;


        let lad = left_lad + right_lad;

        (lad, threshold)
    }



    /// Move an instance of right node to the left node.
    /// After that, this method returns the pair of
    /// the variance and the new threshold.
    pub(self) fn move_to_left(&mut self)
        -> Option<(LossValue, Threshold)>
    {
        if let Some(r) = self.right.pop() {
            self.left.push(r);

            let feature = r.0;

            while let Some(r2) = self.right.pop() {
                if feature == r2.0 {
                    self.left.push(r2);
                } else {
                    self.right.push(r2);
                    break;
                }
            }

            let lad_threshold = self.lad_and_threshold();
            Some(lad_threshold)
        } else {
            None
        }
    }
}


fn inner_prediction_and_loss(triplets: &[(Feature, Mass, Target)])
    -> (Prediction<f64>, LossValue)
{
    let m_total = triplets.iter()
        .map(|(_, m, _)| m.0)
        .sum::<f64>();

    if m_total == 0.0 {
        let prediction = 0.0.into();
        let loss = 0.0.into();
        return (prediction, loss);
    }

    let mut triplets = triplets.to_vec();
    triplets.sort_by(|(_, _, t1), (_, _, t2)|
        t1.partial_cmp(&t2).unwrap()
    );


    let mut p_small = 1e9;
    let mut p_large = 1e9;
    let mut m_partial_sum = 0.0;
    for (_, m, t) in triplets.iter() {
        m_partial_sum += m.0;
        if m_partial_sum >= 0.5 * m_total {
            p_large = t.0;
            if p_small == 1e9 {
                p_small = t.0;
            }
            break;
        }
        p_small = t.0;
    }

    assert!(p_small < 1e9);
    assert!(p_large < 1e9);

    let loss_p_small = triplets.par_iter()
        .map(|(_, m, t)| m.0 * (t.0 - p_small).abs())
        .sum::<f64>();
    let loss_p_large = triplets.par_iter()
        .map(|(_, m, t)| m.0 * (t.0 - p_large).abs())
        .sum::<f64>();

    let prediction;
    let loss;
    if loss_p_small < loss_p_large {
        prediction = p_small.into();
        loss = loss_p_small.into();
    } else {
        prediction = p_large.into();
        loss = loss_p_large.into();
    }
    (prediction, loss)
}



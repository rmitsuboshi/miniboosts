//! Defines some criterions for regression tree.
use polars::prelude::*;
use rayon::prelude::*;
use serde::{
    Serialize,
    Deserialize
};


/// The type of loss (error) function.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Loss {
    /// Least Absolute Error
    L1,
    /// Least Squared Error
    L2,
}


impl Loss {
    /// Returns the best splitting rule based on the loss function.
    pub(super) fn best_split<'a>(
        &self,
        data: &'a DataFrame,
        target: &Series,
        dist: &[f64],
        idx: &[usize],
    ) -> (&'a str, f64)
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
    ) -> (f64, f64)
    {
        match self {
            Loss::L1 => best_split_l1(column, target, dist, idx),
            Loss::L2 => best_split_l2(column, target, dist, idx),
        }
    }

}


/// Returns the pair of the best LAE and the best threshold.
fn best_split_l1(
    column: &Series,
    target: &Series,
    dist: &[f64],
    idx: &[usize]
) -> (f64, f64)
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


    let mut node = LocalNodeL1::new(triplets);
    let mut best_variance;
    let mut best_threshold;


    (best_variance, best_threshold) = node.lae_and_threshold();


    while let Some((variance, threshold)) = node.move_to_left() {
        if variance < best_variance {
            best_variance = variance;
            best_threshold = threshold;
        }
    }

    (best_variance, best_threshold)
}


/// Returns the pair of the best variance and the best threshold.
fn best_split_l2(
    column: &Series,
    target: &Series,
    dist: &[f64],
    idx: &[usize]
) -> (f64, f64)
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


    let mut node = LocalNodeL2::new(triplets);
    let mut best_variance;
    let mut best_threshold;


    (best_variance, best_threshold) = node.variance_and_threshold();


    while let Some((variance, threshold)) = node.move_to_left() {
        if variance < best_variance {
            best_variance = variance;
            best_threshold = threshold;
        }
    }

    (best_variance, best_threshold)
}


/// This struct stores the temporary information to find a best split.
struct LocalNodeL2 {
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


impl LocalNodeL2 {
    /// Create an instance of `LocalNodeL2` that contains
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
struct LocalNodeL1 {
    /// The list of target values for the left/right nodes.
    left: Vec<(f64, f64, f64)>,
    right: Vec<(f64, f64, f64)>,

    /// sum of the target values reached to left/right nodes.
    left_sum_target: f64,
    right_sum_target: f64,
}


impl LocalNodeL1 {
    /// Create an instance of `LocalNodeL1` that contains
    /// all instances in `triplets`.
    /// Note that `triplets` is sorted in the ascending order
    /// with respect to the first element.
    pub(self) fn new(triplets: Vec<(f64, f64, f64)>) -> Self {
        let m = triplets.len();

        let left = Vec::with_capacity(m);
        let right = triplets;

        let left_sum_target: f64 = 0.0;
        let right_sum_target = right.iter().map(|t| t.2).sum::<f64>();


        Self {
            left,
            right,

            left_sum_target,
            right_sum_target,
        }
    }


    /// Returns the pair of variance and threshold.
    pub(self) fn lae_and_threshold(&self) -> (f64, f64) {
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
        let left_lae = calc_lae(&self.left[..], self.left_sum_target);

        let right_lae = calc_lae(&self.right[..], self.right_sum_target);


        let lae = left_lae + right_lae;

        (lae, threshold)
    }



    /// Move an instance of right node to the left node.
    /// After that, this method returns the pair of
    /// the variance of the new threshold.
    pub(self) fn move_to_left(&mut self) -> Option<(f64, f64)> {
        if let Some(r) = self.right.pop() {
            self.right_sum_target -= r.2;
            self.left_sum_target  += r.2;

            self.left.push(r);

            while let Some(r2) = self.right.pop() {
                if r.0 == r2.0 {
                    self.right_sum_target -= r2.2;
                    self.left_sum_target  += r2.2;

                    self.left.push(r2);
                } else {
                    self.right.push(r2);
                    break;
                }
            }

            let lae_threshold = self.lae_and_threshold();
            Some(lae_threshold)
        } else {
            None
        }
    }
}


fn calc_lae(triplets: &[(f64, f64, f64)], y_sum: f64)
    -> f64
{
    let n_sample = triplets.len() as f64;
    let y_mean = y_sum / n_sample;
    triplets.into_par_iter()
        .map(|(_, w, y)| w * (y - y_mean).abs())
        .sum::<f64>()
}



//! This file defines some options of MLPBoost.

use crate::{
    Sample,
    Classifier,
    common::{utils, checker},
    common::frank_wolfe::FWType,
};

const SUB_TOLERANCE: f64 = 1e-9;

pub(crate) struct DeformedFrankWolfe {
    deform: f64, // deformation parameter
    eta: f64,    // regularization parameter
    nu: f64,
    fw_type: FWType,
}

impl DeformedFrankWolfe {
    /// Create a new DeformedFrankWolfe instance
    pub(crate) fn new(deform: f64, eta: f64, nu: f64, fw_type: FWType) -> Self {
        assert!((0f64..=1f64).contains(&deform));
        Self { deform, eta, nu, fw_type, }
    }

    /// Set `self.eta`.
    pub(crate) fn eta(&mut self, eta: f64) {
        self.eta = eta;
    }

    /// Set `self.deform`.
    pub(crate) fn deform(&mut self, deform: f64) {
        assert!((0f64..=1f64).contains(&deform));
        self.deform = deform;
    }

    /// Set `nu`.
    pub(crate) fn nu(&mut self, nu: f64) {
        self.nu = nu;
    }

    /// Get the current Frank-Wolfe strategy.
    pub(crate) fn current_type(&self) -> FWType {
        self.fw_type
    }

    /// Update the Frank-Wolfe type.
    /// Currently, you can choose types from below:
    /// - Classic. `1.0 / t + 1.0` for `t >= 1`.
    /// - ShortStep. The step size that minimizes the strongly-smooth bound.
    /// - LineSearch. Optimal step size that minimizes the objective.
    /// - Pairwise. Pairwise update.
    pub(crate) fn fw_type(&mut self, fw_type: FWType) {
        self.fw_type = fw_type;
    }

    /// Returns the next weights on hypotheses
    pub(crate) fn next_iterate<H>(
        &self,
        iteration: usize,
        sample: &Sample,
        dist: &[f64],
        hypotheses: &[H],
        position_of_new_one: usize,
        weights: Vec<f64>,
    ) -> Vec<f64>
        where H: Classifier,
    {
        let n_hypotheses = hypotheses.len();
        assert!((0..n_hypotheses).contains(&position_of_new_one));
        match self.fw_type {
            FWType::Classic
                => self.classic(iteration, position_of_new_one, weights),
            FWType::ShortStep
                => self.short_step(
                    sample, dist, hypotheses, position_of_new_one, weights,
                ),
            FWType::LineSearch
                => self.line_search(
                    sample, hypotheses, position_of_new_one, weights,
                ),
            FWType::BlendedPairwise
                => self.blended_pairwise(
                    sample, dist, hypotheses, position_of_new_one, weights,
                ),
        }
    }

    fn classic(
        &self,
        iteration: usize,
        position_of_new_one: usize,
        weights: Vec<f64>,
    ) -> Vec<f64>
    {
        // Compute the step-size
        let step_size = 2f64 / (iteration as f64 + 1f64);

        // Update the weights
        interior_point(step_size, position_of_new_one, weights)
    }

    fn short_step<H>(
        &self,
        sample: &Sample,
        dist: &[f64],
        hypotheses: &[H],
        position_of_new_one: usize,
        weights: Vec<f64>,
    ) -> Vec<f64>
        where H: Classifier,
    {
        if hypotheses.len() == 1 { return vec![1.0]; }

        let h = &hypotheses[position_of_new_one];

        let old_margins = utils::margins_of_weighted_hypothesis(
            sample, &weights[..], hypotheses,
        );
        let new_margins = utils::margins_of_hypothesis(sample, h);

        // Compute the step size `= numer / denom`, where
        // `numer = d^T A (e_h - w)` and
        // `denom = ‖ A (e_h - w) ‖_{(2-t)/(1-t)}`.
        let power = (2f64 - self.deform) / (1f64 - self.deform);
        let mut numer: f64 = 0f64;
        let mut denom: f64 = 0f64;
        new_margins.into_iter()
            .zip(old_margins)
            .zip(dist)
            .for_each(|((ae, aw), &d)| {
                let diff = ae - aw;
                numer += d * diff;
                denom += diff.abs().powf(power);
            });

        assert!(
            denom.is_finite() && denom > 0f64,
            "denominator of the short-step size \
            must be a positive finite value. \
            got = {denom}"
        );

        let step = numer * (2f64 - self.deform) / (self.eta * denom.powf(2f64 / power));

        // Clip the step size to `[0, 1]`.
        let step_size = step.clamp(0f64, 1f64);

        // Update the weights
        interior_point(step_size, position_of_new_one, weights)
    }

    fn line_search<H>(
        &self,
        sample: &Sample,
        hypotheses: &[H],
        position_of_new_one: usize,
        mut weights: Vec<f64>,
    ) -> Vec<f64>
        where H: Classifier,
    {
        // base: w
        // dir:  e_h - w
        let base = weights.clone();
        let dir  = weights.iter()
            .copied()
            .enumerate()
            .map(|(j, w)|
                if j == position_of_new_one { 1.0 - w } else { -w }
            )
            .collect::<Vec<_>>();

        // A(e_h - w)
        let dir_margins = utils::margins_of_weighted_hypothesis(
            sample, &dir[..], hypotheses,
        );
        // Aw
        let base_margins = utils::margins_of_weighted_hypothesis(
            sample, &base[..], hypotheses,
        );

        // Check the case where the step size is `1`.
        let dist = utils::deformed_exp_distribution(
            self.deform, self.eta, self.nu, sample, &dir[..], hypotheses,
        );

        // `dot` is `d * A (e_h - w)`
        let dot = utils::inner_product(&dist[..], &dir_margins[..]);

        if dot <= 0.0 {
            let n_weights = weights.len();
            weights = vec![0.0; n_weights];
            weights[position_of_new_one] = 1.0;
            return weights;
        }

        let mut ub = 1.0;
        let mut lb = 0.0;
        while ub - lb > SUB_TOLERANCE {
            let step_size = (lb + ub) / 2.0;

            let margins = base_margins.iter()
                .zip(&dir_margins[..])
                .map(|(&b, &d)| b + step_size * d);
            let dist = utils::deformed_exp_distribution_from_margins(
                self.deform, self.eta, self.nu, margins,
            );

            // Compute the gradient for the direction `dir`.
            let dot = utils::inner_product(&dist[..], &dir_margins[..]);

            if dot < 0.0 {
                lb = step_size;
            } else if dot > 0.0 {
                ub = step_size;
            } else {
                break;
            }
        }

        // Update the weights
        let step_size = (lb + ub) / 2.0;

        interior_point(step_size, position_of_new_one, base)
    }

    fn blended_pairwise<H>(
        &self,
        sample: &Sample,
        dist: &[f64],
        hypotheses: &[H],
        position_of_new_one: usize,
        mut weights: Vec<f64>,
    ) -> Vec<f64>
        where H: Classifier,
    {
        // Find a hypothesis that has a smallest edge.
        let mut worst_edge = 2.0;
        let mut local_best_edge = -2.0;
        let mut global_best_edge = -2.0;
        let mut position_of_worst_one = hypotheses.len();
        let mut position_of_local_best_one = hypotheses.len();
        let mut position_of_global_best_one = position_of_new_one;
        weights.iter()
            .zip(hypotheses)
            .enumerate()
            .filter_map(|(j, (w, h))| {
                if *w <= 0.0 {
                    None
                } else {
                    let edge = utils::edge_of_hypothesis(sample, dist, h);
                    Some((j, edge))
                }
            })
            .for_each(|(j, edge)| {
                if j != position_of_new_one && edge > local_best_edge {
                    local_best_edge = edge;
                    position_of_local_best_one = j;
                }

                if edge > global_best_edge {
                    global_best_edge = edge;
                    position_of_global_best_one = j;
                }

                if edge < worst_edge {
                    worst_edge = edge;
                    position_of_worst_one = j;
                }
            });

        // If the following condition holds,
        // the global FW atom is not the newly attained one
        // so that the local one is the same as the global one.
        if position_of_global_best_one + 1 != hypotheses.len() {
            local_best_edge = global_best_edge;
            position_of_local_best_one = position_of_global_best_one;
        }

        let current_edge = utils::edge_of_weighted_hypothesis(
            sample, dist, &weights[..], hypotheses
        );

        let lhs = local_best_edge - worst_edge;
        let rhs = global_best_edge - current_edge;
        if lhs >= rhs {
            // Pairwise update!
            let max_stepsize = weights[position_of_worst_one];

            // Find the best stepsize by line-search
            let local_best_margins = utils::margins_of_hypothesis(
                sample, &hypotheses[position_of_local_best_one]
            );
            let worst_margins = utils::margins_of_hypothesis(
                sample, &hypotheses[position_of_worst_one]
            );

            let dir_margins = local_best_margins.into_iter()
                .zip(worst_margins)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>();

            // You don't need to remove the newly attaind hypothesis
            // since the weight of new one is assigned as 0 at this point.
            let base_margins = utils::margins_of_weighted_hypothesis(
                sample, &weights[..], hypotheses,
            );

            let margins = dir_margins.iter()
                .zip(base_margins.iter())
                .map(|(dir, cur)| cur + max_stepsize * dir)
                .collect::<Vec<_>>();

            let dist = utils::deformed_exp_distribution(
                self.deform, self.eta, self.nu, sample, &margins[..], hypotheses,
            );

            // If the max step size is the best one,
            // 1. Set 0 weight on `position_of_worst_one`,
            // 2. Set `max_stepsize` on `position_of_local_best_one`.
            if utils::inner_product(&dist[..], &dir_margins[..]) <= 0.0 {
                weights[position_of_new_one] = max_stepsize;
                weights[position_of_worst_one] = 0.0;
                return weights;
            }

            let mut ub = max_stepsize;
            let mut lb = 0.0;
            while ub - lb > SUB_TOLERANCE {
                let step_size = (lb + ub) / 2.0;

                let margins = base_margins.iter()
                    .zip(&dir_margins[..])
                    .map(|(&b, &d)| b + step_size * d);
                let dist = utils::deformed_exp_distribution_from_margins(
                    self.deform, self.eta, self.nu, margins,
                );

                // Compute the gradient for the direction `dir`.
                let dot = utils::inner_product(&dist[..], &dir_margins[..]);

                if dot < 0.0 {
                    lb = step_size;
                } else if dot > 0.0 {
                    ub = step_size;
                } else {
                    break;
                }
            }

            // Update the weights
            let step_size = (lb + ub) / 2.0;

            weights[position_of_local_best_one] += step_size;
            weights[position_of_worst_one] -= step_size;

            weights
        } else {
            // Ordinal FW update!
            // Find the best step size over [0, 1].
            self.line_search(
                sample, hypotheses, position_of_new_one, weights,
            )
        }
    }
}

/// Take the interior point of the given two arrays.
pub(crate) fn interior_point(
    step_size: f64,
    new_basis: usize,
    base: Vec<f64>,
) -> Vec<f64>
{
    checker::check_stepsize(step_size);
    base.into_iter()
        .enumerate()
        .map(|(j, b)| {
            let dir = if j == new_basis { 1.0 - b } else { -b };

            b + step_size * dir
        })
        .collect()
}


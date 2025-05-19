//! This file defines some options of MLPBoost.
use miniboosts_core::{
    tools::{
        checkers,
        helpers,
    },
    constants::BINARY_SEARCH_TOLERANCE,
};
use crate::objective_function::ObjectiveFunction;
use std::fmt;

/// FwUpdateRule updates.
/// These options correspond to the Frank-Wolfe strategies.
#[derive(Clone, Copy)]
pub enum FwUpdateRule {
    /// Classic step size. 
    /// This step size uses `2 / (t + 2)`.
    Classic,

    /// Short-step size.
    /// This step size uses the minimizer of the strongly-smooth bound.
    ShortStep,

    /// Line-search step size.
    /// This step size uses the minimizer of the line segment.
    LineSearch,

    /// The Blended-Pairwise Frank-Wolfe, 
    /// See [this paper](https://proceedings.mlr.press/v162/tsuji22a). 
    BlendedPairwise,
}

impl fmt::Display for FwUpdateRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rule = match self {
            Self::Classic => "Classic update",
            Self::ShortStep => "Short-step update",
            Self::LineSearch => "Line-search update",
            Self::BlendedPairwise => "Blended Pairwise update",
        };
        write!(f, "{rule}")
    }
}

#[derive(Clone)]
pub enum StepSize {
    /// An ordinal Frank-Wolfe step.
    Normal(f64),
    /// Moves the weight on an away atom to a local fw atom.
    BpfwMoveWeights {
        stepsize: f64,
        dir: Vec<f64>,
    },
}

/// The Frank-Wolfe algorithm.
pub struct FrankWolfe<F> {
    objective: F,
    update_rule: FwUpdateRule,
    iteration: usize,
}

impl<F> FrankWolfe<F> {
    /// Creates a new instance of `FrankWolfe.`
    pub fn new(
        objective: F,
        update_rule: FwUpdateRule,
    ) -> Self
    {
        let iteration = 0;
        Self { objective, update_rule, iteration, }
    }

    /// Sets `FwUpdateRule.`
    pub fn update_rule(&mut self, update_rule: FwUpdateRule) {
        self.update_rule = update_rule;
    }

    /// Returns the current Frank-Wolfe strategy.
    pub fn current_update_rule(&self) -> FwUpdateRule {
        self.update_rule
    }

    /// Sets current iteration.
    pub fn iteration(&mut self, iteration: usize) {
        self.iteration = iteration;
    }

    /// Creates a new instance of `FrankWolfe` by
    /// changing the objective function.
    pub fn objective(self, objective: F) -> Self {
        Self {
            objective,
            update_rule: self.update_rule,
            iteration: self.iteration,
        }
    }

    /// Updates the current weights on hypotheses 
    /// based on the classic step size.
    /// ```txt
    ///     w_{t+1} = w_{t} + λ ( e_{h_{t+1}} - w_{t} )
    ///     where λ = 2 / (t + 2) and t is the current iteration.
    /// ```
    /// The last entry of `hypotheses` is `h_{t+1}` in the above update rule.
    pub fn classic(&self, _cur: &[f64], _nxt: &[f64]) -> StepSize {
        let stepsize = 2f64 / (self.iteration as f64 + 2f64);
        StepSize::Normal(stepsize)
    }
}

impl<F> FrankWolfe<F>
    where F: ObjectiveFunction,
{
    /// Updates the current weights on hypotheses 
    /// based on the short-step size.
    /// ```txt
    ///     w_{t+1} = w_{t} + λ ( e_{h_{t+1}} - w_{t} )
    ///     where λ = 
    /// ```
    /// The last entry of `hypotheses` is `h_{t+1}` in the above update rule.
    pub fn short_step(&self, cur: &[f64], nxt: &[f64]) -> StepSize {
        if cur.len() == 1 { return StepSize::Normal(1f64); }
        // if cur.len() == 1 { return vec![1f64]; }

        let grad = self.objective.gradient(cur);

        let (s, p) = self.objective.smooth();
        let numer = helpers::inner_product(&grad[..], cur)
            - helpers::inner_product(&grad[..], nxt);
        let denom = nxt.iter()
            .zip(cur)
            .fold(0f64, |acc, (n, c)| {
                if p == f64::MAX {
                    (n - c).abs().max(acc)
                } else {
                    acc + (n - c).abs().powf(p)
                }
            })
            .powf(if p == f64::MAX { 2f64 } else { 2f64 / p });

        let stepsize = (numer / (s * denom)).clamp(0f64, 1f64);
        StepSize::Normal(stepsize)
    }

    /// Seeks the best step size `λ ∈ [0, 1]` by line-search.
    /// ```txt
    /// λ =   arg min  f( (-Aw) + λ ((-Ae) - (-Aw)) )
    ///     λ ∈ [0, 1]
    /// ```
    fn line_search(&self, cur: &[f64], nxt: &[f64]) -> StepSize {
        let dir = nxt.iter()
            .zip(cur)
            .map(|(n, c)| n - c)
            .collect::<Vec<_>>();

        let stepsize = self.line_search_inner(cur, &dir[..], 1f64);
        StepSize::Normal(stepsize)
    }

    fn line_search_inner(&self, cur: &[f64], dir: &[f64], mut ub: f64) -> f64 {
        let mut lb = 0f64;

        assert!(
            lb < ub,
            "the line segment is empty. [lb, ub] = [{lb}, {ub}]",
        );

        while ub - lb > BINARY_SEARCH_TOLERANCE {
            let stepsize = (lb + ub) / 2f64;
            let mid = cur.iter()
                .zip(dir)
                .map(|(c, d)| c + stepsize * d)
                .collect::<Vec<_>>();
            let grad = self.objective.gradient(&mid[..]);
            let val = helpers::inner_product(&grad[..], dir);
            if val < 0f64 {
                lb = stepsize;
            } else if val > 0f64 {
                ub = stepsize;
            } else {
                break;
            }
        }

        (lb + ub) / 2f64
    }

    fn find_away_atom(&self, cur: &[f64], grad: &[f64]) -> (f64, Vec<f64>) {
        let (ix, _) = grad.iter()
            .zip(cur)
            .enumerate()
            .fold((usize::MAX, f64::MIN), |acc, (i, (&g, &c))| {
                if c == 0f64 || g < acc.1 {
                    acc
                } else {
                    (i, g)
                }
            });
        assert_ne!(
            ix, usize::MAX,
            "failed to find an away atom."
        );

        let dim = cur.len();
        let mut away = vec![0f64; dim];
        away[ix] = 1f64;
        (cur[ix], away)
    }

    fn find_localfw_atom(&self, cur: &[f64], grad: &[f64]) -> (f64, Vec<f64>) {
        let (ix, _) = grad.iter()
            .zip(cur)
            .enumerate()
            .fold((usize::MAX, f64::MAX), |acc, (i, (&g, &c))| {
                if c == 0f64 || acc.1 < g {
                    acc
                } else {
                    (i, g)
                }
            });
        assert_ne!(
            ix, usize::MAX,
            "failed to find an local fw atom."
        );

        let dim = cur.len();
        let mut localfw = vec![0f64; dim];
        localfw[ix] = 1f64;
        (cur[ix], localfw)
    }

    /// The blended pairwise Frank-Wolfe algorithm.
    /// This function assumes that `nxt` is a canonical basis vector.
    fn bpfw(&self, cur: &[f64], nxt: &[f64]) -> StepSize {
        let grad = self.objective.gradient(cur);

        let (ub, a) = self.find_away_atom(cur, &grad[..]);
        let (_,  s) = self.find_localfw_atom(cur, &grad[..]);
        let x = cur;
        let w = nxt;
        let sa = s.iter()
            .zip(&a[..])
            .map(|(si, ai)| si - ai)
            .collect::<Vec<_>>();
        let wx = w.iter()
            .zip(x)
            .map(|(wi, xi)| wi - xi)
            .collect::<Vec<_>>();

        let stepsize;
        let dir;
        if helpers::inner_product(&grad[..], &sa[..])
            <= helpers::inner_product(&grad[..], &wx[..]) {
            dir = sa;
            stepsize = self.line_search_inner(x, &dir[..], ub);
            StepSize::BpfwMoveWeights {
                stepsize,
                dir,
            }
        } else {
            dir = wx;
            stepsize = self.line_search_inner(x, &dir[..], 1f64);
            StepSize::Normal(stepsize)
        }
    }

    /// Get the step size for Frank-Wolfe update.
    pub fn get_stepsize_mut(&mut self, cur: &[f64], nxt: &[f64]) -> StepSize {
        let stepsize = match self.update_rule {
            FwUpdateRule::Classic         => self.classic(cur, nxt),
            FwUpdateRule::ShortStep       => self.short_step(cur, nxt),
            FwUpdateRule::LineSearch      => self.line_search(cur, nxt),
            FwUpdateRule::BlendedPairwise => self.bpfw(cur, nxt),
        };
        self.iteration += 1;

        stepsize
    }

    /// Updates the current weights on hypotheses.
    /// ```txt
    ///     w_{t+1} = w_{t} + λ ( e_{h_{t+1}} - w_{t} )
    /// ```
    /// The last entry of `hypotheses` is `h_{t+1}` in the above update rule.
    pub fn next(&mut self, cur: Vec<f64>, nxt: Vec<f64>) -> Vec<f64> {
        match self.get_stepsize_mut(&cur[..], &nxt[..]) {
            StepSize::Normal(stepsize) => interior_point(cur, nxt, stepsize),
            StepSize::BpfwMoveWeights { stepsize, dir } => {
                move_to_dir(cur, dir, stepsize)
            },
        }
    }
}

/// Take the interior point of the given two arrays.
pub(crate) fn interior_point(
    cur: Vec<f64>,
    nxt: Vec<f64>,
    stepsize: f64,
) -> Vec<f64>
{
    checkers::stepsize(stepsize);
    cur.iter()
        .zip(nxt)
        .map(|(c, n)| c + stepsize * (n - c))
        .collect()
}
/// Take the interior point of the given two arrays.
pub(crate) fn move_to_dir(
    cur: Vec<f64>,
    dir: Vec<f64>,
    stepsize: f64,
) -> Vec<f64>
{
    cur.iter()
        .zip(dir)
        .map(|(c, d)| c + stepsize * d)
        .collect()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::objective_function::Entropy;

    const TEST_TOLERANCE: f64 = 1e-3;

    /// ℓ₂-norm objective
    struct L2;
    impl L2 {
        fn new() -> Self { Self {} }
    }
    impl ObjectiveFunction for L2 {
        fn name(&self) -> &str { "L2-norm objective" }

        fn smooth(&self) -> (f64, f64) { (1f64, 2f64) }

        fn objective_value(&self, point: &[f64]) -> f64 {
            point.iter()
                .map(|p| p.powi(2))
                .sum::<f64>()
                / 2f64
        }

        fn gradient(&self, point: &[f64]) -> Vec<f64> {
            point.to_vec()
        }
    }

    /// Returns a point `v ∈ [-1, +1]ⁿ` that minimizes
    /// the inner product `v · ∇f(x).`
    fn linear_minimization_oracle_l2(grad: &[f64]) -> Vec<f64> {
        grad.iter()
            .map(|&g| if g < 0f64 { 1f64 } else { -1f64 })
            .collect::<Vec<_>>()
    }

    /// Returns a point `v ∈ [0, +1]ⁿ` that minimizes
    /// the inner product `v · ∇f(x).`
    fn linear_minimization_oracle_entropy(grad: &[f64]) -> Vec<f64> {
        let dim = grad.len();
        let (i, v) = grad.iter()
            .enumerate()
            .fold((dim, f64::MAX), |acc, (i, &g)| {
                if acc.1 > g {
                    (i, g)
                } else {
                    acc
                }
            });
        assert!(
            i < dim && v < f64::MAX,
            "failed to call linear minimization oracle."
        );
        let mut minimizer = vec![0f64; dim];
        minimizer[i] = 1f64;
        minimizer
    }

    /// Generate an initial point `e₁ ∈ [-1, +1]ⁿ`, a basis vector.
    fn initial_point_l2(dim: usize) -> Vec<f64> {
        let mut point = vec![0f64; dim];
        point[0] = 1f64;
        point
    }

    /// Generate an initial point `x₁ ∈ [0, +1]ⁿ`, a basis vector.
    /// In order not to take infinite value, we choose the following one
    /// as the initial point.
    /// ```txt
    ///     x₁[i] = 1 - (n-1)ε,     if i = 1,
    ///             ε,              if i ≠ 1.
    /// ```
    fn initial_point_entropy(dim: usize) -> Vec<f64> {
        const EPS: f64 = 1e-9;
        let mut point = vec![EPS; dim];
        point[0] = 1f64 - (dim - 1) as f64 * EPS;
        point
    }

    /// Returns the optimal solution to the test optimization problem
    fn optimal_solution_l2(dim: usize) -> Vec<f64> {
        vec![0f64; dim]
    }

    /// Returns the optimal solution to the test optimization problem
    fn optimal_solution_entropy(dim: usize) -> Vec<f64> {
        vec![1f64 / dim as f64; dim]
    }

    fn stopping_criterion(nxt: &[f64], cur: &[f64], grad: &[f64]) -> bool {
        let duality_gap = nxt.iter()
            .zip(&cur[..])
            .zip(&grad[..])
            .map(|((n, c), g)| (c - n) * g)
            .sum::<f64>();
        duality_gap < TEST_TOLERANCE
    }

    #[test]
    fn classic_frank_wolfe_l2_objective() {
        let dim      = 100usize;
        let f        = L2::new();
        let mut x    = initial_point_l2(dim);
        let mut algo = FrankWolfe::new(L2::new(), FwUpdateRule::Classic);

        loop {
            let g = f.gradient(&x[..]);
            let v = linear_minimization_oracle_l2(&g[..]);
            if stopping_criterion(&v[..], &x[..], &g[..]) {
                break;
            }
            x = algo.next(x, v);
        }

        let xoptimal = optimal_solution_l2(dim);
        let optval = f.objective_value(&xoptimal[..]);
        let objval = f.objective_value(&x[..]);

        assert!(
            objval - optval < TEST_TOLERANCE,
            "expected {optval}, got {objval}."
        );
    }

    #[test]
    fn shortstep_frank_wolfe_l2_objective() {
        let dim      = 100usize;
        let f        = L2::new();
        let mut x    = initial_point_l2(dim);
        let mut algo = FrankWolfe::new(L2::new(), FwUpdateRule::ShortStep);

        loop {
            let g = f.gradient(&x[..]);
            let v = linear_minimization_oracle_l2(&g[..]);
            if stopping_criterion(&v[..], &x[..], &g[..]) {
                break;
            }
            x = algo.next(x, v);
        }

        let xoptimal = optimal_solution_l2(dim);
        let optval = f.objective_value(&xoptimal[..]);
        let objval = f.objective_value(&x[..]);

        assert!(
            objval - optval < TEST_TOLERANCE,
            "expected {optval}, got {objval}."
        );
    }

    #[test]
    fn line_search_frank_wolfe_l2_objective() {
        let dim      = 100usize;
        let f        = L2::new();
        let mut x    = initial_point_l2(dim);
        let mut algo = FrankWolfe::new(L2::new(), FwUpdateRule::LineSearch);

        loop {
            let g = f.gradient(&x[..]);
            let v = linear_minimization_oracle_l2(&g[..]);
            if stopping_criterion(&v[..], &x[..], &g[..]) {
                break;
            }
            x = algo.next(x, v);
        }

        let xoptimal = optimal_solution_l2(dim);
        let optval = f.objective_value(&xoptimal[..]);
        let objval = f.objective_value(&x[..]);

        assert!(
            objval - optval < TEST_TOLERANCE,
            "expected {optval}, got {objval}."
        );
    }

    #[test]
    fn bpfw_frank_wolfe_l2_objective() {
        let dim      = 100usize;
        let f        = L2::new();
        let mut x    = initial_point_l2(dim);
        let mut algo = FrankWolfe::new(
            L2::new(),
            FwUpdateRule::BlendedPairwise,
        );

        loop {
            let g = f.gradient(&x[..]);
            let v = linear_minimization_oracle_l2(&g[..]);
            if stopping_criterion(&v[..], &x[..], &g[..]) {
                break;
            }
            x = algo.next(x, v);
        }

        let xoptimal = optimal_solution_l2(dim);
        let optval = f.objective_value(&xoptimal[..]);
        let objval = f.objective_value(&x[..]);

        assert!(
            objval - optval < TEST_TOLERANCE,
            "expected {optval}, got {objval}."
        );
    }

    #[test]
    fn classic_frank_wolfe_entropy_objective() {
        let dim      = 100usize;
        let f        = Entropy::new();
        let mut x    = initial_point_entropy(dim);
        let mut algo = FrankWolfe::new(Entropy::new(), FwUpdateRule::Classic);

        loop {
            let g = f.gradient(&x[..]);
            let v = linear_minimization_oracle_entropy(&g[..]);
            if stopping_criterion(&v[..], &x[..], &g[..]) {
                break;
            }
            x = algo.next(x, v);
        }

        let xoptimal = optimal_solution_entropy(dim);
        let optval = f.objective_value(&xoptimal[..]);
        let objval = f.objective_value(&x[..]);

        assert!(
            objval - optval < TEST_TOLERANCE,
            "expected {optval}, got {objval}."
        );
    }

    #[test]
    fn line_search_frank_wolfe_entropy_objective() {
        let dim      = 100usize;
        let f        = Entropy::new();
        let mut x    = initial_point_entropy(dim);
        let mut algo = FrankWolfe::new(
            Entropy::new(),
            FwUpdateRule::LineSearch,
        );

        loop {
            let g = f.gradient(&x[..]);
            let v = linear_minimization_oracle_entropy(&g[..]);
            if stopping_criterion(&v[..], &x[..], &g[..]) {
                break;
            }
            x = algo.next(x, v);
        }

        let xoptimal = optimal_solution_entropy(dim);
        let optval = f.objective_value(&xoptimal[..]);
        let objval = f.objective_value(&x[..]);

        assert!(
            objval - optval < TEST_TOLERANCE,
            "expected {optval}, got {objval}."
        );
    }

    #[test]
    fn bpfw_frank_wolfe_entropy_objective() {
        let dim      = 100usize;
        let f        = Entropy::new();
        let mut x    = initial_point_entropy(dim);
        let mut algo = FrankWolfe::new(
            Entropy::new(),
            FwUpdateRule::BlendedPairwise,
        );

        loop {
            let g = f.gradient(&x[..]);
            let v = linear_minimization_oracle_entropy(&g[..]);
            if stopping_criterion(&v[..], &x[..], &g[..]) {
                break;
            }
            x = algo.next(x, v);
        }

        let xoptimal = optimal_solution_entropy(dim);
        let optval = f.objective_value(&xoptimal[..]);
        let objval = f.objective_value(&x[..]);

        assert!(
            objval - optval < TEST_TOLERANCE,
            "expected {optval}, got {objval}."
        );
    }
}


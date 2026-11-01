use crate::core::tools::helpers;

/// This trait defines the loss functions.
pub trait ObjectiveFunction {
    /// The name of the loss function.
    fn name(&self) -> &str;

    /// Returns the smoothness constant and norm.
    /// We say that a function `f` is L-smooth w.r.t. l_p-norm if
    /// ```txt
    /// f(y) ≤ f(x) + ∇f(x)∙(y - x) + (L/2) ‖y - x‖_p^2
    /// ```
    /// This method returns the pair `(L, p).`
    fn smooth(&self) -> (f64, f64);

    /// Loss value for a single point.
    fn objective_value(&self, point: &[f64]) -> f64;

    /// Gradient vector at the current point.
    fn gradient(&self, point: &[f64]) -> Vec<f64>;
}

pub struct SoftMarginObjective(f64);

impl SoftMarginObjective {
    pub fn new(capping: f64) -> Self {
        Self(capping)
    }
}

impl ObjectiveFunction for SoftMarginObjective {
    fn name(&self) -> &str {
        "Soft Margin Objective"
    }

    fn smooth(&self) -> (f64, f64) {
        (0f64, 0f64)
    }

    /// returns the objective value `f^*(θ)` at given point `θ = -Aw.`
    fn objective_value(&self, point: &[f64]) -> f64 {
        let dist = self.gradient(point);
        -helpers::inner_product(point, &dist[..])
    }

    /// returns the gradient `∇f^*(θ)` at given point `θ = -Aw.`
    fn gradient(&self, point: &[f64]) -> Vec<f64> {
        let m = point.len();
        let indices = {
            let mut indices = (0..m).collect::<Vec<_>>();
            indices.sort_by(|&i, &j| point[j].partial_cmp(&point[i]).unwrap());
            indices
        };
        let mut grad = vec![0f64; m];

        let cap = 1f64 / self.0;
        let mut rest = 1f64;
        for i in indices {
            if rest > cap {
                grad[i] = cap;
                rest -= cap;
            } else {
                grad[i] = rest;
                break;
            }
        }
        grad
    }
}

/// The objective function
/// for soft margin optimization with entropy regularization.
#[derive(Clone)]
pub struct ErlpSoftMarginObjective {
    nu: f64,
    eta: f64,
}

impl ErlpSoftMarginObjective {
    pub fn new(nu: f64, eta: f64) -> Self {
        Self { nu, eta }
    }
}

impl ObjectiveFunction for ErlpSoftMarginObjective {
    fn name(&self) -> &str {
        "The dual objective for ErlpBoost"
    }

    fn smooth(&self) -> (f64, f64) {
        (self.eta, f64::MAX)
    }

    /// returns the objective value `f^*(θ)` at given point `θ = -Aw.`
    fn objective_value(&self, point: &[f64]) -> f64 {
        let dist = self.gradient(point);
        let edge = -helpers::inner_product(point, &dist[..]);
        let entropy = helpers::entropy_from_uni_distribution(&dist[..]);

        edge + (entropy / self.eta)
    }

    /// returns the gradient `∇f^*(θ)` at given point `θ = -Aw.`
    fn gradient(&self, point: &[f64]) -> Vec<f64> {
        let point = point.iter().map(|&pi| -pi);
        helpers::exp_distribution_from_margins(self.eta, self.nu, point)
    }
}

/// Entropy objective.
/// Note that this function is **NOT** smooth,
/// so we do not test the short-step size.
/// (I believe that other step sizes work without convergence guarantees)
pub struct Entropy;

impl Entropy {
    pub fn new() -> Self {
        Self {}
    }
}

impl ObjectiveFunction for Entropy {
    fn name(&self) -> &str {
        "Entropy"
    }

    fn smooth(&self) -> (f64, f64) {
        (1f64, 1f64)
    }

    fn objective_value(&self, point: &[f64]) -> f64 {
        point
            .iter()
            .map(|&p| if p == 0f64 { 0f64 } else { p * p.ln() })
            .sum::<f64>()
    }

    fn gradient(&self, point: &[f64]) -> Vec<f64> {
        point
            .iter()
            .map(|&p| if p != 0f64 { 1f64 + p.ln() } else { f64::MIN })
            .collect()
    }
}

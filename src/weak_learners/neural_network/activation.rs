//! Activation functions of the hidden layer and their initialization.

use serde::{Deserialize, Serialize};

/// The activation function `φ` of the hidden layer.
///
/// # Examples
/// ```
/// use miniboosts::weak_learners::neural_network::Activation;
///
/// assert_eq!(Activation::Relu.apply(-1.0), 0.0);
/// assert!((Activation::Softplus.apply(0.0) - 2f64.ln()).abs() < 1e-15);
/// assert_eq!(Activation::Tanh.apply(0.0), 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Activation {
    /// `tanh(z)`. Smooth, so the training objective is smooth.
    Tanh,
    /// `max(z, 0)`. Not differentiable at `0`, where the derivative is taken
    /// to be `0`; the training objective is then not smooth, which the
    /// convergence guarantees of [`Adam`](crate::Adam) and
    /// [`Adopt`](crate::Adopt) assume.
    Relu,
    /// `ln(1 + e^z)`, a smooth approximation of `max(z, 0)`.
    Softplus,
}

impl Activation {
    /// Returns `φ(z)`.
    ///
    /// # Examples
    /// See [`Activation`].
    pub fn apply(self, z: f64) -> f64 {
        match self {
            Self::Tanh => z.tanh(),
            Self::Relu => z.max(0.0),
            // ln(1 + e^z) = max(z, 0) + ln(1 + e^{-|z|}), without overflow.
            Self::Softplus => z.max(0.0) + (-z.abs()).exp().ln_1p(),
        }
    }

    /// Returns `φ'(z)`; for [`Activation::Relu`], `0` at `z = 0`.
    pub(crate) fn derivative(self, z: f64) -> f64 {
        match self {
            Self::Tanh => {
                let t = z.tanh();
                1.0 - t * t
            }
            Self::Relu => {
                if z > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Softplus => logistic(z),
        }
    }

    /// The half-width `a` of the uniform distribution `U(-a, a)` that
    /// initializes the weights of a layer with `fan_in` inputs and `fan_out`
    /// outputs followed by this activation: `sqrt(6 / (fan_in + fan_out))`
    /// for `tanh` (Glorot and Bengio, 2010, Eq. 16), and `sqrt(6 / fan_in)`
    /// for the rectifier-like `max(z, 0)` and softplus (He et al., 2015,
    /// variance `2 / fan_in`).
    pub(crate) fn init_bound(self, fan_in: usize, fan_out: usize) -> f64 {
        match self {
            Self::Tanh => (6.0 / (fan_in + fan_out) as f64).sqrt(),
            Self::Relu | Self::Softplus => (6.0 / fan_in as f64).sqrt(),
        }
    }
}

impl std::fmt::Display for Activation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Tanh => "tanh",
            Self::Relu => "ReLU",
            Self::Softplus => "softplus",
        };
        f.write_str(name)
    }
}

/// `1 / (1 + e^{-z})`, without overflow.
pub(crate) fn logistic(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivatives_match_finite_differences() {
        // Central differences have error O(h²) ≈ 1e-10 for these smooth
        // functions; 1e-7 leaves room for rounding (≈ 1e-16 / h).
        let h = 1e-5;
        for activation in [Activation::Tanh, Activation::Relu, Activation::Softplus] {
            for z in [-3.0, -0.5, 0.25, 2.0, 40.0, -40.0] {
                let numeric = (activation.apply(z + h) - activation.apply(z - h)) / (2.0 * h);
                assert!(
                    (numeric - activation.derivative(z)).abs() < 1e-7,
                    "{activation} at {z}"
                );
            }
        }
        assert_eq!(Activation::Relu.derivative(0.0), 0.0);
        // No overflow for large arguments.
        assert_eq!(Activation::Softplus.apply(1000.0), 1000.0);
        assert_eq!(Activation::Softplus.apply(-1000.0), 0.0);
    }
}

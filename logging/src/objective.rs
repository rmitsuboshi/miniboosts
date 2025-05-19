use miniboosts_core::{
    Sample,
    Classifier,
    helpers,
};
use optimization::{
    ObjectiveFunction,
    SoftMarginObjective,
};

pub trait LoggingObjective {
    fn name(&self) -> String;
    fn objective_value<H: Classifier>(&self, sample: &Sample, f: &H) -> f64;
}

pub struct LoggingSoftMarginObjective(SoftMarginObjective);

impl LoggingSoftMarginObjective {
    pub fn new(nu: f64) -> Self {
        let obj = SoftMarginObjective::new(nu);
        Self(obj)
    }
}

impl LoggingObjective for LoggingSoftMarginObjective {
    fn name(&self) -> String {
        self.0.name().to_string()
    }

    fn objective_value<H: Classifier>(&self, sample: &Sample, f: &H) -> f64 {
        let neg_margins = helpers::margins(sample, f)
            .map(|yf| -yf)
            .collect::<Vec<_>>();
        self.0.objective_value(&neg_margins[..])
    }
}


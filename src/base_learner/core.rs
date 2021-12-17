use crate::data_type::{Data, Label, Sample};

pub trait Classifier<D, L> {
    fn predict(&self, example: &Data<D>) -> Label<L>;


    fn predict_all(&self, examples: &[Data<D>]) -> Vec<Label<L>> {
        examples.iter()
                .map(|example| self.predict(&example))
                .collect()
    }
}


pub trait BaseLearner<D, L> {
    fn best_hypothesis(&self, sample: &Sample<D, L>, distribution: &[f64]) -> Box<dyn Classifier<D, L>>;
}


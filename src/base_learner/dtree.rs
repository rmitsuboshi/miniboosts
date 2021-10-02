use crate::data_type::{Data, Label, Sample};
use crate::base_learner::core::{BaseLearner, Classifier};
use crate::base_learner::dstump::DStump;


struct Node {
    left:  Option<Box<Node>>,
    right: Option<Box<Node>>,
    condition: Box<dyn Classifier>,
}


impl Node {
    fn descent<D, L>(&self, _example: &Data<D>) -> Label<L> {
        let mut prediction = self.condition.predict(_example);


        if prediction < 0.0 {
            if let Some(l) = &self.left {
                prediction = l.descent(_example);
            };
        } else {
            if let Some(r) = &self.right {
                prediction = r.descent(_example);
            };
        }
        prediction
    }
}


pub struct DTreeClassifier {
    root: Box<Node>
}


impl Classifier<D, L> for DTreeClassifier {
    fn predict(&self, example: &Data<D>) -> Label<L> {
        self.root.descent(example)
    }
}



pub struct DTree {
    max_depth: Option<usize>,     // `None` implies unlimited depth
    // max_node:  Option<usize>
}


impl DTree {
    pub fn new() -> DTree {
        DTree { max_depth: None }
    }


    pub fn with_sample<D, L>(sample: &Sample<D, L>) -> DTree {
        let max_depth = (sample.len() as f64).log2() as usize;
        let max_depth = Some(max_depth);
        DTree { max_depth }
    }


    pub fn with_depth(max_depth: usize) -> DTree {
        let max_depth = Some(max_depth);
        DTree { max_depth }
    }


    fn construct_tree<D, L>(&self, sample: &Sample<D, L>, _indices: Vec<usize>, distribution: &[f64], depth: usize) -> Box<Node> {
        // Get best split condition by decision stump
        let mut sub_sample = Vec::with_capacity(_indices.len());
        let mut sub_distribution = Vec::with_capacity(_indices.len());
        let mut _sum = 0.0;
        for &i in _indices.iter() {
            sub_sample.push(sample[i].clone());
            sub_distribution.push(distribution[i]);
            _sum += distribution[i];
        }
        let sub_distribution = sub_distribution.iter().map(|&d| d / _sum).collect::<Vec<f64>>();
        let dstump = DStump::with_sample(&sub_sample);
        let f = dstump.best_hypothesis(&sub_sample, &sub_distribution);


        // Split indices
        let (l_indices, r_indices): (Vec<usize>, Vec<usize>) = _indices.into_iter().partition(|&i| f.predict(&sample[i].0) < 0.0);


        // Construct children
        let (left, right) = match self.max_depth {
            None => {
                let _left  = self.construct_tree(sample, l_indices, distribution, depth + 1);
                let _right = self.construct_tree(sample, r_indices, distribution, depth + 1);
                (Some(_left), Some(_right))
            },
            Some(max_depth) => {
                if depth < max_depth {
                    (None, None)
                } else {
                    let _left  = self.construct_tree(sample, l_indices, distribution, depth + 1);
                    let _right = self.construct_tree(sample, r_indices, distribution, depth + 1);
                    (Some(_left), Some(_right))
                }
            }
        };


        Box::new(
            Node { left, right, condition: f }
        )
    }
}

impl BaseLearner<D, L> for DTree {
    fn best_hypothesis(&self, sample: &Sample<D, L>, distribution: &[f64]) -> Box<dyn Classifier> {
        let indices = (0..distribution.len()).collect::<Vec<usize>>();
        // let mut _distribution = vec![0.0; distribution.len()];
        // _distribution.copy_from_slice(distribution);


        let root = self.construct_tree(sample, indices, distribution, 0_usize);


        let dtree = DTreeClassifier { root };
        Box::new(dtree)
    }
}





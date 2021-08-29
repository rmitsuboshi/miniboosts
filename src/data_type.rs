pub type Sample = Vec<(Vec<f64>, f64)>;


pub fn to_sample(examples: Vec<Vec<f64>>, labels: Vec<f64>) -> Sample {
    // let mut sample = Vec::new();
    examples.into_iter()
            .zip(labels)
            .collect::<Sample>()
}


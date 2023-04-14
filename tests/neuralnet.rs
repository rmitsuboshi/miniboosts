use std::env;
use miniboosts::prelude::*;



/// Tests for `NeuralNetwork`.
#[cfg(test)]
pub mod neuralnetwork_tests {
    use super::*;
    #[test]
    fn german() {
        let mut path = env::current_dir().unwrap();
        path.push("tests/dataset/german.csv");

        let sample = Sample::from_csv(path, true)
            .unwrap()
            .set_target("class");
        let n_sample = sample.shape().0;

        let nn = NeuralNetwork::init(&sample)
            .append(100, Activation::ReLu(1.0))
            .append(2, Activation::SoftMax(1.0))
            .n_epoch(10)
            .n_iter(100);


        let dist = vec![1.0 / n_sample as f64; n_sample];
        let f = nn.produce(&sample, &dist[..]);

        f.stats();


        let (m, _) = sample.shape();
        let predictions = f.predict_all(&sample);

        let loss = sample.target()
            .into_iter()
            .zip(predictions)
            .map(|(t, p)| if *t != p as f64 { 1.0 } else { 0.0 })
            .sum::<f64>() / m as f64;

        println!("Loss (german.csv, NN): {loss}");
        assert!(true);
    }
}

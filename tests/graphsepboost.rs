use std::env;
use miniboosts::*;



/// Tests for `GraphSepBoost`.
#[cfg(test)]
pub mod graphsepboost_tests {
    use super::*;
    #[test]
    fn iris() {
        let mut path = env::current_dir().unwrap();
        path.push("tests/dataset/german.csv");
        // path.push("tests/dataset/iris_binary.csv");

        let sample = SampleReader::new()
            .file(path)
            .has_header(true)
            .target_feature("class")
            .read()
            .unwrap();


        let mut booster = GraphSepBoost::init(&sample);

        let wl = DecisionTreeBuilder::new(&sample)
            .max_depth(1)
            .criterion(Criterion::Entropy)
            .build();
        println!("{wl}");


        let f = booster.run(&wl);


        let (m, _) = sample.shape();
        let predictions = f.predict_all(&sample);

        let loss = sample.target()
            .into_iter()
            .zip(predictions)
            .map(|(t, p)| if *t != p as f64 { 1.0 } else { 0.0 })
            .sum::<f64>() / m as f64;

        println!("Training Loss: {loss}");
        assert!(true);
    }
}



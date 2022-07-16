use polars::prelude::*;

use std::env;

use lycaon::Booster;
use lycaon::AdaBoost;
use lycaon::{Classifier, DTree};



/// Tests for `AdaBoost`.
#[cfg(test)]
pub mod adaboost_iris {
    use super::*;
    #[test]
    fn iris() {
        let mut path = env::current_dir().unwrap();
        println!("path: {:?}", path);
        path.push("tests/iris.csv");

        let df = CsvReader::from_path(path)
            .unwrap()
            .has_header(true)
            .finish()
            .unwrap();


        let mask = df.column("class")
            .unwrap()
            .i64()
            .unwrap()
            .not_equal(0);

        let mut df = df.filter(&mask).unwrap();
        let data = df.apply("class", |col| {
                col.i64().unwrap().into_iter()
                    .map(|v| v.map(|i| if i == 1 { -1 } else { 1 }))
                    .collect::<Int64Chunked>()
            }).unwrap();


        let target = data.drop_in_place(&"class").unwrap();


        let mut adaboost = AdaBoost::init(&data);
        let dtree = DTree::init(&data);

        dtree.seed(0);


        let f = adaboost.run(&dtree, &data, &target, 0.1);


        let (m, _) = data.shape();
        let predictions = f.predict_all(&data);

        let loss = target.i64().unwrap()
            .into_iter()
            .zip(predictions)
            .map(|(t, p)| if t.unwrap() != p { 1.0 } else { 0.0 })
            .sum::<f64>() / m as f64;

        println!("Loss (iris.csv, AdaBoost, DTree): {loss}");
        println!("classifier: {f:?}");
        assert!(true);
    }
}



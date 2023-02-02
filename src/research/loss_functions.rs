/// Zero-one loss
pub fn zero_one_loss(true_label: f64, prediction: f64) -> f64 {
    let prediction = if prediction > 0.0 { 1.0 } else { -1.0 };
    if true_label * prediction > 0.0 { 0.0 } else { 1.0 }
}

/// Squared loss
pub fn squared_loss(true_label: f64, prediction: f64) -> f64 {
    (true_label - prediction).powi(2)
}


/// Absolute loss
pub fn absolute_loss(true_label: f64, prediction: f64) -> f64 {
    (true_label - prediction).abs()
}

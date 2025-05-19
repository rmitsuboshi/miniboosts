use std::mem;
use std::ops::Index;

use crate::{
    tools::helpers,
    constants::BUFFER_SIZE,
};

#[derive(Debug, Clone)]
pub enum Feature {
    Dense {
        name: String,
        vals: Vec<f64>,
    },
    Sparse {
        name: String,
        vals: Vec<(usize, f64)>,
        size: usize,
    },
}

impl Feature {
    pub fn name(&self) -> &str {
        match self {
            Self::Dense  { name, .. } => name,
            Self::Sparse { name, .. } => name,
        }
    }

    pub fn dense<T: ToString>(name: T) -> Self {
        Self::Dense {
            name: name.to_string(),
            vals: Vec::with_capacity(BUFFER_SIZE),
        }
    }

    pub fn sparse<T: ToString>(name: T, size: usize) -> Self {
        Self::Sparse {
            name: name.to_string(),
            vals: Vec::with_capacity(BUFFER_SIZE),
            size,
        }
    }

    pub fn into_vals(self) -> Vec<f64> {
        match self {
            Self::Dense  { vals, .. } => vals,
            Self::Sparse { vals, size, .. } => {
                let mut ret = vec![0f64; size];
                vals.into_iter()
                    .for_each(|(i, v)| { ret[i] = v; });
                ret
            },
        }
    }

    pub fn is_sparse(&self) -> bool {
        match self {
            Self::Dense  { .. } => false,
            Self::Sparse { .. } => true,
        }
    }

    pub fn is_dense(&self) -> bool { !self.is_sparse() }

    pub fn append(&mut self, (ix, val): (usize, f64)) {
        match self {
            Self::Dense  { vals, .. } => vals.push(val),
            Self::Sparse { vals, .. } => {
                if val != 0f64 { vals.push((ix, val)); }
            },
        }
    }

    pub(crate) fn replace_name<T>(&mut self, name: T)
        -> String
        where T: ToString,
    {
        let n = name.to_string();
        match self {
            Self::Dense  { name, .. } => { mem::replace(name, n) },
            Self::Sparse { name, .. } => { mem::replace(name, n) },
        }
    }

    pub(crate) fn set_size(&mut self, size: usize) {
        let s = size;
        if let Self::Sparse { size, .. } = self { *size = s; }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Dense  { vals, .. } => { vals.is_empty() },
            Self::Sparse { vals, .. } => { vals.is_empty() },
        }
    }

    fn zero_counts(&self) -> usize {
        match self {
            Self::Dense  { vals, ..       } => {
                vals.iter()
                    .filter(|&&v| v == 0f64)
                    .count()
            },
            Self::Sparse { vals, size, .. } => { size - vals.len() },
        }
    }

    pub fn has_zero(&self) -> bool {
        self.zero_counts() > 0
    }

    pub fn distinct_value_count(&self) -> usize {
        let mut values = match self {
            Self::Dense  { vals, .. } => { vals.clone() },
            Self::Sparse { vals, .. } => {
                let mut values = vals.iter()
                    .map(|&(_, v)| v)
                    .collect::<Vec<_>>();
                if self.has_zero() { values.push(0f64); }
                values
            },
        };

        if values.is_empty() { return 0; }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (mut count, mut value) = (1, values[0]);
        for &v in values.iter().skip(1) {
            if v != value {
                value = v;
                count += 1;
            }
        }
        count
    }

    pub fn weighted_mean(&self, weights: &[f64]) -> f64 {
        match self {
            Self::Dense { vals, .. } => {
                helpers::inner_product(weights, &vals[..])
            },
            Self::Sparse { vals, .. } => {
                vals.iter()
                    .map(|&(ix, val)| { weights[ix] * val })
                    .sum()
            }
        }
    }

    pub fn weighted_variance(
        &self,
        mean: f64,
        weights: &[f64],
    ) -> f64
    {
        match self {
            Self::Dense { vals, .. } => {
                vals.iter()
                    .zip(weights)
                    .map(|(&v, &w)| w * (v - mean).powi(2))
                    .sum()
            },
            Self::Sparse { vals, .. } => {
                let w0 = {
                    let wsum = weights.iter().sum::<f64>();
                    let nz_w = vals.iter()
                        .map(|&(i, _)| weights[i])
                        .sum::<f64>();
                    wsum - nz_w
                };
                vals.iter()
                    .map(|&(i, v)| weights[i] * (v - mean).powi(2))
                    .sum::<f64>()
                    + w0 * mean.powi(2)
            },
        }
    }

    pub fn weighted_mean_for_label<T>(
        &self,
        label: f64,
        labels:  T,
        weights: T,
    ) -> f64
        where T: AsRef<[f64]>,
    {
        let labels  = labels.as_ref();
        let weights = weights.as_ref();

        match self {
            Self::Dense { vals, .. } => {
                vals.iter()
                    .zip(weights)
                    .zip(labels)
                    .filter(|&(_, &l)| l == label)
                    .map(|((&v, &w), _)| v * w)
                    .sum()
            },
            Self::Sparse { vals, .. } => {
                vals.iter()
                    .filter(|&(i, _)| labels[*i] == label)
                    .map(|&(i, v)| weights[i] * v)
                    .sum()
            },
        }
    }

    pub fn weighted_variance_for_label(
        &self,
        mean:       f64,
        label:      f64,
        labels:  &[f64],
        weights: &[f64],
    ) -> f64
    {
        match self {
            Self::Dense { vals, .. } => {
                let total = weights.iter()
                    .zip(labels)
                    .filter(|&(_, &l)| l == label)
                    .map(|(w, _)| w)
                    .sum::<f64>();
                vals.iter()
                    .zip(weights)
                    .zip(labels)
                    .filter(|&(_, &l)| l == label)
                    .map(|((v, w), _)| w * (v - mean).powi(2))
                    .sum::<f64>()
                    / total
            },
            Self::Sparse { vals, .. } => {
                let total = labels.iter()
                    .zip(weights)
                    .filter(|&(&l, _)| l == label)
                    .map(|(_, &w)| w)
                    .sum::<f64>();

                if total == 0f64 {
                    panic!("weight sum for label y = {label} is zero.");
                }
                let w0 = {
                    let nonzero = vals.iter()
                        .filter(|&&(i, _)| labels[i] == label)
                        .map(|&(i, _)| weights[i])
                        .sum::<f64>();
                    total - nonzero
                };
                let variance = vals.iter()
                    .map(|&(i, v)| weights[i] * (v - mean).powi(2))
                    .sum::<f64>()
                    + w0 * mean.powi(2);
                variance / total
            },
        }
    }

    pub fn weighted_mean_and_variance<T>(&self, weights: T)
        -> (f64, f64)
        where T: AsRef<[f64]>,
    {
        let weights = weights.as_ref();
        let mean = self.weighted_mean(weights);
        let vars = self.weighted_variance(mean, weights);
        (mean, vars)
    }

    pub fn weighted_mean_and_variance_for_label<T>(
        &self,
        label: f64,
        labels:  T,
        weights: T,
    ) -> (f64, f64)
        where T: AsRef<[f64]>,
    {
        let labels  = labels.as_ref();
        let weights = weights.as_ref();
        let mean = self.weighted_mean_for_label(label, labels, weights);
        let vars = self.weighted_variance_for_label(
            mean,
            label,
            labels,
            weights,
        );
        (mean, vars)
    }
}

impl Index<usize> for Feature {
    type Output = f64;
    fn index(&self, idx: usize) -> &Self::Output {
        match self {
            Self::Dense  { vals, .. } => &vals[idx],
            Self::Sparse { vals, .. } => {
                let pos = vals.binary_search_by(|(i, _)| i.cmp(&idx));
                match pos {
                    Ok(p)  => &vals[p].1,
                    Err(_) => &0f64,
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense() {
        let n = "test-001";
        let f = Feature::dense(n);

        if let Feature::Dense { name, vals, } = f {
            assert_eq!(name, n, "expected {n}, got {name}");
            assert!(
                vals.is_empty(),
                "expected 0-length vector, got {vals:?}.",
            );
        } else {
            panic!("expected `Feature::Dense {{ .. }}`, got {f:?}");
        }
    }

    #[test]
    fn test_sparse() {
        let n = "test-002";
        let s = 1_234;
        let f = Feature::sparse(n, s);

        if let Feature::Sparse { name, vals, size, } = f {
            assert_eq!(name, n, "expected {n}, got {name}");
            assert_eq!(size, s, "expected {s}, got {size}");
            assert!(
                vals.is_empty(),
                "expected 0-length vector, got {vals:?}.",
            );
        } else {
            panic!("expected `Feature::Sparse {{ .. }}`, got {f:?}");
        }
    }

    #[test]
    fn test_into_vals_01() {
        let n = "test-003";
        let mut f = Feature::dense(n);
        f.append((  0,   0.7));
        f.append((  8,   0.0));
        f.append((  0,   9.3));
        f.append((100, -30.4));

        let result = f.into_vals();
        let expect = vec![0.7, 0.0, 9.3, -30.4];
        for (r, e) in result.into_iter().zip(expect) {
            assert_eq!(r, e, "expected {e}, got {r}");
        }
    }

    #[test]
    fn test_into_vals_02() {
        let n = "test-004";
        let s = 10;
        let mut f = Feature::sparse(n, s);
        f.append((0,   0.7));
        f.append((8,   0.0));
        f.append((9,   9.3));
        f.append((5, -30.4));
        f.append((3, 300.8));

        let result = f.into_vals();
        let expect = vec![
              0.7, 0.0, 0.0, 300.8, 0.0,
            -30.4, 0.0, 0.0,   0.0, 9.3,
        ];
        assert_eq!(
            result.len(), expect.len(),
            "expected {length}-length vector, got {result:?}",
            length = expect.len(),
        );

        for (r, e) in result.into_iter().zip(expect) {
            assert_eq!(r, e, "expected {e}, got {r}");
        }
    }

    #[test]
    fn test_zero_counts_01() {
        let n = "test-005";
        let mut f = Feature::dense(n);
        f.append((0,   0.7));
        f.append((8,   0.0));
        f.append((9,   9.3));
        f.append((5, -30.4));
        f.append((3, 300.8));

        let result = f.zero_counts();
        let expect = 1;

        assert_eq!(result, expect);
    }

    #[test]
    fn test_zero_counts_02() {
        let n = "test-006";
        let s = 1_000;
        let mut f = Feature::sparse(n, s);
        f.append((0,   0.7));
        f.append((8,   0.0));
        f.append((9,   9.3));
        f.append((5, -30.4));
        f.append((3, 300.8));

        let result = f.zero_counts();
        let expect = 996;

        assert_eq!(result, expect);
    }

    #[test]
    fn test_is_dense_01() {
        let name = "test-007";
        let f = Feature::dense(name);
        assert!(f.is_dense());
    }

    #[test]
    fn test_is_dense_02() {
        let name = "test-008";
        let size = 100;
        let f = Feature::sparse(name, size);
        assert!(!f.is_dense());
    }

    #[test]
    fn test_is_sparse_01() {
        let name = "test-009";
        let f = Feature::dense(name);
        assert!(!f.is_sparse());
    }

    #[test]
    fn test_is_sparse_02() {
        let name = "test-010";
        let size = 100;
        let f = Feature::sparse(name, size);
        assert!(f.is_sparse());
    }

    #[test]
    fn test_name_01() {
        let name = "test-011";
        let f = Feature::dense(name);
        assert_eq!(name, f.name());
    }

    #[test]
    fn test_name_02() {
        let name = "test-012";
        let size = 100;
        let f = Feature::sparse(name, size);
        assert_eq!(name, f.name());
    }
}


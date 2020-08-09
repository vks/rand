//! Implementation of the sampling algorithm in
//!
//! > Feras A. Saad, Cameron E. Freer, Martin C. Rinard, and Vikash K. Mansinghka.
//! > The Fast Loaded Dice Roller: A Near-Optimal Exact Sampler for Discrete
//! > Probability Distributions. In AISTATS 2020: Proceedings of the 23rd
//! > International Conference on Artificial Intelligence and Statistics,
//! > Proceedings of Machine Learning Research 108, Palermo, Sicily, Italy, 2020.
use core::cell::Cell;

use alloc::vec::Vec;
use alloc::vec;

use super::WeightedError;

use crate::Distribution;
use rand::Rng;

fn bit_length(x: i32) -> i32 {
    (32 - x.leading_zeros()) as i32
}

/// Distribution of weighted indices with Fast Loaded Dice Roller method.
#[derive(Debug, Clone)]
pub struct WeightedIndex {
    n: i32, m: i32, k: i32,
    h1: Vec<i32>, h2: Vec<i32>,
    buffer: Cell<u64>, bit: Cell<u8>,
}

impl WeightedIndex {
    /// Preprocess weights.
    pub fn new(weights: Vec<i32>) -> Result<Self, WeightedError> {
        let n = weights.len();
        if n == 0 {
            return Err(WeightedError::NoItem);
        } else if n > ::core::i32::MAX as usize {
            return Err(WeightedError::TooMany);
        }
        let n = n as i32;
        let mut m = 0;
        for &w in &weights {
            if w < 0 {
                return Err(WeightedError::InvalidWeight);
            }
            m += w;
        }
        if m == 0 {
            return Err(WeightedError::AllWeightsZero);
        }
        let k = bit_length(m - 1);
        let r = (1 << k) - m;

        let mut h1 = vec![0; k as usize];
        let mut h2 = vec![-1; ((n + 1) * k) as usize];

        let mut d;
        for j in 0..k {
            d = 0;
            for i in 0..n {
                let w = (weights[i as usize] >> ((k-1) - j)) & 1;
                if w > 0 {
                    h1[j as usize] += 1;
                    h2[(d*k + j) as usize] = i;
                    d += 1;
                }
            }
            let w = (r >> ((k - 1) - j)) & 1;
            if w > 0 {
                h1[j as usize] += 1;
                h2[(d*k + j) as usize] = n;
            }
        }

        Ok(WeightedIndex { n, m, k, h1, h2, buffer: 0.into(), bit: 64.into() })
    }
}

impl Distribution<i32> for WeightedIndex {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> i32 {
        let n = self.n;
        let k = self.k;
        let h1 = &self.h1;
        let h2 = &self.h2;
        let mut c: i32 = 0;
        let mut d: i32 = 0;

        loop {
            if self.bit.get() >= 64 {
                self.buffer.set(rng.next_u64());
                self.bit.set(0);
            }
            let b = ((self.buffer.get() >> self.bit.get()) & 1) as i32;
            d = 2*d + (1 - b);
            if d < h1[c as usize] {
                let z = h2[(d*k + c) as usize];
                if z < n {
                    return z;
                } else {
                    d = 0;
                    c = 0;
                }
            } else {
                d -= h1[c as usize];
                c += 1;
            }
            self.bit.set(self.bit.get() + 1);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::distributions::Uniform;

    #[test]
    fn test_weighted_fldr() {
        const NUM_WEIGHTS: i32 = 10;
        const ZERO_WEIGHT_INDEX: i32 = 3;
        const NUM_SAMPLES: i32 = 15000;
        let mut rng = crate::test::rng(0x9c9fa0b0580a7031);

        let weights = {
            let mut weights = Vec::with_capacity(NUM_WEIGHTS as usize);
            let random_weight_distribution = Uniform::new_inclusive(
                0, NUM_WEIGHTS,
            );
            for _ in 0..NUM_WEIGHTS {
                weights.push(rng.sample(&random_weight_distribution));
            }
            weights[ZERO_WEIGHT_INDEX as usize] = 0;
            weights
        };
        let weight_sum = weights.iter().map(|w| *w).sum::<i32>();
        let expected_counts = weights
            .iter()
            .map(|&w| (w as f64) / (weight_sum as f64) * NUM_SAMPLES as f64)
            .collect::<Vec<f64>>();
        let weight_distribution = WeightedIndex::new(weights).unwrap();

        let mut counts = vec![0; NUM_WEIGHTS as usize];
        for _ in 0..NUM_SAMPLES {
            counts[rng.sample(&weight_distribution) as usize] += 1;
        }

        assert_eq!(counts[ZERO_WEIGHT_INDEX as usize], 0);
        for (count, expected_count) in counts.into_iter().zip(expected_counts) {
            let difference = (count as f64 - expected_count).abs();
            let max_allowed_difference = NUM_SAMPLES as f64 / NUM_WEIGHTS as f64 * 0.1;
            assert!(difference <= max_allowed_difference);
        }

        assert_eq!(
            WeightedIndex::new(vec![]).unwrap_err(),
            WeightedError::NoItem
        );
        assert_eq!(
            WeightedIndex::new(vec![0]).unwrap_err(),
            WeightedError::AllWeightsZero
        );

        // Signed integer special cases
        assert_eq!(
            WeightedIndex::new(vec![-1]).unwrap_err(),
            WeightedError::InvalidWeight
        );
        assert_eq!(
            WeightedIndex::new(vec![core::i32::MIN]).unwrap_err(),
            WeightedError::InvalidWeight
        );
    }
}

//! Implementation of the Fast Dice Roller method for uniformly sampling integers.

use core::cell::Cell;

use rand::Rng;
use rand::distributions::Distribution;
use rand::distributions::uniform::{UniformSampler, SampleBorrow};

/// Helper trait for creating objects using the correct Fast Dice Roller
/// implementation of [`UniformSampler`] for the sampling type.
pub trait SampleFDR: Sized {
    /// The `UniformSampler` implementation supporting type `X`.
    type Sampler: UniformSampler<X = Self>;
}

/// Sample values uniformly between two bounds.
///
/// [`Uniform::new`] and [`Uniform::new_inclusive`] construct a uniform
/// distribution sampling from the given range; these functions may do extra
/// work up front to make sampling of multiple values faster. If only one sample
/// from the range is required, [`Rng::gen_range`] can be more efficient.
///
/// This implementation employs the Fast Dice Roller method, which only consumes
/// as many random bits as required for sampling from the specified range.
#[derive(Clone, Copy, Debug)]
pub struct Uniform<X: SampleFDR>(X::Sampler);

impl<X: SampleFDR> Uniform<X> {
    /// Create a new `Uniform` instance which samples uniformly from the half
    /// open range `[low, high)` (excluding `high`). Panics if `low >= high`.
    pub fn new<B1, B2>(low: B1, high: B2) -> Uniform<X>
    where
        B1: SampleBorrow<X> + Sized,
        B2: SampleBorrow<X> + Sized,
    {
        Uniform(X::Sampler::new(low, high))
    }

    /// Create a new `Uniform` instance which samples uniformly from the closed
    /// range `[low, high]` (inclusive). Panics if `low > high`.
    pub fn new_inclusive<B1, B2>(low: B1, high: B2) -> Uniform<X>
    where
        B1: SampleBorrow<X> + Sized,
        B2: SampleBorrow<X> + Sized,
    {
        Uniform(X::Sampler::new_inclusive(low, high))
    }
}

impl<X: SampleFDR> Distribution<X> for Uniform<X> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> X {
        self.0.sample(rng)
    }
}

/// Uniformly sample an integer with the Fast Dice Roller method.
#[derive(Clone, Debug)]
pub struct UniformInt<X> {
    n: X,
    low: X,
    buffer: Cell<u64>,
    bit: Cell<u64>,
}

macro_rules! uniform_int_impl {
    ($ty:ty, $unsigned:ident, $u_large:ident) => {
        impl SampleFDR for $ty {
            type Sampler = UniformInt<$ty>;
        }

        impl UniformSampler for UniformInt<$ty> {
            // We play free and fast with unsigned vs signed here
            // (when $ty is signed), but that's fine, since the
            // contract of this macro is for $ty and $unsigned to be
            // "bit-equal", so casting between them is a no-op.

            type X = $ty;

            #[inline] // if the range is constant, this helps LLVM to do the
                      // calculations at compile-time.
            fn new<B1, B2>(low_b: B1, high_b: B2) -> Self
            where
                B1: SampleBorrow<Self::X> + Sized,
                B2: SampleBorrow<Self::X> + Sized,
            {
                let low = *low_b.borrow();
                let high = *high_b.borrow();
                Self::new_inclusive(low, high - 1)
            }

            #[inline] // if the range is constant, this helps LLVM to do the
                      // calculations at compile-time.
            fn new_inclusive<B1, B2>(low_b: B1, high_b: B2) -> Self
            where
                B1: SampleBorrow<Self::X> + Sized,
                B2: SampleBorrow<Self::X> + Sized,
            {
                let low = *low_b.borrow();
                let high = *high_b.borrow();
                assert!(low <= high);
                UniformInt {
                    n: high - low,
                    low,
                    buffer: Cell::new(0),
                    bit: Cell::new(0),
                }
            }

            #[inline]
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
                let mut v: Self::X = 1;
                let mut c: Self::X = 0;
                loop {
                    v *= 2;
                    c *= 2;
                    c += ((self.buffer.get() >> self.bit.get()) & 1) as Self::X;
                    if self.bit.get() == 63 {
                        self.bit.set(0);
                        self.buffer.set(rng.next_u64());
                    } else {
                        self.bit.set(self.bit.get() + 1);
                    }
                    if v >= self.n {
                        if c <= self.n {
                            return self.low + c;
                        }
                        v -= self.n;
                        c -= self.n;
                    }
                }
            }
        }
    };
}

uniform_int_impl! { i8, u8, u32 }
uniform_int_impl! { i16, u16, u32 }
uniform_int_impl! { i32, u32, u32 }
uniform_int_impl! { i64, u64, u64 }
#[cfg(not(target_os = "emscripten"))]
uniform_int_impl! { i128, u128, u128 }
uniform_int_impl! { isize, usize, usize }
uniform_int_impl! { u8, u8, u32 }
uniform_int_impl! { u16, u16, u32 }
uniform_int_impl! { u32, u32, u32 }
uniform_int_impl! { u64, u64, u64 }
uniform_int_impl! { usize, usize, usize }
#[cfg(not(target_os = "emscripten"))]
uniform_int_impl! { u128, u128, u128 }

#[cfg(test)]
mod test {
    use super::*;

    #[should_panic]
    #[test]
    fn test_uniform_bad_limits_equal_int() {
        Uniform::new(10, 10);
    }

    #[test]
    fn test_uniform_good_limits_equal_int() {
        let mut rng = crate::test::rng(804);
        let dist = Uniform::new_inclusive(10, 10);
        for _ in 0..20 {
            assert_eq!(rng.sample(&dist), 10);
        }
    }

    #[should_panic]
    #[test]
    fn test_uniform_bad_limits_flipped_int() {
        Uniform::new(10, 5);
    }

    #[test]
    fn test_uniform_simple() {
        let mut rng = crate::test::rng(251);
        let (low, high) = (-10, 10);
        let dist = Uniform::new(low, high);
        for _ in 0..1000 {
            let v: i8 = rng.sample(&dist);
            assert!(low <= v && v < high);
        }
    }

    /*
    #[test]
    #[cfg_attr(miri, ignore)] // Miri is too slow
    fn test_integers() {
        #[cfg(not(target_os = "emscripten"))] use core::{i128, u128};
        use core::{i16, i32, i64, i8, isize};
        use core::{u16, u32, u64, u8, usize};

        let mut rng = crate::test::rng(251);
        macro_rules! t {
            ($ty:ident, $v:expr, $le:expr, $lt:expr) => {{
                for &(low, high) in $v.iter() {
                    let my_uniform = Uniform::new(low, high);
                    for _ in 0..1000 {
                        let v: $ty = rng.sample(&my_uniform);
                        assert!($le(low, v) && $lt(v, high));
                    }

                    let my_uniform = Uniform::new_inclusive(low, high);
                    for _ in 0..1000 {
                        let v: $ty = rng.sample(&my_uniform);
                        assert!($le(low, v) && $le(v, high));
                    }

                    let my_uniform = Uniform::new(&low, high);
                    for _ in 0..1000 {
                        let v: $ty = rng.sample(&my_uniform);
                        assert!($le(low, v) && $lt(v, high));
                    }

                    let my_uniform = Uniform::new_inclusive(&low, &high);
                    for _ in 0..1000 {
                        let v: $ty = rng.sample(&my_uniform);
                        assert!($le(low, v) && $le(v, high));
                    }

                    for _ in 0..1000 {
                        let v = <$ty as SampleFDR>::Sampler::sample_single(low, high, &mut rng);
                        assert!($le(low, v) && $lt(v, high));
                    }
                }
            }};

            // scalar bulk
            ($($ty:ident),*) => {{
                $(t!(
                    $ty,
                    [(0, 10), (10, 127), ($ty::MIN, $ty::MAX)],
                    |x, y| x <= y,
                    |x, y| x < y
                );)*
            }};
        }
        t!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);
        #[cfg(not(target_os = "emscripten"))]
        t!(i128, u128);
    }
    */
}

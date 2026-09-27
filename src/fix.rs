use num_bigint::{BigInt, BigUint};
use num_traits::{ToPrimitive, Zero};

/// Operations assume both operands share the same `prec`.
/// Use [`Fix::with_prec`] to move a value between precisions.
#[derive(Clone, Debug)]
pub struct Fix {
    v: BigInt,
    prec: u64,
}

impl Fix {
    pub fn zero(prec: u64) -> Self {
        Fix {
            v: BigInt::ZERO,
            prec,
        }
    }

    /// Exact conversion: decomposes the f64 into mantissa and binary exponent,
    /// so tiny values (deep-zoom pixel deltas) keep every bit `prec` can hold.
    pub fn from_f64(x: f64, prec: u64) -> Self {
        let bits = x.to_bits();
        let negative = bits >> 63 == 1;
        let biased_exp = ((bits >> 52) & 0x7ff) as i64;
        let frac = bits & ((1u64 << 52) - 1);
        let (mantissa, exp) = if biased_exp == 0 {
            (frac, -1074) // subnormal or zero
        } else {
            (frac | (1u64 << 52), biased_exp - 1075)
        };
        // x = mantissa * 2^exp  =>  v = mantissa * 2^(exp + prec)
        let shift = exp + prec as i64;
        let magnitude = BigInt::from(mantissa);
        let v = if shift >= 0 {
            magnitude << shift
        } else {
            magnitude >> (-shift)
        };
        Fix {
            v: if negative { -v } else { v },
            prec,
        }
    }

    /// Parse a plain decimal (`-1.25`, `0.5`, `3`), rounded to the nearest
    /// `prec`-bit value. Rejects exponents and anything else `to_decimal`
    /// would not produce.
    pub fn from_decimal(s: &str, prec: u64) -> Option<Self> {
        let (negative, s) = match s.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, s),
        };
        let (int_part, frac_part) = s.split_once('.').unwrap_or((s, ""));
        if int_part.is_empty() && frac_part.is_empty() {
            return None;
        }
        let all_digits = |p: &str| p.bytes().all(|b| b.is_ascii_digit());
        if !all_digits(int_part) || !all_digits(frac_part) {
            return None;
        }
        // value = (int * 10^d + frac) / 10^d  =>  v = that << prec
        let digits = format!("{int_part}{frac_part}");
        let scaled = BigUint::parse_bytes(digits.as_bytes(), 10)?;
        let denom = BigUint::from(10u32).pow(frac_part.len() as u32);
        let v = BigInt::from(((scaled << prec) + (&denom >> 1u32)) / denom);
        Some(Fix {
            v: if negative { -v } else { v },
            prec,
        })
    }

    pub fn prec(&self) -> u64 {
        self.prec
    }

    /// Rescale to a new precision (truncating when shrinking).
    pub fn with_prec(&self, prec: u64) -> Self {
        let v = if prec >= self.prec {
            &self.v << (prec - self.prec)
        } else {
            &self.v >> (self.prec - prec)
        };
        Fix { v, prec }
    }

    pub fn mul(&self, other: &Fix) -> Fix {
        Fix {
            v: (&self.v * &other.v) >> self.prec,
            prec: self.prec,
        }
    }

    /// Multiply by a small integer without losing precision.
    pub fn scale(&self, k: i64) -> Fix {
        Fix {
            v: &self.v * k,
            prec: self.prec,
        }
    }

    pub fn add(&self, other: &Fix) -> Fix {
        Fix {
            v: &self.v + &other.v,
            prec: self.prec,
        }
    }

    pub fn sub(&self, other: &Fix) -> Fix {
        Fix {
            v: &self.v - &other.v,
            prec: self.prec,
        }
    }

    pub fn double(&self) -> Fix {
        Fix {
            v: &self.v << 1,
            prec: self.prec,
        }
    }

    pub fn gt(&self, other: &Fix) -> bool {
        self.v > other.v
    }

    /// Nearest f64. Works at any precision as long as the *value* is in f64
    /// range: the integer is first reduced to ~62 significant bits so `2^prec`
    /// never has to be represented as a float.
    pub fn to_f64(&self) -> f64 {
        if self.v.is_zero() {
            return 0.0;
        }
        let drop = self.v.bits() as i64 - 62;
        let reduced = if drop > 0 {
            &self.v >> drop
        } else {
            self.v.clone()
        };
        let exp = drop.max(0) - self.prec as i64;
        reduced.to_f64().expect("62-bit integer fits f64") * 2f64.powi(exp as i32)
    }

    /// Decimal representation with exactly `digits` fractional digits
    /// (truncated, not rounded). Unlike `to_f64` this shows every bit the
    /// fixed-point value holds, which is what deep-zoom coordinates need.
    pub fn to_decimal(&self, digits: usize) -> String {
        let negative = self.v.sign() == num_bigint::Sign::Minus;
        let magnitude = self.v.magnitude();
        let int_part = magnitude >> self.prec;
        let frac_part = magnitude - (&int_part << self.prec);
        let frac_scaled = (frac_part * BigUint::from(10u32).pow(digits as u32)) >> self.prec;
        format!(
            "{}{int_part}.{frac_scaled:0>digits$}",
            if negative { "-" } else { "" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deep-zoom pixel deltas are far below 2^-60; converting them must not
    /// collapse to zero.
    #[test]
    fn from_f64_is_exact_for_tiny_values() {
        for x in [1e-30, -6.25e-22, 3.5, -2.0, 0.0, 1.0 / 3.0] {
            assert_eq!(Fix::from_f64(x, 200).to_f64(), x, "{x}");
        }
    }

    #[test]
    fn with_prec_preserves_value() {
        let x = Fix::from_f64(-0.743_643_887_037_158_7, 64);
        assert_eq!(x.with_prec(300).to_f64(), x.to_f64());
        assert_eq!(x.with_prec(300).with_prec(64).to_f64(), x.to_f64());
    }

    #[test]
    fn to_decimal_shows_full_precision() {
        assert_eq!(Fix::from_f64(-0.75, 64).to_decimal(4), "-0.7500");
        assert_eq!(Fix::from_f64(2.5, 64).to_decimal(1), "2.5");
        // 1e-30 is far below f64-printable precision relative to the integer part.
        let tiny = Fix::from_f64(1e-30, 200).add(&Fix::from_f64(1.0, 200));
        let s = tiny.to_decimal(32);
        assert!(s.starts_with("1.000000000000000000000000000001"), "{s}");
    }

    #[test]
    fn from_decimal_is_nearest_representable() {
        // 38 decimals ≈ 126 bits: the status-line coordinates of a deep zoom.
        let s = "-1.24990508287924271184980842559554417152";
        let prec = 200;
        let x = Fix::from_decimal(s, prec).unwrap();
        // |v * 10^38 - k * 2^prec| <= 10^38 / 2  <=>  |v/2^prec - k/10^38| <= 2^-(prec+1)
        let k: BigInt = -s.replace(['-', '.'], "").parse::<BigInt>().unwrap();
        let denom = BigInt::from(10).pow(38);
        let err = &x.v * &denom - (k << prec);
        assert!(err.magnitude() * 2u32 <= *denom.magnitude(), "{err}");
        assert_eq!(x.to_decimal(38), s);

        assert_eq!(Fix::from_decimal("-0.75", 64).unwrap().to_f64(), -0.75);
        assert_eq!(Fix::from_decimal("3", 64).unwrap().to_f64(), 3.0);
        assert_eq!(Fix::from_decimal(".5", 64).unwrap().to_f64(), 0.5);
        for bad in ["", "-", ".", "1e5", "0x1", "1.2.3", "abc"] {
            assert!(Fix::from_decimal(bad, 64).is_none(), "{bad:?}");
        }
    }

    #[test]
    fn arithmetic_matches_f64() {
        let a = Fix::from_f64(1.5, 128);
        let b = Fix::from_f64(-0.25, 128);
        assert_eq!(a.mul(&b).to_f64(), -0.375);
        assert_eq!(a.add(&b).to_f64(), 1.25);
        assert_eq!(a.sub(&b).to_f64(), 1.75);
        assert_eq!(a.double().to_f64(), 3.0);
        assert_eq!(b.scale(-7).to_f64(), 1.75);
        assert!(a.gt(&b) && !b.gt(&a));
    }
}

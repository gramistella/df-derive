//! Self-tests for the contract harness.
//!
//! Two axes are exercised:
//!
//! 1. **Correct backend passes.** A faithful round-half-to-even
//!    `try_to_i128_mantissa` (the same algorithm the in-tree
//!    `rust_decimal::Decimal` impl uses) must satisfy the harness on every
//!    case in the battery.
//! 2. **Broken backends fail loudly.** Three deliberately wrong backends
//!    are checked, each violating exactly one part of the contract:
//!    truncation (drops every half-tie down), half-away-from-zero (rounds
//!    every half-tie outward), and overflow-hiding (wraps scale-up
//!    overflow instead of returning `None`). Each one must trigger
//!    `assert_decimal128_encode_contract` to panic with a message naming
//!    the divergent values — `#[should_panic(expected = ...)]` enforces
//!    that the panic carries enough context to debug the failure.
//!
//! All four backends operate on the literal string the harness feeds them
//! (no `From<&str>` bound on a backend type required), which is why the
//! harness uses the closure form rather than a trait bound.

use df_derive_test_harness::assert_decimal128_encode_contract;
use rust_decimal::Decimal;
use std::str::FromStr as _;

// ---- Reference: round-half-to-even ----
//
// This is byte-equivalent to the `Decimal128Encode for rust_decimal::Decimal`
// impl that ships in `tests/common.rs` of the parent crate. Reproduced here
// so the harness self-test does not depend on the parent's test module.
fn rust_decimal_correct(value: &Decimal, target_scale: u32) -> Option<i128> {
    let source_scale = value.scale();
    let mantissa: i128 = value.mantissa();
    if source_scale == target_scale {
        return Some(mantissa);
    }
    if source_scale < target_scale {
        let diff = target_scale - source_scale;
        let pow = 10i128.checked_pow(diff)?;
        return mantissa.checked_mul(pow);
    }
    let diff = source_scale - target_scale;
    let pow = 10i128.pow(diff).cast_unsigned();
    let neg = mantissa < 0;
    let abs = mantissa.unsigned_abs();
    let q = (abs / pow).cast_signed();
    let r = abs % pow;
    let half = pow / 2;
    let rounded = match r.cmp(&half) {
        std::cmp::Ordering::Greater => q + 1,
        std::cmp::Ordering::Less => q,
        std::cmp::Ordering::Equal => q + (q & 1),
    };
    Some(if neg { -rounded } else { rounded })
}

#[test]
fn correct_rust_decimal_backend_passes() {
    assert_decimal128_encode_contract(|literal, target_scale| {
        let value = Decimal::from_str(literal).unwrap_or_else(|err| {
            panic!("rust_decimal could not parse `{literal}`: {err}");
        });
        rust_decimal_correct(&value, target_scale)
    });
}

// ---- Broken: truncation (always rounds toward zero) ----
//
// Triggers on every half-tie case: e.g., `9.8755 -> scale 3` should produce
// 9876 (half-to-even via odd quotient), but truncation produces 9875.
fn rust_decimal_truncating(value: &Decimal, target_scale: u32) -> Option<i128> {
    let source_scale = value.scale();
    let mantissa: i128 = value.mantissa();
    if source_scale == target_scale {
        return Some(mantissa);
    }
    if source_scale < target_scale {
        let diff = target_scale - source_scale;
        let pow = 10i128.checked_pow(diff)?;
        return mantissa.checked_mul(pow);
    }
    let diff = source_scale - target_scale;
    let pow = 10i128.pow(diff);
    Some(mantissa / pow) // integer division truncates toward zero
}

#[test]
#[should_panic(expected = "Decimal128Encode contract violation")]
fn truncating_backend_panics() {
    assert_decimal128_encode_contract(|literal, target_scale| {
        let value = Decimal::from_str(literal).unwrap();
        rust_decimal_truncating(&value, target_scale)
    });
}

// ---- Broken: half-away-from-zero (rust_decimal's native `rescale`) ----
//
// `rust_decimal::Decimal::rescale` rounds half-away-from-zero. The harness
// must detect this on a half-tie where the magnitude rounds up rather than
// to even — e.g., `9.8745 -> scale 3` should stay at 9874 (q=9874 even),
// but rescale produces 9875.
fn rust_decimal_half_away_from_zero(value: &Decimal, target_scale: u32) -> Option<i128> {
    let mut working = *value;
    working.rescale(target_scale);
    if working.scale() != target_scale {
        // Could happen on overflow during rescale; surface as None to
        // isolate the panic to the rounding axis only.
        return None;
    }
    Some(working.mantissa())
}

#[test]
#[should_panic(expected = "Decimal128Encode contract violation")]
fn half_away_from_zero_backend_panics() {
    assert_decimal128_encode_contract(|literal, target_scale| {
        let value = Decimal::from_str(literal).unwrap();
        rust_decimal_half_away_from_zero(&value, target_scale)
    });
}

// ---- Broken: overflow-hider ----
//
// Correct on rounding, but uses wrapping arithmetic on scale-up. Must be
// caught by the OVERFLOW_CASES section of the harness.
fn rust_decimal_overflow_hider(value: &Decimal, target_scale: u32) -> Option<i128> {
    let source_scale = value.scale();
    let mantissa: i128 = value.mantissa();
    if source_scale == target_scale {
        return Some(mantissa);
    }
    if source_scale < target_scale {
        let diff = target_scale - source_scale;
        // Wrapping power and wrapping multiply hide the overflow that the
        // contract demands surface as `None`.
        let pow = 10i128.wrapping_pow(diff);
        return Some(mantissa.wrapping_mul(pow));
    }
    let diff = source_scale - target_scale;
    let pow = 10i128.pow(diff).cast_unsigned();
    let neg = mantissa < 0;
    let abs = mantissa.unsigned_abs();
    let q = (abs / pow).cast_signed();
    let r = abs % pow;
    let half = pow / 2;
    let rounded = match r.cmp(&half) {
        std::cmp::Ordering::Greater => q + 1,
        std::cmp::Ordering::Less => q,
        std::cmp::Ordering::Equal => q + (q & 1),
    };
    Some(if neg { -rounded } else { rounded })
}

#[test]
#[should_panic(expected = "the contract requires `None`")]
fn overflow_hiding_backend_panics() {
    assert_decimal128_encode_contract(|literal, target_scale| {
        let value = Decimal::from_str(literal).unwrap();
        rust_decimal_overflow_hider(&value, target_scale)
    });
}

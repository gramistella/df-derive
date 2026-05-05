//! Contract test harness for [`df-derive`]'s `Decimal128Encode` trait.
//!
//! # Why this crate exists
//!
//! `df-derive` lowers `Decimal` fields straight to a polars
//! `Decimal(precision, scale)` column by going through a user-pluggable
//! trait whose contract is documented next to the derive:
//!
//! ```ignore
//! pub trait Decimal128Encode {
//!     fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128>;
//! }
//! ```
//!
//! The contract is **load-bearing**: implementations MUST round
//! half-to-even (banker's rounding) on scale-down. Polars' own
//! `str_to_dec128` rounds that way, so a backend that disagrees on
//! tie-breaking (for instance `rust_decimal::Decimal::rescale`, which rounds
//! half-away-from-zero) silently produces a column whose bytes diverge from
//! the historical `to_string + cast` path the codegen replaced.
//!
//! A backend that gets this wrong still compiles, still runs, and still
//! produces a `Decimal(p, s)` column. Nothing in the type system catches
//! it. The only way the bug surfaces is byte-level mismatch in downstream
//! pipelines — which is exactly the failure mode the codegen is designed to
//! avoid.
//!
//! This harness exists to catch that bug class at the backend's own test
//! suite, before the broken impl reaches a consumer. It cross-checks each
//! candidate `try_to_i128_mantissa` against polars's `str_to_dec128` on a
//! battery of inputs that exercises:
//!
//! - positive and negative half-tie boundaries (the round-half-to-even
//!   axis),
//! - non-tie scale-down (the basic floor/ceil axis),
//! - same-scale and scale-up paths (where the mantissa is forwarded
//!   unchanged or multiplied by a power of ten),
//! - very large magnitudes near `i128::MAX` (where naive arithmetic
//!   overflows),
//! - scale-up overflow (where the contract requires `None`).
//!
//! # Relationship to the encoder IR invariants
//!
//! See `docs/encoder-ir.md` in the `df-derive` repository for the broader
//! context. The `Decimal` leaf in the encoder IR commits to a single
//! rounding rule for cross-backend byte-equivalence; this harness is the
//! external surface that lets backend authors prove their impl honours
//! that commitment.
//!
//! # Usage
//!
//! Call [`assert_decimal128_encode_contract`] from a `#[test]` in your
//! backend crate, passing a closure that parses a decimal literal string
//! and rescales it to `target_scale` via your `Decimal128Encode` impl:
//!
//! ```ignore
//! use df_derive_test_harness::assert_decimal128_encode_contract;
//! use my_decimal::MyDecimal;
//! use std::str::FromStr as _;
//!
//! #[test]
//! fn my_decimal_matches_polars_rounding() {
//!     assert_decimal128_encode_contract(|literal, target_scale| {
//!         let value = MyDecimal::from_str(literal).unwrap();
//!         <MyDecimal as my_runtime::Decimal128Encode>::try_to_i128_mantissa(
//!             &value,
//!             target_scale,
//!         )
//!     });
//! }
//! ```
//!
//! The closure form sidesteps any trait-bound dance: the harness does not
//! need to import your `Decimal128Encode` trait, and you do not need a
//! `From<&str>` adapter on your decimal type. If the closure ever returns
//! a mantissa that disagrees with polars's `str_to_dec128`, the harness
//! panics with both values, the input literal, and the target
//! `(precision, scale)`, so the failure points directly at the rounding
//! direction that diverged.
//!
//! [`df-derive`]: https://docs.rs/df-derive

use polars::prelude::*;

/// One row of the contract battery: an input decimal literal, the target
/// `(precision, scale)` to rescale into, and an English-language note that
/// will appear in the panic message when the case fails.
#[derive(Debug, Clone, Copy)]
struct ContractCase {
    literal: &'static str,
    precision: usize,
    scale: usize,
    /// Why this case is in the battery — surfaces in panic messages so the
    /// failure mode (e.g. "negative half-tie") is obvious without reading
    /// the harness source.
    note: &'static str,
}

/// Cases where the user's impl must agree with polars's `str_to_dec128` on
/// the resulting i128 mantissa.
///
/// Each case is hit by [`assert_decimal128_encode_contract`]: the harness
/// computes polars's mantissa via `Series::cast(&DataType::Decimal(p, s))`
/// (which routes through `str_to_dec128`) and the user's mantissa via the
/// supplied closure, then asserts byte-equality.
const AGREEMENT_CASES: &[ContractCase] = &[
    // ---- Same-scale forward ----
    ContractCase {
        literal: "0",
        precision: 38,
        scale: 0,
        note: "zero, same scale",
    },
    ContractCase {
        literal: "0.000000",
        precision: 18,
        scale: 6,
        note: "zero, same scale (non-trivial)",
    },
    ContractCase {
        literal: "0.000042",
        precision: 18,
        scale: 6,
        note: "small positive, same scale",
    },
    ContractCase {
        literal: "-0.000042",
        precision: 18,
        scale: 6,
        note: "small negative, same scale",
    },
    // ---- Scale-up (multiply mantissa by 10^diff) ----
    ContractCase {
        literal: "123.45",
        precision: 12,
        scale: 4,
        note: "scale-up by 100",
    },
    ContractCase {
        literal: "-123.45",
        precision: 12,
        scale: 4,
        note: "scale-up by 100, negative",
    },
    ContractCase {
        literal: "1",
        precision: 38,
        scale: 10,
        note: "integer scale-up by 10^10",
    },
    // ---- Scale-down, no tie (basic round) ----
    ContractCase {
        literal: "1.23456",
        precision: 12,
        scale: 3,
        note: "scale-down, remainder > half",
    },
    ContractCase {
        literal: "1.23451",
        precision: 12,
        scale: 3,
        note: "scale-down, remainder < half",
    },
    ContractCase {
        literal: "-1.23456",
        precision: 12,
        scale: 3,
        note: "scale-down negative, remainder > half (toward 0 by magnitude up)",
    },
    ContractCase {
        literal: "-1.23451",
        precision: 12,
        scale: 3,
        note: "scale-down negative, remainder < half",
    },
    // ---- Scale-down half-tie boundaries (the round-half-to-even axis) ----
    ContractCase {
        literal: "9.8755",
        precision: 10,
        scale: 3,
        note: "half-tie, q=9875 odd -> +1 (even 9876)",
    },
    ContractCase {
        literal: "9.8745",
        precision: 10,
        scale: 3,
        note: "half-tie, q=9874 even -> stay (even 9874)",
    },
    ContractCase {
        literal: "9.8735",
        precision: 10,
        scale: 3,
        note: "half-tie, q=9873 odd -> +1 (even 9874)",
    },
    ContractCase {
        literal: "9.8765",
        precision: 10,
        scale: 2,
        note: "non-tie magnitude, scale-down by 100, rounds to 988",
    },
    ContractCase {
        literal: "-9.8765",
        precision: 10,
        scale: 2,
        note: "non-tie negative, rounds to -988",
    },
    ContractCase {
        literal: "-9.8755",
        precision: 10,
        scale: 3,
        note: "negative half-tie, q=9875 odd -> +1 (even -9876)",
    },
    ContractCase {
        literal: "-9.8745",
        precision: 10,
        scale: 3,
        note: "negative half-tie, q=9874 even -> stay (even -9874)",
    },
    // ---- Tiny scale-down to zero ----
    ContractCase {
        literal: "0.0000000000000000000000000001",
        precision: 38,
        scale: 6,
        note: "tiny value, scales down to 0",
    },
    // ---- Very large magnitudes (must not overflow internally) ----
    ContractCase {
        literal: "99999999999999999999999999.99",
        precision: 38,
        scale: 0,
        note: "near-i128::MAX, scale-down by 100",
    },
    ContractCase {
        literal: "-99999999999999999999999999.99",
        precision: 38,
        scale: 0,
        note: "near-i128::MIN, scale-down by 100",
    },
];

/// Polars caps decimal precision at 38, which means the magnitude after
/// rescaling has to fit in 38 decimal digits (`< 10^38 < 2^127`). A
/// scale-up that would land at `10^38` digits or more must therefore
/// surface as `None` from `try_to_i128_mantissa` rather than wrapping or
/// returning a meaningless mantissa. This mirrors the historical
/// `to_string + cast` overflow path that the codegen replaced.
const OVERFLOW_CASES: &[ContractCase] = &[
    // 10^28 + a residue, scale-up by 10^11 -> demands ~39 digits, must overflow.
    ContractCase {
        literal: "10000000000000000000000000000",
        precision: 38,
        scale: 11,
        note: "scale-up overflow: 10^28 * 10^11 > i128::MAX",
    },
    ContractCase {
        literal: "-10000000000000000000000000000",
        precision: 38,
        scale: 11,
        note: "scale-up overflow, negative",
    },
];

/// Compute the mantissa polars itself would write when given the literal
/// `s` cast into `Decimal(precision, scale)`. This is the byte-level ground
/// truth the harness measures backends against; it is the same path the
/// pre-direct-mantissa codegen used (`to_string + cast`).
fn polars_str_path_mantissa(s: &str, precision: usize, scale: usize) -> i128 {
    let cast = Series::new("x".into(), &[s.to_string()])
        .cast(&DataType::Decimal(precision, scale))
        .unwrap_or_else(|err| {
            panic!(
                "polars rejected the str cast for `{s}` -> Decimal({precision}, {scale}): {err}",
            )
        });
    match cast.get(0).expect("polars produced an empty series") {
        AnyValue::Decimal(v, _p, _s) => v,
        AnyValue::Null => panic!(
            "polars produced a NULL when casting `{s}` to Decimal({precision}, {scale}); the \
             harness does not test null-bearing inputs",
        ),
        other => panic!(
            "polars produced an unexpected AnyValue when casting `{s}` to \
             Decimal({precision}, {scale}): {other:?}",
        ),
    }
}

/// Run the cross-check battery against `encoder` and panic with a
/// diagnostic message on the first divergence.
///
/// `encoder` is the user's adapter: given a decimal literal string and a
/// target scale, it parses the literal into the backend's decimal type and
/// returns `<Backend as Decimal128Encode>::try_to_i128_mantissa(target_scale)`.
/// Most backends will write something like:
///
/// ```ignore
/// |literal, scale| MyDecimal::from_str(literal).unwrap().try_to_i128_mantissa(scale)
/// ```
///
/// # What the harness checks
///
/// 1. **Agreement.** For each input in the agreement battery, the
///    mantissa returned by `encoder` must equal the mantissa polars writes
///    when casting the same literal string through
///    `DataType::Decimal(p, s)`. Coverage: same-scale, scale-up,
///    scale-down with non-tie remainders, scale-down at half-tie
///    boundaries (positive and negative, both parities of the truncated
///    quotient), and very-large magnitudes near `i128::MAX`.
/// 2. **Overflow.** For inputs whose scale-up would not fit in `i128`,
///    `encoder` must return `None`. Wrapping or producing a garbage
///    mantissa is a contract violation.
///
/// # Failure modes
///
/// Each panic message names the input literal, the target precision and
/// scale, and the divergent values, so the failure is traceable to a
/// specific rounding direction:
///
/// - "user impl returned `Some(987)` but polars wrote `988`": user is
///   rounding half-to-even incorrectly (probably truncating, or rounding
///   half-away-from-zero on a negative magnitude).
/// - "user impl returned `Some(_)` but the contract requires `None`":
///   user is wrapping or otherwise hiding a scale-up overflow.
///
/// # Panics
///
/// Panics on the first contract violation. Subsequent cases are not
/// checked — fix one violation and rerun the test to surface the next.
pub fn assert_decimal128_encode_contract<F>(mut encoder: F)
where
    F: FnMut(&str, u32) -> Option<i128>,
{
    for case in AGREEMENT_CASES {
        let polars_mantissa = polars_str_path_mantissa(case.literal, case.precision, case.scale);
        let user_mantissa = encoder(
            case.literal,
            u32::try_from(case.scale).expect("scale must fit in u32 (polars caps it at 38)"),
        );
        let Some(user_mantissa) = user_mantissa else {
            panic!(
                "Decimal128Encode contract violation: user impl returned `None` but polars \
                 produced `{polars_mantissa}` for input `{literal}` -> Decimal({precision}, \
                 {scale}). Case: {note}.",
                literal = case.literal,
                precision = case.precision,
                scale = case.scale,
                note = case.note,
            );
        };
        assert_eq!(
            user_mantissa, polars_mantissa,
            "Decimal128Encode contract violation: user impl returned `{user_mantissa}` but \
             polars wrote `{polars_mantissa}` for input `{literal}` -> Decimal({precision}, \
             {scale}). Case: {note}. The contract requires round-half-to-even (banker's \
             rounding) on scale-down — see the README's \"Custom decimal backends\" section.",
            literal = case.literal,
            precision = case.precision,
            scale = case.scale,
            note = case.note,
        );
    }

    for case in OVERFLOW_CASES {
        let user_mantissa = encoder(
            case.literal,
            u32::try_from(case.scale).expect("scale must fit in u32 (polars caps it at 38)"),
        );
        assert!(
            user_mantissa.is_none(),
            "Decimal128Encode contract violation: user impl returned `{user_mantissa:?}` but \
             the contract requires `None` for input `{literal}` -> Decimal({precision}, \
             {scale}). Case: {note}. Scale-up that would not fit in i128 must surface as `None` \
             so the codegen can map it to a polars `ComputeError`.",
            literal = case.literal,
            precision = case.precision,
            scale = case.scale,
            note = case.note,
        );
    }
}

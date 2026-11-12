#![cfg(feature = "rust_decimal")]

use df_derive_core::dataframe::builders::DecimalBuilder;
use df_derive_core::dataframe::{ColumnBuilder, Decimal128Encode as _};
use polars::prelude::*;
use rust_decimal::Decimal;
use std::str::FromStr as _;

#[derive(Debug, Clone, Copy)]
struct Case {
    literal: &'static str,
    precision: usize,
    scale: usize,
}

const AGREEMENT_CASES: &[Case] = &[
    Case {
        literal: "0",
        precision: 38,
        scale: 0,
    },
    Case {
        literal: "0.000000",
        precision: 18,
        scale: 6,
    },
    Case {
        literal: "123.45",
        precision: 12,
        scale: 4,
    },
    Case {
        literal: "-123.45",
        precision: 12,
        scale: 4,
    },
    Case {
        literal: "1.23456",
        precision: 12,
        scale: 3,
    },
    Case {
        literal: "1.23451",
        precision: 12,
        scale: 3,
    },
    Case {
        literal: "-1.23456",
        precision: 12,
        scale: 3,
    },
    Case {
        literal: "-1.23451",
        precision: 12,
        scale: 3,
    },
    Case {
        literal: "9.8755",
        precision: 10,
        scale: 3,
    },
    Case {
        literal: "9.8745",
        precision: 10,
        scale: 3,
    },
    Case {
        literal: "9.8735",
        precision: 10,
        scale: 3,
    },
    Case {
        literal: "-9.8755",
        precision: 10,
        scale: 3,
    },
    Case {
        literal: "-9.8745",
        precision: 10,
        scale: 3,
    },
    Case {
        literal: "0.0000000000000000000000000001",
        precision: 38,
        scale: 6,
    },
    Case {
        literal: "99999999999999999999999999.99",
        precision: 38,
        scale: 0,
    },
    Case {
        literal: "-99999999999999999999999999.99",
        precision: 38,
        scale: 0,
    },
];

const OVERFLOW_CASES: &[Case] = &[
    Case {
        literal: "10000000000000000000000000000",
        precision: 38,
        scale: 11,
    },
    Case {
        literal: "-10000000000000000000000000000",
        precision: 38,
        scale: 11,
    },
];

fn polars_str_path_mantissa(s: &str, precision: usize, scale: usize) -> i128 {
    let cast = Series::new("x".into(), &[s.to_string()])
        .cast(&DataType::Decimal(precision, scale))
        .unwrap_or_else(|err| {
            panic!(
                "polars rejected `{s}` -> Decimal({precision}, {scale}) in contract test: {err}",
            )
        });
    match cast.get(0).expect("polars produced an empty series") {
        AnyValue::Decimal(v, _p, _s) => v,
        other => panic!(
            "polars produced unexpected value for `{s}` -> Decimal({precision}, {scale}): {other:?}",
        ),
    }
}

#[test]
fn rust_decimal_reference_impl_matches_polars_decimal_cast() {
    for case in AGREEMENT_CASES {
        let value = Decimal::from_str(case.literal)
            .unwrap_or_else(|err| panic!("rust_decimal could not parse `{}`: {err}", case.literal));
        let actual = value
            .try_to_i128_mantissa(case.scale.try_into().expect("scale fits in u32"))
            .unwrap_or_else(|| {
                panic!(
                    "Decimal128Encode returned None for `{}` -> Decimal({}, {})",
                    case.literal, case.precision, case.scale,
                )
            });
        let expected = polars_str_path_mantissa(case.literal, case.precision, case.scale);
        assert_eq!(
            actual, expected,
            "mantissa mismatch for `{}` -> Decimal({}, {})",
            case.literal, case.precision, case.scale,
        );
    }
}

#[test]
fn rust_decimal_reference_impl_returns_none_on_scale_up_overflow() {
    for case in OVERFLOW_CASES {
        let value = Decimal::from_str(case.literal)
            .unwrap_or_else(|err| panic!("rust_decimal could not parse `{}`: {err}", case.literal));
        assert!(
            value
                .try_to_i128_mantissa(case.scale.try_into().expect("scale fits in u32"))
                .is_none(),
            "expected scale-up overflow for `{}` -> Decimal({}, {})",
            case.literal,
            case.precision,
            case.scale,
        );
    }
}

#[test]
fn rust_decimal_reference_impl_returns_none_for_invalid_target_scale() {
    let value = Decimal::from_str("1").expect("literal decimal parses");

    assert!(value.try_to_i128_mantissa(39).is_none());
    assert!(value.try_to_i128_mantissa(u32::MAX).is_none());
}

struct CustomReferenceBackend(i128);

impl df_derive_core::dataframe::Decimal128Encode for CustomReferenceBackend {
    fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128> {
        Some(self.0 + i128::from(target_scale))
    }
}

#[test]
fn decimal128_encode_delegates_through_references() {
    let value = CustomReferenceBackend(100);
    let by_ref = &value;
    let by_ref_ref = &&value;

    assert_eq!(by_ref.try_to_i128_mantissa(2), Some(102));
    assert_eq!(by_ref_ref.try_to_i128_mantissa(3), Some(103));
}

/// What Polars itself stores for `literal` cast to `Decimal(38, scale)`:
/// the mantissa, or `None` when the value does not fit in 38 digits.
fn polars_mantissa_at_full_precision(literal: &str, scale: usize) -> Option<i128> {
    let cast = Series::new("x".into(), &[literal.to_string()])
        .cast(&DataType::Decimal(38, scale))
        .ok()?;
    match cast.get(0).expect("polars produced an empty series") {
        AnyValue::Decimal(mantissa, _, _) => Some(mantissa),
        AnyValue::Null => None,
        other => panic!("unexpected value for `{literal}` at scale {scale}: {other:?}"),
    }
}

/// What a `Decimal(38, S)` column built by df-derive stores for `value`, or
/// `None` when the builder rejects it.
fn stored_at_full_precision<const S: usize>(value: &Decimal) -> Option<i128> {
    let mut builder = <DecimalBuilder<38, S> as ColumnBuilder<Decimal>>::new();
    builder.push(Some(value)).ok()?;
    let mut columns = Vec::new();
    ColumnBuilder::<Decimal>::finish(builder, "x".into(), None, &mut columns)
        .expect("a one-value decimal column assembles");
    match columns[0].get(0).expect("the column holds one value") {
        AnyValue::Decimal(mantissa, _, _) => Some(mantissa),
        other => panic!("unexpected value in a decimal column: {other:?}"),
    }
}

#[test]
fn decimal_columns_agree_with_polars_at_the_38_digit_limit() {
    // A Polars decimal holds at most 38 digits, so a mantissa of 10^38 or
    // more cannot be stored at any precision even though it fits in `i128`.
    // The trait leaves that to the builder, which knows the precision. Each
    // case sits on one side of the limit.
    let at_scale_38 = [
        "0.1",  // 10^37: fits
        "-0.1", // -10^37: fits
        "0.9999999999999999999999999999",
        "1",   // 10^38: one digit too many
        "-1",  // -10^38
        "1.7", // fits in `i128`, not in 38 digits
    ];
    for literal in at_scale_38 {
        let value = Decimal::from_str(literal).expect("literal decimal parses");
        assert_eq!(
            stored_at_full_precision::<38>(&value),
            polars_mantissa_at_full_precision(literal, 38),
            "`{literal}` at scale 38: df-derive and Polars disagree",
        );
    }
    for literal in ["9.9999", "10", "-10", "1"] {
        let value = Decimal::from_str(literal).expect("literal decimal parses");
        assert_eq!(
            stored_at_full_precision::<37>(&value),
            polars_mantissa_at_full_precision(literal, 37),
            "`{literal}` at scale 37: df-derive and Polars disagree",
        );
    }

    // The value the limit exists for: it has an `i128` mantissa, so the
    // trait returns it, and the builder is what turns it away.
    let one = Decimal::from_str("1").expect("literal decimal parses");
    assert_eq!(one.try_to_i128_mantissa(38), Some(10i128.pow(38)));
    assert_eq!(stored_at_full_precision::<38>(&one), None);
}

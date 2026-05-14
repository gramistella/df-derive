#![cfg(feature = "rust_decimal")]

use df_derive_core::dataframe::Decimal128Encode as _;
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

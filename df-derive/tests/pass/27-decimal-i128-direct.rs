// Lock in the direct `Decimal::mantissa() -> i128` columnar path.
//
// The codegen route for `Decimal` fields used to materialize each row as a
// `String` and cast the column to `Decimal(p, s)` at the finisher. The new
// path materializes `i128` mantissa values rescaled to the schema scale and
// hands them to `Int128Chunked::into_decimal_unchecked(p, s)`. The original
// `to_string + parse` round-trip went through polars's `str_to_dec128`, which
// rounds half-to-even (banker's rounding) on scale-down. The rescale
// arithmetic baked into `map_primitive_expr` reproduces that semantics
// directly on the i128 mantissa, so observable values stay identical to the
// historical path even at half-tie boundaries.
//
// We assert mantissa-level equality on the cells rather than re-rendering as
// strings, because polars' `AnyValue::Decimal(mantissa, p, s)` exposes the
// underlying `i128` directly — that's the value our codegen now writes. The
// banker's-rounding edge cases at the bottom of `main` exercise the four
// possible half-tie outcomes (round up to even, stay at even, round up to
// even after a non-zero remainder, negative magnitude) so that drifting back
// to truncation or to half-away-from-zero would fail loudly.

use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};

use polars::prelude::*;
use pretty_assertions::assert_eq;
use rust_decimal::Decimal;

#[derive(ToDataFrame, Clone)]
struct Row {
    // Schema scale (4) > rust_decimal scale (2): rescale must scale-up by 100.
    #[df_derive(decimal(precision = 12, scale = 4))]
    price: Decimal,
    // Schema scale (2) < rust_decimal scale (4): scale-down with banker's
    // rounding. `9.8765` rounds to `9.88` (the trailing `65` exceeds half so
    // the magnitude rounds up regardless of parity).
    #[df_derive(decimal(precision = 10, scale = 2))]
    fee: Decimal,
    // Optional column exercises the `Vec<Option<i128>>` finisher branch.
    #[df_derive(decimal(precision = 18, scale = 6))]
    bonus: Option<Decimal>,
    // Same-scale path: schema scale matches the value's scale, so the
    // mantissa is forwarded unchanged.
    #[df_derive(decimal(precision = 18, scale = 6))]
    direct: Decimal,
    // List exercises the `Vec<i128>` → `Series::new` → `cast(Decimal)` inner
    // path used by `Vec<Decimal>` shapes.
    #[df_derive(decimal(precision = 14, scale = 3))]
    history: Vec<Decimal>,
    // List with optional values exercises `Vec<Option<i128>>` inner path.
    #[df_derive(decimal(precision = 14, scale = 3))]
    nullable_history: Vec<Option<Decimal>>,
}

#[derive(ToDataFrame, Clone)]
struct EdgeRow {
    // Half-tie at scale 3 → 9876 (q=9875 odd ⇒ +1 ⇒ even 9876).
    #[df_derive(decimal(precision = 10, scale = 3))]
    half_to_even_up: Decimal,
    // Half-tie at scale 3 → 9874 (q=9874 even ⇒ stays 9874).
    #[df_derive(decimal(precision = 10, scale = 3))]
    half_to_even_stay: Decimal,
    // Half-tie at scale 3 → 9874 (q=9873 odd ⇒ +1 ⇒ even 9874).
    #[df_derive(decimal(precision = 10, scale = 3))]
    half_to_even_up_low: Decimal,
    // Negative magnitude with banker's rounding: -9.8765 → scale 2 → -988.
    #[df_derive(decimal(precision = 10, scale = 2))]
    negative_round_up: Decimal,
    // Negative half-tie magnitude: -9.8755 → scale 3 → -9876.
    #[df_derive(decimal(precision = 10, scale = 3))]
    negative_half_to_even: Decimal,
}

fn decimal_mantissa(av: AnyValue<'_>) -> Option<i128> {
    match av {
        AnyValue::Decimal(v, _p, _s) => Some(v),
        AnyValue::Null => None,
        other => panic!("expected AnyValue::Decimal or Null, got {other:?}"),
    }
}

/// Cross-check: produce the i128 polars itself would write when given the
/// `Decimal::to_string()` representation and the same target precision/scale.
/// If our direct-mantissa rescale ever diverges from this, the test fails
/// with the literal mantissa values from each path.
fn polars_str_path_mantissa(d: Decimal, precision: usize, scale: usize) -> i128 {
    let s = Series::new("x".into(), &[d.to_string()])
        .cast(&DataType::Decimal(precision, scale))
        .expect("polars rejected the cast");
    decimal_mantissa(s.get(0).expect("empty Series")).expect("expected non-null cell")
}

fn main() {
    let row = Row {
        price: Decimal::new(12345, 2),                            // 123.45 → scale 4 → 1234500
        fee: Decimal::new(98765, 4),                              // 9.8765 → scale 2 → 988
        bonus: Some(Decimal::new(1, 18)),                         // 0.000000000000000001 → scale 6 → 0
        direct: Decimal::new(42, 6),                              // 0.000042 → scale 6 → 42
        history: vec![Decimal::new(1, 3), Decimal::new(7000, 6)], // [0.001, 0.007] → scale 3 → [1, 7]
        nullable_history: vec![Some(Decimal::new(2, 3)), None, Some(Decimal::new(0, 0))],
    };

    // Single-row materialization (delegates to `Columnar::columnar_from_refs`).
    let df = row.to_dataframe().unwrap();
    assert_eq!(df.height(), 1);

    assert_eq!(df.column("price").unwrap().dtype(), &DataType::Decimal(12, 4));
    assert_eq!(decimal_mantissa(df.column("price").unwrap().get(0).unwrap()), Some(1_234_500));

    assert_eq!(df.column("fee").unwrap().dtype(), &DataType::Decimal(10, 2));
    assert_eq!(decimal_mantissa(df.column("fee").unwrap().get(0).unwrap()), Some(988));

    assert_eq!(df.column("bonus").unwrap().dtype(), &DataType::Decimal(18, 6));
    assert_eq!(decimal_mantissa(df.column("bonus").unwrap().get(0).unwrap()), Some(0));

    assert_eq!(df.column("direct").unwrap().dtype(), &DataType::Decimal(18, 6));
    assert_eq!(decimal_mantissa(df.column("direct").unwrap().get(0).unwrap()), Some(42));

    assert_eq!(
        df.column("history").unwrap().dtype(),
        &DataType::List(Box::new(DataType::Decimal(14, 3)))
    );
    assert_eq!(
        df.column("nullable_history").unwrap().dtype(),
        &DataType::List(Box::new(DataType::Decimal(14, 3)))
    );

    // Batch path also exercises the columnar `from_vec` / `from_iter_options`
    // finishers — assert the same mantissa values come back across rows.
    let batch = vec![row.clone(), row.clone(), row.clone()];
    let df_batch = batch.as_slice().to_dataframe().unwrap();
    assert_eq!(df_batch.height(), 3);
    assert_eq!(df_batch.column("price").unwrap().dtype(), &DataType::Decimal(12, 4));
    for i in 0..3 {
        assert_eq!(
            decimal_mantissa(df_batch.column("price").unwrap().get(i).unwrap()),
            Some(1_234_500)
        );
        assert_eq!(decimal_mantissa(df_batch.column("fee").unwrap().get(i).unwrap()), Some(988));
        assert_eq!(decimal_mantissa(df_batch.column("direct").unwrap().get(i).unwrap()), Some(42));
    }

    // None bonus → ensure the optional finisher preserves nulls in batch.
    let mut batch_with_null = batch.clone();
    batch_with_null[1].bonus = None;
    let df_null = batch_with_null.as_slice().to_dataframe().unwrap();
    let bonus_col = df_null.column("bonus").unwrap();
    assert_eq!(decimal_mantissa(bonus_col.get(0).unwrap()), Some(0));
    assert_eq!(decimal_mantissa(bonus_col.get(1).unwrap()), None);
    assert_eq!(decimal_mantissa(bonus_col.get(2).unwrap()), Some(0));

    // Banker's rounding at half-tie boundaries — drives the codegen path that
    // used to silently truncate. Each value is constructed at scale 4 and
    // dropped to scale 3 (or 2 for the "fee"-style cases), so the trailing
    // digit is exactly 5 and the rounding direction depends on the parity of
    // the truncated quotient.
    let edge = EdgeRow {
        half_to_even_up: Decimal::new(98755, 4),     // 9.8755 → 9.876 (q=9875 odd → +1)
        half_to_even_stay: Decimal::new(98745, 4),   // 9.8745 → 9.874 (q=9874 even → stay)
        half_to_even_up_low: Decimal::new(98735, 4), // 9.8735 → 9.874 (q=9873 odd → +1)
        negative_round_up: Decimal::new(-98765, 4),  // -9.8765 → -9.88 (magnitude > half)
        negative_half_to_even: Decimal::new(-98755, 4), // -9.8755 → -9.876 (q=9875 odd → +1)
    };

    let df_edge = edge.to_dataframe().unwrap();
    assert_eq!(
        decimal_mantissa(df_edge.column("half_to_even_up").unwrap().get(0).unwrap()),
        Some(9876)
    );
    assert_eq!(
        decimal_mantissa(df_edge.column("half_to_even_stay").unwrap().get(0).unwrap()),
        Some(9874)
    );
    assert_eq!(
        decimal_mantissa(df_edge.column("half_to_even_up_low").unwrap().get(0).unwrap()),
        Some(9874)
    );
    assert_eq!(
        decimal_mantissa(df_edge.column("negative_round_up").unwrap().get(0).unwrap()),
        Some(-988)
    );
    assert_eq!(
        decimal_mantissa(df_edge.column("negative_half_to_even").unwrap().get(0).unwrap()),
        Some(-9876)
    );

    // Cross-check the direct-mantissa path against polars's own
    // `str_to_dec128` for several scale-down inputs. Both paths must agree
    // exactly: any drift would mean the new codegen rounds differently than
    // the historical `to_string + cast` path it replaced.
    let cross_check_cases: &[(Decimal, usize, usize)] = &[
        (Decimal::new(98765, 4), 10, 2),     // 9.8765 → scale 2
        (Decimal::new(98755, 4), 10, 3),     // 9.8755 → scale 3 (half-tie up)
        (Decimal::new(98745, 4), 10, 3),     // 9.8745 → scale 3 (half-tie stay)
        (Decimal::new(98735, 4), 10, 3),     // 9.8735 → scale 3 (half-tie up)
        (Decimal::new(-98765, 4), 10, 2),    // negative round up
        (Decimal::new(-98755, 4), 10, 3),    // negative half-tie up
        (Decimal::new(123456, 5), 12, 3),    // 1.23456 → scale 3 (round-down, r > half)
        (Decimal::new(123451, 5), 12, 3),    // 1.23451 → scale 3 (round-down, r < half)
        (Decimal::new(1, 28), 38, 6), // tiny value: scale-down to 0
        // Largest 28-digit unsigned mantissa rust_decimal can hold; routed
        // through `Decimal::from_i128_with_scale` because `Decimal::new`
        // takes `i64`.
        (Decimal::from_i128_with_scale(99_999_999_999_999_999_999_999_999i128, 28), 38, 0),
    ];

    #[derive(ToDataFrame, Clone)]
    struct Single {
        #[df_derive(decimal(precision = 38, scale = 0))]
        v_38_0: Decimal,
    }

    for &(d, p, s) in cross_check_cases {
        let polars_path = polars_str_path_mantissa(d, p, s);

        // We can't dynamically vary the schema attribute, so re-route the
        // value through the matching field schema by selecting one of a
        // small set of pre-declared rows. This keeps the cross-check tight
        // without spinning up a new derive per case.
        let our_path = match (p, s) {
            (10, 2) => {
                let r = Row {
                    price: Decimal::new(0, 0),
                    fee: d,
                    bonus: None,
                    direct: Decimal::new(0, 0),
                    history: vec![],
                    nullable_history: vec![],
                };
                let df = r.to_dataframe().unwrap();
                decimal_mantissa(df.column("fee").unwrap().get(0).unwrap()).unwrap()
            }
            (10, 3) => {
                let r = EdgeRow {
                    half_to_even_up: d,
                    half_to_even_stay: Decimal::new(0, 0),
                    half_to_even_up_low: Decimal::new(0, 0),
                    negative_round_up: Decimal::new(0, 0),
                    negative_half_to_even: Decimal::new(0, 0),
                };
                let df = r.to_dataframe().unwrap();
                decimal_mantissa(df.column("half_to_even_up").unwrap().get(0).unwrap()).unwrap()
            }
            (38, 0) => {
                let r = Single { v_38_0: d };
                let df = r.to_dataframe().unwrap();
                decimal_mantissa(df.column("v_38_0").unwrap().get(0).unwrap()).unwrap()
            }
            // Used by `1.23456 / 1.23451` cases; route through a dedicated
            // schema so the precision/scale match what polars sees.
            (12, 3) => {
                #[derive(ToDataFrame, Clone)]
                struct V12_3 {
                    #[df_derive(decimal(precision = 12, scale = 3))]
                    v: Decimal,
                }
                let df = V12_3 { v: d }.to_dataframe().unwrap();
                decimal_mantissa(df.column("v").unwrap().get(0).unwrap()).unwrap()
            }
            (38, 6) => {
                #[derive(ToDataFrame, Clone)]
                struct V38_6 {
                    #[df_derive(decimal(precision = 38, scale = 6))]
                    v: Decimal,
                }
                let df = V38_6 { v: d }.to_dataframe().unwrap();
                decimal_mantissa(df.column("v").unwrap().get(0).unwrap()).unwrap()
            }
            _ => unreachable!("unexpected (p, s) = ({p}, {s}) in cross-check"),
        };

        assert_eq!(
            our_path, polars_path,
            "df-derive/polars rescale mismatch for {d} → Decimal({p}, {s}): df-derive={our_path}, polars={polars_path}"
        );
    }
}

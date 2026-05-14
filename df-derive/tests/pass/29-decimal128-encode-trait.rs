// Lock in the `Decimal128Encode` trait dispatch.
//
// The Decimal codegen no longer inlines `rust_decimal::Decimal::scale()` /
// `mantissa()`; it dispatches the rescale through a user-pluggable
// `Decimal128Encode` trait. This test exercises three properties that
// the trait dispatch must preserve:
//
//  1. With the default trait path, `rust_decimal::Decimal` rescales identically
//     to the historical inlined path (covered indirectly by tests/pass/27).
//  2. A user can override the trait path via
//     `#[df_derive(decimal128_encode = "...")]` and have the codegen call
//     their custom impl. The override resolves the same way the existing
//     `trait` / `columnar` keys do.
//  3. A `None` return from `try_to_i128_mantissa` surfaces as a polars
//     `ComputeError`, matching the historical scale-up overflow path.
//  4. Trait dispatch is unambiguous UFCS dispatch. An inherent method with
//     the same name must not hijack decimal encoding.
//
// We verify (2) and (3) with a stub `MyDecimal128Encode` trait whose impl
// for `rust_decimal::Decimal` always returns `None`. The codegen, with the
// override pointing at this trait, must call it on every value and turn
// the `None` into a `PolarsError::ComputeError` rather than panicking or
// silently emitting a zero mantissa.

use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::{Decimal128Encode, ToDataFrame, ToDataFrameVec};

use polars::prelude::*;
use rust_decimal::Decimal;

mod failing_traits {
    pub trait MyDecimal128Encode {
        fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128>;
    }

    // Always returns None: simulates a backend that can't represent the
    // value at the schema scale. The codegen must surface this as a
    // PolarsError::ComputeError on both the per-row and columnar paths.
    impl MyDecimal128Encode for rust_decimal::Decimal {
        fn try_to_i128_mantissa(&self, _target_scale: u32) -> Option<i128> {
            None
        }
    }

    // Implementing the trait for `Option<Decimal>` is unnecessary: the
    // codegen unwraps the Option layer before invoking the trait method.
}

#[derive(ToDataFrame, Clone)]
#[df_derive(decimal128_encode = "failing_traits::MyDecimal128Encode")]
struct AlwaysFailingRow {
    #[df_derive(decimal(precision = 18, scale = 6))]
    value: Decimal,
}

#[derive(ToDataFrame, Clone)]
#[df_derive(decimal128_encode = "failing_traits::MyDecimal128Encode")]
struct AlwaysFailingOption {
    #[df_derive(decimal(precision = 18, scale = 6))]
    value: Option<Decimal>,
}

#[derive(ToDataFrame, Clone)]
#[df_derive(decimal128_encode = "failing_traits::MyDecimal128Encode")]
struct AlwaysFailingVec {
    #[df_derive(decimal(precision = 18, scale = 6))]
    values: Vec<Decimal>,
}

#[derive(Clone)]
struct HijackDecimal(i128);

impl HijackDecimal {
    #[allow(dead_code)]
    fn try_to_i128_mantissa(&self, _target_scale: u32) -> Option<i128> {
        None
    }
}

impl Decimal128Encode for HijackDecimal {
    fn try_to_i128_mantissa(&self, _target_scale: u32) -> Option<i128> {
        Some(self.0)
    }
}

#[derive(ToDataFrame, Clone)]
struct UfcsDispatchRow<'a> {
    #[df_derive(decimal(precision = 18, scale = 2))]
    bare: HijackDecimal,
    #[df_derive(decimal(precision = 18, scale = 2))]
    by_ref: &'a HijackDecimal,
    #[df_derive(decimal(precision = 18, scale = 2))]
    boxed: Box<HijackDecimal>,
    #[df_derive(decimal(precision = 18, scale = 2))]
    maybe: Option<HijackDecimal>,
    #[df_derive(decimal(precision = 18, scale = 2))]
    maybe_boxed: Option<Box<HijackDecimal>>,
    #[df_derive(decimal(precision = 18, scale = 2))]
    values: Vec<HijackDecimal>,
    #[df_derive(decimal(precision = 18, scale = 2))]
    maybe_values: Vec<Option<HijackDecimal>>,
}

mod alias_decimal {
    use super::*;

    type Decimal = HijackDecimal;

    #[derive(ToDataFrame, Clone)]
    struct AliasRow {
        value: Decimal,
        maybe: Option<Decimal>,
        values: Vec<Decimal>,
    }

    pub fn assert_bare_decimal_alias_uses_encode_trait() {
        let row = AliasRow {
            value: HijackDecimal(70),
            maybe: Some(HijackDecimal(71)),
            values: vec![HijackDecimal(72)],
        };
        let df = row
            .to_dataframe()
            .expect("bare Decimal alias should route through Decimal128Encode");
        assert_eq!(df.height(), 1);
    }
}

fn assert_compute_err(err: PolarsError, ctx: &str) {
    match err {
        PolarsError::ComputeError(msg) => assert!(
            msg.contains("decimal mantissa rescale"),
            "{ctx}: unexpected ComputeError message: {msg}"
        ),
        other => panic!("{ctx}: expected PolarsError::ComputeError, got {other:?}"),
    }
}

fn main() {
    // (3) Per-row path: `to_dataframe` on a single instance routes through
    // the columnar pipeline, which calls the trait and surfaces None as
    // a PolarsError::ComputeError.
    let r = AlwaysFailingRow { value: Decimal::new(1, 0) };
    let err = r.to_dataframe().expect_err("expected ComputeError on single row");
    assert_compute_err(err, "single-row");

    // (3) Columnar path: same thing on a slice, exercising the bulk
    // populator finisher.
    let batch = vec![AlwaysFailingRow { value: Decimal::new(2, 0) }];
    let err = batch
        .as_slice()
        .to_dataframe()
        .expect_err("expected ComputeError on slice");
    assert_compute_err(err, "columnar");

    // (3) `Option<Decimal>` path: a `Some(_)` value flows through the
    // trait the same way; only `None` skips the call.
    let r_opt = AlwaysFailingOption { value: Some(Decimal::new(3, 0)) };
    let err = r_opt.to_dataframe().expect_err("expected ComputeError on Option<Decimal>");
    assert_compute_err(err, "option-some");

    // A pure `None` decimal value bypasses the trait entirely (no Some to
    // map), so it succeeds with a null cell in the column.
    let r_none = AlwaysFailingOption { value: None };
    let df = r_none
        .to_dataframe()
        .expect("None decimal should not call the encode trait");
    assert_eq!(df.height(), 1);
    assert_eq!(df.column("value").unwrap().get(0).unwrap(), AnyValue::Null);

    // (3) `Vec<Decimal>` path: the inner-Series builder maps each element
    // through the trait, so a single failing element fails the row build.
    let r_vec = AlwaysFailingVec { values: vec![Decimal::new(4, 0)] };
    let err = r_vec.to_dataframe().expect_err("expected ComputeError on Vec<Decimal>");
    assert_compute_err(err, "vec");

    // An empty `Vec<Decimal>` never invokes the trait, so it succeeds and
    // the column is an empty list.
    let r_vec_empty = AlwaysFailingVec { values: vec![] };
    let df = r_vec_empty
        .to_dataframe()
        .expect("empty Vec<Decimal> should not call the encode trait");
    assert_eq!(df.height(), 1);

    // (4) UFCS dispatch: every shape below would call the inherent method
    // and fail if the generated code used decimal dot syntax.
    let referenced = HijackDecimal(20);
    let hijack = UfcsDispatchRow {
        bare: HijackDecimal(10),
        by_ref: &referenced,
        boxed: Box::new(HijackDecimal(30)),
        maybe: Some(HijackDecimal(40)),
        maybe_boxed: Some(Box::new(HijackDecimal(50))),
        values: vec![HijackDecimal(60)],
        maybe_values: vec![Some(HijackDecimal(61))],
    };
    let df = hijack
        .to_dataframe()
        .expect("inherent method must not hijack Decimal128Encode");
    assert_eq!(df.height(), 1);

    alias_decimal::assert_bare_decimal_alias_uses_encode_trait();

    println!("Decimal128Encode None-return path: OK across single, columnar, option, vec shapes");
}

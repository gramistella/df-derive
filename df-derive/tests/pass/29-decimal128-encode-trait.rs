// Lock in the `Decimal128Encode` contract.
//
// Decimal columns are filled through the `Decimal128Encode` trait. This test
// pins three properties:
//
//  1. A `None` from `try_to_i128_mantissa` surfaces as a
//     `PolarsError::ComputeError` on single rows, slices, options and lists,
//     instead of panicking or storing a zero.
//  2. A `None` value in an `Option` never calls the trait.
//  3. Dispatch goes through the trait. An inherent method with the same name
//     must not hijack decimal encoding.
//
// A custom decimal type can also declare its column encoding once with
// `impl Columnar`, after which its fields need no attribute.

use df_derive::ToDataFrame;
use df_derive::dataframe::{Columnar, Decimal128Encode, ToDataFrame, ToDataFrameVec};

use polars::prelude::*;
use rust_decimal::Decimal;

// Always returns None: simulates a value that cannot be represented at the
// schema scale.
#[derive(Clone)]
struct Failing(#[allow(dead_code)] Decimal);

impl Decimal128Encode for Failing {
    fn try_to_i128_mantissa(&self, _target_scale: u32) -> Option<i128> {
        None
    }
}

#[derive(ToDataFrame, Clone)]
struct AlwaysFailingRow {
    #[df_derive(decimal(precision = 18, scale = 6))]
    value: Failing,
}

#[derive(ToDataFrame, Clone)]
struct AlwaysFailingOption {
    #[df_derive(decimal(precision = 18, scale = 6))]
    value: Option<Failing>,
}

#[derive(ToDataFrame, Clone)]
struct AlwaysFailingVec {
    #[df_derive(decimal(precision = 18, scale = 6))]
    values: Vec<Failing>,
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

impl Columnar for HijackDecimal {
    type Builder = df_derive::dataframe::builders::DecimalBuilder<38, 10>;
}

mod declared_decimal {
    use super::*;

    #[derive(ToDataFrame, Clone)]
    struct DeclaredRow {
        value: HijackDecimal,
        maybe: Option<HijackDecimal>,
        values: Vec<HijackDecimal>,
    }

    pub fn assert_declared_decimal_needs_no_attribute() {
        let row = DeclaredRow {
            value: HijackDecimal(70),
            maybe: Some(HijackDecimal(71)),
            values: vec![HijackDecimal(72)],
        };
        let df = row
            .to_dataframe()
            .expect("a type with a declared decimal encoding needs no field attribute");
        assert_eq!(df.height(), 1);
        assert_eq!(
            DeclaredRow::schema().unwrap()[0],
            ("value".to_string(), DataType::Decimal(38, 10))
        );
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
    // (3) One value: the trait returns `None`, which comes back as a
    // `PolarsError::ComputeError`.
    let r = AlwaysFailingRow { value: Failing(Decimal::new(1, 0)) };
    let err = r.to_dataframe().expect_err("expected ComputeError on single row");
    assert_compute_err(err, "single-row");

    // (3) A slice of values fails the same way.
    let batch = vec![AlwaysFailingRow { value: Failing(Decimal::new(2, 0)) }];
    let err = batch
        .as_slice()
        .to_dataframe()
        .expect_err("expected ComputeError on slice");
    assert_compute_err(err, "columnar");

    // (3) `Option<Decimal>` path: a `Some(_)` value flows through the
    // trait the same way; only `None` skips the call.
    let r_opt = AlwaysFailingOption { value: Some(Failing(Decimal::new(3, 0))) };
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
    let r_vec = AlwaysFailingVec { values: vec![Failing(Decimal::new(4, 0))] };
    let err = r_vec.to_dataframe().expect_err("expected ComputeError on Vec<Decimal>");
    assert_compute_err(err, "vec");

    // An empty `Vec<Decimal>` never invokes the trait, so it succeeds and
    // the column is an empty list.
    let r_vec_empty = AlwaysFailingVec { values: vec![] };
    let df = r_vec_empty
        .to_dataframe()
        .expect("empty Vec<Decimal> should not call the encode trait");
    assert_eq!(df.height(), 1);

    // (4) A type with an inherent method of the same name: every shape below
    // must call the trait's method, not the inherent one.
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

    declared_decimal::assert_declared_decimal_needs_no_attribute();

    println!("Decimal128Encode None-return path: OK across single, columnar, option, vec shapes");
}

use df_derive::ToDataFrame;
use df_derive::dataframe::{Columnar, Decimal128Encode, ToDataFrame, ToDataFrameVec};
use polars::prelude::*;
use std::cell::Cell;
use std::rc::Rc;

// A deliberately non-`Decimal`-named backend. The explicit `decimal(...)`
// attribute is the semantic opt-in that tells the macro to route this leaf
// through `Decimal128Encode`; unannotated custom structs still flatten as
// nested structs.
#[derive(Clone)]
struct MoneyAmount {
    mantissa: i128,
    scale: u32,
}

impl MoneyAmount {
    const fn new(mantissa: i128, scale: u32) -> Self {
        Self { mantissa, scale }
    }
}

impl Decimal128Encode for MoneyAmount {
    fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128> {
        match self.scale.cmp(&target_scale) {
            std::cmp::Ordering::Equal => Some(self.mantissa),
            std::cmp::Ordering::Less => {
                let pow = 10_i128.pow(target_scale - self.scale);
                self.mantissa.checked_mul(pow)
            }
            std::cmp::Ordering::Greater => {
                let pow = 10_i128.pow(self.scale - target_scale);
                Some(self.mantissa / pow)
            }
        }
    }
}

#[derive(ToDataFrame, Clone)]
struct CustomDecimalRow {
    #[df_derive(decimal(precision = 18, scale = 4))]
    price: MoneyAmount,
    #[df_derive(decimal(precision = 18, scale = 4))]
    optional: Option<MoneyAmount>,
    #[df_derive(decimal(precision = 18, scale = 4))]
    history: Vec<MoneyAmount>,
    #[df_derive(decimal(precision = 18, scale = 4))]
    nullable_history: Vec<Option<MoneyAmount>>,
}

#[derive(ToDataFrame)]
struct GenericDecimalRow<T> {
    #[df_derive(decimal(precision = 12, scale = 2))]
    amount: T,
}

#[derive(ToDataFrame, Clone)]
struct PrecisionScalar {
    #[df_derive(decimal(precision = 5, scale = 2))]
    amount: MoneyAmount,
}

#[derive(ToDataFrame, Clone)]
struct PrecisionOption {
    #[df_derive(decimal(precision = 5, scale = 2))]
    amount: Option<MoneyAmount>,
}

#[derive(ToDataFrame, Clone)]
struct PrecisionVec {
    #[df_derive(decimal(precision = 5, scale = 2))]
    amounts: Vec<MoneyAmount>,
}

#[derive(ToDataFrame, Clone)]
struct PrecisionNullableVec {
    #[df_derive(decimal(precision = 5, scale = 2))]
    amounts: Vec<Option<MoneyAmount>>,
}

#[derive(Clone)]
struct TrackedDecimal {
    mantissa: Option<i128>,
    calls: Rc<Cell<usize>>,
    panic: bool,
}

impl Decimal128Encode for TrackedDecimal {
    fn try_to_i128_mantissa(&self, _target_scale: u32) -> Option<i128> {
        self.calls.set(self.calls.get() + 1);
        assert!(!self.panic, "tracked decimal panic");
        self.mantissa
    }
}

#[derive(ToDataFrame)]
struct TrackedDecimalVec {
    #[df_derive(decimal(precision = 18, scale = 4))]
    values: Vec<TrackedDecimal>,
}

#[derive(ToDataFrame)]
struct TrackedNullableDecimalVec {
    #[df_derive(decimal(precision = 18, scale = 4))]
    values: Vec<Option<TrackedDecimal>>,
}

fn decimal_mantissa(av: AnyValue<'_>) -> Option<i128> {
    match av {
        AnyValue::Decimal(v, _p, _s) => Some(v),
        AnyValue::Null => None,
        other => panic!("expected AnyValue::Decimal or Null, got {other:?}"),
    }
}

fn assert_precision_error(result: PolarsResult<DataFrame>, ctx: &str) {
    let err = match result {
        Ok(df) => panic!("{ctx}: expected precision error, got {df:?}"),
        Err(err) => err,
    };
    match err {
        PolarsError::ComputeError(msg) => assert!(
            msg.contains("exceeds declared precision"),
            "{ctx}: unexpected ComputeError message: {msg}"
        ),
        other => panic!("{ctx}: expected PolarsError::ComputeError, got {other:?}"),
    }
}

fn schema_fields(schema: &SchemaRef) -> Vec<(String, DataType)> {
    schema
        .iter()
        .map(|(name, dtype)| (name.as_str().to_owned(), dtype.clone()))
        .collect()
}

#[test]
fn runtime_semantics() {
    let row = CustomDecimalRow {
        price: MoneyAmount::new(12_345, 2),
        optional: Some(MoneyAmount::new(7, 0)),
        history: vec![MoneyAmount::new(1, 4), MoneyAmount::new(25, 1)],
        nullable_history: vec![Some(MoneyAmount::new(9, 2)), None],
    };

    let schema = schema_fields(&CustomDecimalRow::schema().unwrap());
    assert_eq!(
        schema,
        vec![
            ("price".into(), DataType::Decimal(18, 4)),
            ("optional".into(), DataType::Decimal(18, 4)),
            (
                "history".into(),
                DataType::List(Box::new(DataType::Decimal(18, 4))),
            ),
            (
                "nullable_history".into(),
                DataType::List(Box::new(DataType::Decimal(18, 4))),
            ),
        ]
    );

    let df = row.to_dataframe().unwrap();
    assert_eq!(df.height(), 1);
    assert_eq!(
        df.column("price").unwrap().dtype(),
        &DataType::Decimal(18, 4)
    );
    assert_eq!(
        decimal_mantissa(df.column("price").unwrap().get(0).unwrap()),
        Some(1_234_500)
    );
    assert_eq!(
        decimal_mantissa(df.column("optional").unwrap().get(0).unwrap()),
        Some(70_000)
    );
    assert_eq!(
        df.column("history").unwrap().dtype(),
        &DataType::List(Box::new(DataType::Decimal(18, 4)))
    );

    let batch = vec![
        row.clone(),
        CustomDecimalRow {
            price: MoneyAmount::new(1, 4),
            optional: None,
            history: vec![],
            nullable_history: vec![None],
        },
    ];
    let df_batch = batch.as_slice().to_dataframe().unwrap();
    assert_eq!(df_batch.height(), 2);
    assert_eq!(
        decimal_mantissa(df_batch.column("price").unwrap().get(1).unwrap()),
        Some(1)
    );
    assert_eq!(
        decimal_mantissa(df_batch.column("optional").unwrap().get(1).unwrap()),
        None
    );

    let generic = GenericDecimalRow {
        amount: MoneyAmount::new(123, 1),
    };
    assert_eq!(
        schema_fields(&GenericDecimalRow::<MoneyAmount>::schema().unwrap()),
        vec![("amount".into(), DataType::Decimal(12, 2))]
    );
    let generic_df = generic.to_dataframe().unwrap();
    assert_eq!(
        decimal_mantissa(generic_df.column("amount").unwrap().get(0).unwrap()),
        Some(1_230)
    );

    let empty = CustomDecimalRow::empty_dataframe().unwrap();
    assert_eq!(
        empty.column("price").unwrap().dtype(),
        &DataType::Decimal(18, 4)
    );
}

#[test]
fn decimal_precision_is_validated_after_encode() {
    let valid = PrecisionScalar {
        amount: MoneyAmount::new(99_999, 2),
    }
    .to_dataframe()
    .unwrap();
    assert_eq!(
        decimal_mantissa(valid.column("amount").unwrap().get(0).unwrap()),
        Some(99_999)
    );

    assert_precision_error(
        PrecisionScalar {
            amount: MoneyAmount::new(100_000, 2),
        }
        .to_dataframe(),
        "scalar",
    );
    assert_precision_error(
        PrecisionOption {
            amount: Some(MoneyAmount::new(-100_000, 2)),
        }
        .to_dataframe(),
        "option",
    );
    assert_precision_error(
        PrecisionVec {
            amounts: vec![MoneyAmount::new(1, 2), MoneyAmount::new(100_000, 2)],
        }
        .to_dataframe(),
        "vec",
    );
    assert_precision_error(
        PrecisionNullableVec {
            amounts: vec![None, Some(MoneyAmount::new(100_000, 2))],
        }
        .to_dataframe(),
        "nullable vec",
    );
}

#[test]
fn custom_decimal_list_errors_do_not_consume_later_source_rows() {
    let yielded = Cell::new(0);
    let rows = [
        PrecisionVec {
            amounts: vec![MoneyAmount::new(i128::MAX, 0)],
        },
        PrecisionVec {
            amounts: vec![MoneyAmount::new(1, 2)],
        },
    ];
    assert_eq!(rows[0].amounts[0].try_to_i128_mantissa(2), None);

    let inspected = rows.iter().inspect(|_| yielded.set(yielded.get() + 1));
    let error = PrecisionVec::encode(inspected).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("decimal mantissa rescale to scale 2 failed"),
        "unexpected error: {error}",
    );
    assert_eq!(yielded.get(), 1);
}

#[test]
fn custom_decimal_list_errors_stop_inside_the_current_segment() {
    let tracked = |mantissa| {
        let calls = Rc::new(Cell::new(0));
        (
            TrackedDecimal {
                mantissa,
                calls: Rc::clone(&calls),
                panic: false,
            },
            calls,
        )
    };

    let (first, first_calls) = tracked(Some(1));
    let (failure, failure_calls) = tracked(None);
    let (trailing, trailing_calls) = tracked(Some(3));
    let error = TrackedDecimalVec {
        values: vec![first, failure, trailing],
    }
    .to_dataframe()
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("decimal mantissa rescale to scale 4 failed"),
        "unexpected error: {error}",
    );
    assert_eq!(first_calls.get(), 1);
    assert_eq!(failure_calls.get(), 1);
    assert_eq!(trailing_calls.get(), 0);

    let (first, first_calls) = tracked(Some(1));
    let (failure, failure_calls) = tracked(None);
    let (trailing, trailing_calls) = tracked(Some(3));
    let error = TrackedNullableDecimalVec {
        values: vec![None, Some(first), None, Some(failure), Some(trailing)],
    }
    .to_dataframe()
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("decimal mantissa rescale to scale 4 failed"),
        "unexpected error: {error}",
    );
    assert_eq!(first_calls.get(), 1);
    assert_eq!(failure_calls.get(), 1);
    assert_eq!(trailing_calls.get(), 0);
}

#[test]
fn custom_decimal_list_panics_stop_inside_the_current_segment() {
    let calls = || Rc::new(Cell::new(0));
    let first_calls = calls();
    let panic_calls = calls();
    let trailing_calls = calls();
    let row = TrackedDecimalVec {
        values: vec![
            TrackedDecimal {
                mantissa: Some(1),
                calls: Rc::clone(&first_calls),
                panic: false,
            },
            TrackedDecimal {
                mantissa: Some(2),
                calls: Rc::clone(&panic_calls),
                panic: true,
            },
            TrackedDecimal {
                mantissa: Some(3),
                calls: Rc::clone(&trailing_calls),
                panic: false,
            },
        ],
    };

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = row.to_dataframe();
    }));

    assert!(panic.is_err());
    assert_eq!(first_calls.get(), 1);
    assert_eq!(panic_calls.get(), 1);
    assert_eq!(trailing_calls.get(), 0);
}

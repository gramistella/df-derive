use crate::core::dataframe::{Decimal128Encode, ToDataFrame, ToDataFrameVec};
use df_derive::ToDataFrame;
use polars::prelude::*;
use std::fmt;

#[derive(ToDataFrame)]
struct Payload {
    value: i32,
}

#[derive(ToDataFrame)]
struct Shadowed<String, Decimal, NaiveDate> {
    direct: String,
    maybe: Option<Decimal>,
    listed: Vec<NaiveDate>,
}

struct LabelOnly {
    value: String,
}

impl LabelOnly {
    fn new(value: &str) -> Self {
        Self {
            value: value.to_string(),
        }
    }
}

impl AsRef<str> for LabelOnly {
    fn as_ref(&self) -> &str {
        &self.value
    }
}

impl fmt::Display for LabelOnly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.value)
    }
}

#[derive(ToDataFrame)]
struct AsStrOnly<T> {
    #[df_derive(as_str)]
    label: T,
}

#[derive(ToDataFrame)]
struct AsStringOnly<T> {
    #[df_derive(as_string)]
    label: T,
}

struct Cents {
    mantissa: i128,
    scale: u32,
}

impl Cents {
    const fn new(mantissa: i128, scale: u32) -> Self {
        Self { mantissa, scale }
    }
}

impl Decimal128Encode for Cents {
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

#[derive(ToDataFrame)]
struct DecimalOnly<T> {
    #[df_derive(decimal(precision = 10, scale = 2))]
    amount: T,
}

#[derive(ToDataFrame)]
struct InnerLabel<T>
where
    T: AsRef<str>,
{
    #[df_derive(as_str)]
    label: T,
}

#[derive(ToDataFrame)]
struct OuterLabel<T>
where
    T: AsRef<str>,
{
    inner: InnerLabel<T>,
}

struct WrappedLabel<T>(T);

impl<T: AsRef<str>> AsRef<str> for WrappedLabel<T> {
    fn as_ref(&self) -> &str {
        self.0.as_ref()
    }
}

#[derive(ToDataFrame)]
struct AsStrStructPath<T> {
    #[df_derive(as_str)]
    label: WrappedLabel<T>,
}

struct WrappedDisplay<T>(T);

impl<T: fmt::Display> fmt::Display for WrappedDisplay<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(ToDataFrame)]
struct AsStringStructPath<T> {
    #[df_derive(as_string)]
    label: WrappedDisplay<T>,
}

struct Money<T>(T);

impl<T> Decimal128Encode for Money<T>
where
    T: Copy + Into<i128>,
{
    fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128> {
        let pow = 10_i128.pow(target_scale);
        self.0.into().checked_mul(pow)
    }
}

#[derive(ToDataFrame)]
struct DecimalStructPath<T> {
    #[df_derive(decimal(precision = 10, scale = 2))]
    amount: Money<T>,
}

#[test]
fn runtime_semantics() {
    let rows = vec![
        Shadowed::<Payload, Payload, Payload> {
            direct: Payload { value: 1 },
            maybe: Some(Payload { value: 2 }),
            listed: vec![Payload { value: 10 }, Payload { value: 20 }],
        },
        Shadowed::<Payload, Payload, Payload> {
            direct: Payload { value: 3 },
            maybe: None,
            listed: vec![],
        },
    ];

    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(df.shape(), (2, 3));
    assert_eq!(df.column("direct.value").unwrap().dtype(), &DataType::Int32);
    assert_eq!(df.column("maybe.value").unwrap().dtype(), &DataType::Int32);
    assert_eq!(
        df.column("listed.value").unwrap().dtype(),
        &DataType::List(Box::new(DataType::Int32))
    );

    assert_eq!(
        df.column("direct.value").unwrap().get(0).unwrap(),
        AnyValue::Int32(1)
    );
    assert_eq!(
        df.column("maybe.value").unwrap().get(1).unwrap(),
        AnyValue::Null
    );

    let as_str_df = AsStrOnly {
        label: LabelOnly::new("hello"),
    }
    .to_dataframe()
    .unwrap();

    assert_eq!(
        as_str_df.column("label").unwrap().dtype(),
        &DataType::String
    );
    assert_eq!(
        as_str_df.column("label").unwrap().get(0).unwrap(),
        AnyValue::String("hello")
    );

    let as_string_df = AsStringOnly {
        label: LabelOnly::new("display"),
    }
    .to_dataframe()
    .unwrap();

    assert_eq!(
        as_string_df.column("label").unwrap().dtype(),
        &DataType::String
    );
    assert_eq!(
        as_string_df.column("label").unwrap().get(0).unwrap(),
        AnyValue::String("display")
    );

    let decimal_df = DecimalOnly {
        amount: Cents::new(123, 0),
    }
    .to_dataframe()
    .unwrap();

    assert_eq!(
        decimal_df.column("amount").unwrap().dtype(),
        &DataType::Decimal(10, 2)
    );

    match decimal_df.column("amount").unwrap().get(0).unwrap() {
        AnyValue::Decimal(mantissa, _, _) => assert_eq!(mantissa, 12_300),
        other => panic!("expected Decimal AnyValue, got {other:?}"),
    }
}

#[test]
fn nested_generic_as_str_does_not_require_inner_t_to_dataframe() {
    let row = OuterLabel {
        inner: InnerLabel {
            label: LabelOnly::new("ok"),
        },
    };

    let df = row.to_dataframe().unwrap();
    assert_eq!(df.shape(), (1, 1));
    assert_eq!(
        df.column("inner.label").unwrap().get(0).unwrap(),
        AnyValue::String("ok")
    );
}

#[test]
fn generic_as_str_struct_path_gets_exact_as_ref_bound() {
    let row = AsStrStructPath {
        label: WrappedLabel(LabelOnly::new("wrapped")),
    };

    let df = row.to_dataframe().unwrap();
    assert_eq!(df.shape(), (1, 1));
    assert_eq!(
        df.column("label").unwrap().get(0).unwrap(),
        AnyValue::String("wrapped")
    );
}

#[test]
fn generic_as_string_struct_path_gets_exact_display_bound() {
    let row = AsStringStructPath {
        label: WrappedDisplay(LabelOnly::new("shown")),
    };

    let df = row.to_dataframe().unwrap();
    assert_eq!(df.shape(), (1, 1));
    assert_eq!(
        df.column("label").unwrap().get(0).unwrap(),
        AnyValue::String("shown")
    );
}

#[test]
fn generic_decimal_struct_path_gets_exact_decimal_backend_bound() {
    let row = DecimalStructPath {
        amount: Money(123_i32),
    };

    let df = row.to_dataframe().unwrap();
    assert_eq!(
        df.column("amount").unwrap().dtype(),
        &DataType::Decimal(10, 2)
    );

    match df.column("amount").unwrap().get(0).unwrap() {
        AnyValue::Decimal(mantissa, _, _) => assert_eq!(mantissa, 12_300),
        other => panic!("expected Decimal AnyValue, got {other:?}"),
    }
}

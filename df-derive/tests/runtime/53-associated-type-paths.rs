use crate::core::dataframe::{Decimal128Encode, ToDataFrameVec};
use df_derive::ToDataFrame;
use polars::prelude::*;
use std::fmt;

trait Provider {
    type Item;
    type Label;
    type Money;
}

struct Spec;

#[derive(ToDataFrame)]
struct Payload {
    value: i32,
    name: String,
}

struct TextLabel {
    value: String,
}

impl TextLabel {
    fn new(value: &str) -> Self {
        Self {
            value: value.to_string(),
        }
    }
}

impl AsRef<str> for TextLabel {
    fn as_ref(&self) -> &str {
        &self.value
    }
}

impl fmt::Display for TextLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.value)
    }
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

impl Provider for Spec {
    type Item = Payload;
    type Label = TextLabel;
    type Money = Cents;
}

#[derive(ToDataFrame)]
struct Row<T: Provider> {
    item: <T as Provider>::Item,
    items: Vec<<T as Provider>::Item>,
    maybe_item: Option<<T as Provider>::Item>,
    pair: (<T as Provider>::Item, i32),
    tuple_items: Vec<(<T as Provider>::Item, Option<i32>)>,

    #[df_derive(as_str)]
    label: <T as Provider>::Label,

    #[df_derive(as_str)]
    labels: Vec<<T as Provider>::Label>,

    #[df_derive(as_string)]
    display: <T as Provider>::Label,

    #[df_derive(decimal(precision = 10, scale = 2))]
    amount: <T as Provider>::Money,

    #[df_derive(decimal(precision = 10, scale = 2))]
    amounts: Vec<<T as Provider>::Money>,
}

fn payload(value: i32, name: &str) -> Payload {
    Payload {
        value,
        name: name.to_string(),
    }
}

fn decimal_mantissa(av: AnyValue<'_>) -> Option<i128> {
    match av {
        AnyValue::Decimal(value, _, _) => Some(value),
        AnyValue::Null => None,
        other => panic!("expected decimal or null, got {other:?}"),
    }
}

#[test]
fn runtime_semantics() {
    let rows = vec![
        Row::<Spec> {
            item: payload(1, "root"),
            items: vec![payload(10, "a"), payload(20, "b")],
            maybe_item: Some(payload(30, "maybe")),
            pair: (payload(40, "pair"), 99),
            tuple_items: vec![
                (payload(50, "tuple-a"), Some(5)),
                (payload(60, "tuple-b"), None),
            ],
            label: TextLabel::new("borrowed"),
            labels: vec![TextLabel::new("left"), TextLabel::new("right")],
            display: TextLabel::new("displayed"),
            amount: Cents::new(123, 0),
            amounts: vec![Cents::new(5, 1), Cents::new(42, 2)],
        },
        Row::<Spec> {
            item: payload(2, "root-2"),
            items: vec![],
            maybe_item: None,
            pair: (payload(41, "pair-2"), 100),
            tuple_items: vec![],
            label: TextLabel::new("borrowed-2"),
            labels: vec![],
            display: TextLabel::new("displayed-2"),
            amount: Cents::new(456, 2),
            amounts: vec![],
        },
    ];

    let df = rows.as_slice().to_dataframe().unwrap();

    assert_eq!(df.shape(), (2, 17));
    assert_eq!(df.column("item.value").unwrap().dtype(), &DataType::Int32);
    assert_eq!(df.column("item.name").unwrap().dtype(), &DataType::String);
    assert_eq!(
        df.column("items.value").unwrap().dtype(),
        &DataType::List(Box::new(DataType::Int32))
    );
    assert_eq!(
        df.column("pair.field_0.value").unwrap().dtype(),
        &DataType::Int32
    );
    assert_eq!(
        df.column("tuple_items.field_0.value").unwrap().dtype(),
        &DataType::List(Box::new(DataType::Int32))
    );
    assert_eq!(
        df.column("tuple_items.field_1").unwrap().dtype(),
        &DataType::List(Box::new(DataType::Int32))
    );
    assert_eq!(df.column("label").unwrap().dtype(), &DataType::String);
    assert_eq!(
        df.column("labels").unwrap().dtype(),
        &DataType::List(Box::new(DataType::String))
    );
    assert_eq!(df.column("display").unwrap().dtype(), &DataType::String);
    assert_eq!(
        df.column("amount").unwrap().dtype(),
        &DataType::Decimal(10, 2)
    );
    assert_eq!(
        df.column("amounts").unwrap().dtype(),
        &DataType::List(Box::new(DataType::Decimal(10, 2)))
    );

    assert_eq!(
        df.column("item.value").unwrap().get(0).unwrap(),
        AnyValue::Int32(1)
    );
    assert_eq!(
        df.column("maybe_item.value").unwrap().get(1).unwrap(),
        AnyValue::Null
    );
    assert_eq!(
        df.column("label").unwrap().get(0).unwrap(),
        AnyValue::String("borrowed")
    );
    assert_eq!(
        df.column("display").unwrap().get(0).unwrap(),
        AnyValue::String("displayed")
    );
    assert_eq!(
        decimal_mantissa(df.column("amount").unwrap().get(0).unwrap()),
        Some(12_300)
    );
    assert_eq!(
        decimal_mantissa(df.column("amount").unwrap().get(1).unwrap()),
        Some(456)
    );
}

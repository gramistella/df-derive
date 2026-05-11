use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};
use df_derive::ToDataFrame;
use polars::prelude::*;

mod domain {
    use df_derive::ToDataFrame;

    #[derive(ToDataFrame, Clone)]
    pub struct Leaf {
        pub value: i32,
        pub label: String,
    }

    #[derive(ToDataFrame, Clone)]
    pub struct GenericBox<T> {
        pub payload: T,
    }

    #[derive(Clone)]
    pub struct Label {
        value: String,
    }

    impl Label {
        pub fn new(value: &str) -> Self {
            Self {
                value: value.to_string(),
            }
        }
    }

    impl AsRef<str> for Label {
        fn as_ref(&self) -> &str {
            &self.value
        }
    }

    #[derive(Clone)]
    pub struct Money {
        mantissa: i128,
        scale: u32,
    }

    impl Money {
        pub const fn new(mantissa: i128, scale: u32) -> Self {
            Self { mantissa, scale }
        }
    }

    impl crate::core::dataframe::Decimal128Encode for Money {
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
}

#[derive(ToDataFrame, Clone)]
struct Row {
    leaf: domain::Leaf,
    leaves: Vec<domain::Leaf>,
    maybe_leaf: Option<domain::Leaf>,
    pair: (domain::Leaf, i64),
    generic_leaf: domain::GenericBox<domain::Leaf>,

    #[df_derive(as_str)]
    label: domain::Label,

    #[df_derive(as_str)]
    labels: Vec<domain::Label>,

    #[df_derive(decimal(precision = 12, scale = 2))]
    amount: domain::Money,

    #[df_derive(decimal(precision = 12, scale = 2))]
    amounts: Vec<domain::Money>,
}

fn decimal_mantissa(av: AnyValue<'_>) -> Option<i128> {
    match av {
        AnyValue::Decimal(value, _precision, _scale) => Some(value),
        AnyValue::Null => None,
        other => panic!("expected decimal or null, got {other:?}"),
    }
}

#[test]
fn runtime_semantics() {
    let row = Row {
        leaf: domain::Leaf {
            value: 1,
            label: "root".to_string(),
        },
        leaves: vec![
            domain::Leaf {
                value: 10,
                label: "a".to_string(),
            },
            domain::Leaf {
                value: 20,
                label: "b".to_string(),
            },
        ],
        maybe_leaf: Some(domain::Leaf {
            value: 30,
            label: "maybe".to_string(),
        }),
        pair: (
            domain::Leaf {
                value: 40,
                label: "pair".to_string(),
            },
            99,
        ),
        generic_leaf: domain::GenericBox {
            payload: domain::Leaf {
                value: 50,
                label: "generic".to_string(),
            },
        },
        label: domain::Label::new("qualified-label"),
        labels: vec![domain::Label::new("left"), domain::Label::new("right")],
        amount: domain::Money::new(123, 1),
        amounts: vec![domain::Money::new(1, 2), domain::Money::new(250, 1)],
    };

    let df = row.clone().to_dataframe().unwrap();

    assert_eq!(df.height(), 1);
    assert_eq!(df.column("leaf.value").unwrap().dtype(), &DataType::Int32);
    assert_eq!(df.column("leaf.label").unwrap().dtype(), &DataType::String);
    assert_eq!(
        df.column("leaves.value").unwrap().dtype(),
        &DataType::List(Box::new(DataType::Int32))
    );
    assert_eq!(
        df.column("maybe_leaf.value").unwrap().dtype(),
        &DataType::Int32
    );
    assert_eq!(
        df.column("pair.field_0.value").unwrap().dtype(),
        &DataType::Int32
    );
    assert_eq!(df.column("pair.field_1").unwrap().dtype(), &DataType::Int64);
    assert_eq!(
        df.column("generic_leaf.payload.value").unwrap().dtype(),
        &DataType::Int32
    );
    assert_eq!(df.column("label").unwrap().dtype(), &DataType::String);
    assert_eq!(
        df.column("labels").unwrap().dtype(),
        &DataType::List(Box::new(DataType::String))
    );
    assert_eq!(
        df.column("amount").unwrap().dtype(),
        &DataType::Decimal(12, 2)
    );
    assert_eq!(
        df.column("amounts").unwrap().dtype(),
        &DataType::List(Box::new(DataType::Decimal(12, 2)))
    );

    assert_eq!(
        df.column("leaf.value").unwrap().get(0).unwrap(),
        AnyValue::Int32(1)
    );
    assert_eq!(
        df.column("label").unwrap().get(0).unwrap(),
        AnyValue::String("qualified-label")
    );
    assert_eq!(
        decimal_mantissa(df.column("amount").unwrap().get(0).unwrap()),
        Some(1_230)
    );

    let batch = vec![
        row,
        Row {
            leaf: domain::Leaf {
                value: 2,
                label: "root2".to_string(),
            },
            leaves: vec![],
            maybe_leaf: None,
            pair: (
                domain::Leaf {
                    value: 41,
                    label: "pair2".to_string(),
                },
                100,
            ),
            generic_leaf: domain::GenericBox {
                payload: domain::Leaf {
                    value: 51,
                    label: "generic2".to_string(),
                },
            },
            label: domain::Label::new("qualified-label-2"),
            labels: vec![],
            amount: domain::Money::new(456, 2),
            amounts: vec![],
        },
    ];

    let batch_df = batch.as_slice().to_dataframe().unwrap();
    assert_eq!(batch_df.height(), 2);
    assert_eq!(
        batch_df.column("maybe_leaf.value").unwrap().get(1).unwrap(),
        AnyValue::Null
    );
    assert_eq!(
        decimal_mantissa(batch_df.column("amount").unwrap().get(1).unwrap()),
        Some(456)
    );
}

use crate::core::dataframe::ToDataFrame;
use df_derive::ToDataFrame;
use polars::prelude::*;

mod domain {
    use df_derive::ToDataFrame;

    #[derive(ToDataFrame)]
    pub struct String {
        pub value: i32,
    }

    #[derive(ToDataFrame)]
    pub struct NaiveDate {
        pub value: i32,
    }

    #[derive(ToDataFrame)]
    pub struct NaiveDateTime {
        pub value: i32,
    }

    #[derive(ToDataFrame)]
    pub struct DateTime<T> {
        pub payload: T,
    }

    #[derive(ToDataFrame)]
    pub struct Decimal {
        pub value: i32,
    }
}

#[derive(ToDataFrame)]
struct Row {
    s: domain::String,
    d: domain::NaiveDate,
    dt: domain::NaiveDateTime,
    event: domain::DateTime<domain::String>,
    amount: domain::Decimal,
}

#[derive(ToDataFrame, Clone)]
struct QualifiedBuiltins {
    s: std::string::String,
    opt_s: std::option::Option<std::string::String>,
    maybe: std::option::Option<i32>,
    v: std::vec::Vec<std::string::String>,
    values: std::vec::Vec<i32>,
    boxed: std::boxed::Box<i64>,
    rc_label: std::rc::Rc<std::string::String>,
    arc_label: std::sync::Arc<std::string::String>,
    cow_label: std::borrow::Cow<'static, str>,
    nz: std::num::NonZeroU32,
    d: chrono::NaiveDate,
    amount: rust_decimal::Decimal,
}

fn schema_dtype(schema: &[(String, DataType)], col: &str) -> DataType {
    schema
        .iter()
        .find(|(name, _)| name == col)
        .map(|(_, dtype)| dtype.clone())
        .unwrap_or_else(|| panic!("column {col} missing"))
}

#[test]
fn runtime_semantics() {
    let df = Row {
        s: domain::String { value: 1 },
        d: domain::NaiveDate { value: 2 },
        dt: domain::NaiveDateTime { value: 3 },
        event: domain::DateTime {
            payload: domain::String { value: 4 },
        },
        amount: domain::Decimal { value: 5 },
    }
    .to_dataframe()
    .unwrap();

    assert_eq!(df.shape(), (1, 5));
    assert_eq!(
        df.get_column_names(),
        [
            "s.value",
            "d.value",
            "dt.value",
            "event.payload.value",
            "amount.value",
        ]
    );
    assert_eq!(df.column("s.value").unwrap().dtype(), &DataType::Int32);
    assert_eq!(df.column("d.value").unwrap().dtype(), &DataType::Int32);
    assert_eq!(df.column("dt.value").unwrap().dtype(), &DataType::Int32);
    assert_eq!(
        df.column("event.payload.value").unwrap().dtype(),
        &DataType::Int32
    );
    assert_eq!(df.column("amount.value").unwrap().dtype(), &DataType::Int32);
    assert_eq!(
        df.column("amount.value").unwrap().get(0).unwrap(),
        AnyValue::Int32(5)
    );
}

#[test]
fn canonical_qualified_builtin_paths() {
    let schema = QualifiedBuiltins::schema().unwrap();
    assert_eq!(schema_dtype(&schema, "s"), DataType::String);
    assert_eq!(schema_dtype(&schema, "opt_s"), DataType::String);
    assert_eq!(schema_dtype(&schema, "maybe"), DataType::Int32);
    assert_eq!(
        schema_dtype(&schema, "v"),
        DataType::List(Box::new(DataType::String))
    );
    assert_eq!(
        schema_dtype(&schema, "values"),
        DataType::List(Box::new(DataType::Int32))
    );
    assert_eq!(schema_dtype(&schema, "boxed"), DataType::Int64);
    assert_eq!(schema_dtype(&schema, "rc_label"), DataType::String);
    assert_eq!(schema_dtype(&schema, "arc_label"), DataType::String);
    assert_eq!(schema_dtype(&schema, "cow_label"), DataType::String);
    assert_eq!(schema_dtype(&schema, "nz"), DataType::UInt32);
    assert_eq!(schema_dtype(&schema, "d"), DataType::Date);
    assert_eq!(schema_dtype(&schema, "amount"), DataType::Decimal(38, 10));

    let df = QualifiedBuiltins {
        s: std::string::String::from("root"),
        opt_s: std::option::Option::Some(std::string::String::from("maybe")),
        maybe: std::option::Option::Some(42),
        v: std::vec![
            std::string::String::from("left"),
            std::string::String::from("right"),
        ],
        values: std::vec![1, 2, 3],
        boxed: std::boxed::Box::new(99),
        rc_label: std::rc::Rc::new(std::string::String::from("rc")),
        arc_label: std::sync::Arc::new(std::string::String::from("arc")),
        cow_label: std::borrow::Cow::Borrowed("cow"),
        nz: std::num::NonZeroU32::new(7).unwrap(),
        d: chrono::NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
        amount: rust_decimal::Decimal::new(12345, 2),
    }
    .to_dataframe()
    .unwrap();

    assert_eq!(df.shape(), (1, 12));
    assert_eq!(
        df.column("s").unwrap().get(0).unwrap(),
        AnyValue::String("root")
    );
    assert_eq!(
        df.column("maybe").unwrap().get(0).unwrap(),
        AnyValue::Int32(42)
    );
    assert_eq!(
        df.column("boxed").unwrap().get(0).unwrap(),
        AnyValue::Int64(99)
    );
    assert_eq!(
        df.column("rc_label").unwrap().get(0).unwrap(),
        AnyValue::String("rc")
    );
    assert_eq!(
        df.column("arc_label").unwrap().get(0).unwrap(),
        AnyValue::String("arc")
    );
    assert_eq!(
        df.column("cow_label").unwrap().get(0).unwrap(),
        AnyValue::String("cow")
    );
    assert_eq!(
        df.column("nz").unwrap().get(0).unwrap(),
        AnyValue::UInt32(7)
    );
}

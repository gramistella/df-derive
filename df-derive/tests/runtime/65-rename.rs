// `rename` replaces the name a field gives its columns: the column's own
// name for a value, the prefix for a nested row.

use df_derive::ToDataFrame;
use df_derive::dataframe::{ToDataFrame, ToDataFrameVec};
use polars::prelude::*;

fn names(df: &DataFrame) -> Vec<String> {
    df.get_column_names()
        .iter()
        .map(|name| name.to_string())
        .collect()
}

enum Venue {
    Lit,
    Dark,
}

impl AsRef<str> for Venue {
    fn as_ref(&self) -> &str {
        match self {
            Self::Lit => "lit",
            Self::Dark => "dark",
        }
    }
}

#[derive(ToDataFrame)]
struct Leg {
    #[df_derive(rename = "px")]
    price: f64,
    quantity: u32,
}

#[derive(ToDataFrame)]
struct Order {
    #[df_derive(rename = "id")]
    identifier: u64,
    // A raw identifier, renamed to a name that is a keyword.
    #[df_derive(rename = "type")]
    r#kind: String,
    // With an encoding.
    #[df_derive(rename = "venue", as_str)]
    place: Venue,
    // On a nested row the name is the prefix of its columns.
    #[df_derive(rename = "leg")]
    first: Leg,
    // Wrappers do not change that.
    #[df_derive(rename = "legs")]
    rest: Vec<Leg>,
    #[df_derive(rename = "last")]
    maybe: Option<Leg>,
    #[df_derive(rename = "pair")]
    tuple: (i32, String),
    untouched: bool,
}

#[derive(ToDataFrame)]
struct Point(
    #[df_derive(rename = "x")] f64,
    #[df_derive(rename = "y")] f64,
    f64,
);

#[test]
fn a_renamed_field_names_its_columns() {
    let rows = [Order {
        identifier: 7,
        r#kind: "limit".to_owned(),
        place: Venue::Dark,
        first: Leg {
            price: 1.5,
            quantity: 2,
        },
        rest: vec![Leg {
            price: 2.5,
            quantity: 3,
        }],
        maybe: None,
        tuple: (4, "four".to_owned()),
        untouched: true,
    }];

    let df = rows.as_slice().to_dataframe().unwrap();
    let expected = [
        "id",
        "type",
        "venue",
        "leg.px",
        "leg.quantity",
        "legs.px",
        "legs.quantity",
        "last.px",
        "last.quantity",
        "pair.field_0",
        "pair.field_1",
        "untouched",
    ];
    assert_eq!(names(&df), expected);
    // The schema says the same without any rows.
    let schema: Vec<String> = Order::schema()
        .unwrap()
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    assert_eq!(schema, expected);

    assert_eq!(
        df.column("id").unwrap().get(0).unwrap(),
        AnyValue::UInt64(7)
    );
    assert_eq!(
        df.column("venue").unwrap().get(0).unwrap(),
        AnyValue::String("dark")
    );
    assert_eq!(
        df.column("leg.px").unwrap().get(0).unwrap(),
        AnyValue::Float64(1.5)
    );
    assert_eq!(
        df.column("last.px").unwrap().get(0).unwrap(),
        AnyValue::Null
    );
    let _ = Venue::Lit;
}

#[test]
fn a_tuple_struct_field_can_be_named() {
    let df = [Point(1.0, 2.0, 3.0)].as_slice().to_dataframe().unwrap();
    assert_eq!(names(&df), ["x", "y", "field_2"]);
    assert_eq!(
        df.column("y").unwrap().get(0).unwrap(),
        AnyValue::Float64(2.0)
    );
}

// --- an error names the column, not the field ------------------------------------

struct Unprintable;

impl std::fmt::Display for Unprintable {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Err(std::fmt::Error)
    }
}

#[derive(ToDataFrame)]
struct Inner {
    #[df_derive(rename = "text", as_string)]
    value: Unprintable,
}

#[derive(ToDataFrame)]
struct Outer {
    #[df_derive(rename = "inner")]
    nested: Inner,
}

#[test]
fn an_error_names_the_renamed_path() {
    let rows = [Outer {
        nested: Inner { value: Unprintable },
    }];
    let error = rows.as_slice().to_dataframe().unwrap_err().to_string();
    assert!(error.contains("Outer[0].inner.text"), "{error}");
}

// --- two fields with one name are still an error ----------------------------------

#[derive(ToDataFrame)]
struct Clash {
    #[df_derive(rename = "b")]
    a: i32,
    b: i32,
}

#[test]
fn a_name_given_twice_is_an_error() {
    assert!(Clash::schema().is_err());
    let error = [Clash { a: 1, b: 2 }]
        .as_slice()
        .to_dataframe()
        .unwrap_err()
        .to_string();
    assert!(error.contains('b'), "{error}");
}

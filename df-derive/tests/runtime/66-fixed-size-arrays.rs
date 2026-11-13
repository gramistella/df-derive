// A fixed-size array is a list column, like a `Vec` or a slice: its length
// is not part of the column's type.

use df_derive::ToDataFrame;
use df_derive::dataframe::{ToDataFrame, ToDataFrameVec};
use polars::prelude::*;

fn dtype(df: &DataFrame, name: &str) -> DataType {
    df.column(name).unwrap().dtype().clone()
}

fn list(inner: DataType) -> DataType {
    DataType::List(Box::new(inner))
}

/// A value written out in full: Polars leaves the middle of a long list out
/// when it prints one.
fn written_out(value: &AnyValue<'_>) -> String {
    match value {
        AnyValue::List(items) => {
            let items: Vec<String> = items.iter().map(|item| written_out(&item)).collect();
            format!("[{}]", items.join(", "))
        }
        AnyValue::String(text) => format!("{text:?}"),
        AnyValue::StringOwned(text) => format!("{:?}", text.as_str()),
        other => other.to_string(),
    }
}

/// The value of one column in one row, as text to compare by.
fn row(df: &DataFrame, name: &str, row: usize) -> String {
    written_out(&df.column(name).unwrap().get(row).unwrap())
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
struct Point {
    x: f64,
    y: f64,
}

#[derive(ToDataFrame)]
struct Sample {
    position: [f64; 3],
    labels: [String; 2],
    grid: [[i32; 2]; 2],
    maybe: Option<[u8; 2]>,
    many: Vec<[i16; 2]>,
    borrowed: &'static [i64; 2],
    bytes: [u8; 4],
    #[df_derive(as_binary)]
    digest: [u8; 4],
    #[df_derive(as_str)]
    venues: [Venue; 2],
    #[df_derive(as_string)]
    optional_ids: [Option<u32>; 2],
    nothing: [i32; 0],
    corners: [Point; 2],
    #[df_derive(flatten)]
    ends: [Point; 2],
}

fn sample(maybe: Option<[u8; 2]>) -> Sample {
    Sample {
        position: [1.0, 2.0, 3.0],
        labels: ["a".to_owned(), "b".to_owned()],
        grid: [[1, 2], [3, 4]],
        maybe,
        many: vec![[1, 2], [3, 4], [5, 6]],
        borrowed: &[7, 8],
        bytes: [1, 2, 3, 4],
        digest: [0xde, 0xad, 0xbe, 0xef],
        venues: [Venue::Lit, Venue::Dark],
        optional_ids: [Some(9), None],
        nothing: [],
        corners: [Point { x: 0.0, y: 0.5 }, Point { x: 1.0, y: 1.5 }],
        ends: [Point { x: 2.0, y: 2.5 }, Point { x: 3.0, y: 3.5 }],
    }
}

#[test]
fn an_array_is_a_list_column() {
    let rows = [sample(Some([5, 6])), sample(None)];
    let df = rows.as_slice().to_dataframe().unwrap();

    assert_eq!(dtype(&df, "position"), list(DataType::Float64));
    assert_eq!(dtype(&df, "labels"), list(DataType::String));
    assert_eq!(dtype(&df, "grid"), list(list(DataType::Int32)));
    assert_eq!(dtype(&df, "maybe"), list(DataType::UInt8));
    assert_eq!(dtype(&df, "many"), list(list(DataType::Int16)));
    assert_eq!(dtype(&df, "borrowed"), list(DataType::Int64));
    // Bytes are a list unless the field says they are binary.
    assert_eq!(dtype(&df, "bytes"), list(DataType::UInt8));
    assert_eq!(dtype(&df, "digest"), DataType::Binary);
    // An encoding applies to the elements.
    assert_eq!(dtype(&df, "venues"), list(DataType::String));
    assert_eq!(dtype(&df, "optional_ids"), list(DataType::String));
    assert_eq!(dtype(&df, "nothing"), list(DataType::Int32));
    // An array of rows is one list column per field of the row.
    assert_eq!(dtype(&df, "corners.x"), list(DataType::Float64));
    assert_eq!(dtype(&df, "x"), list(DataType::Float64));

    assert_eq!(row(&df, "position", 0), "[1.0, 2.0, 3.0]");
    assert_eq!(row(&df, "labels", 1), r#"["a", "b"]"#);
    assert_eq!(row(&df, "grid", 0), "[[1, 2], [3, 4]]");
    assert_eq!(row(&df, "maybe", 0), "[5, 6]");
    assert_eq!(df.column("maybe").unwrap().get(1).unwrap(), AnyValue::Null);
    assert_eq!(row(&df, "many", 1), "[[1, 2], [3, 4], [5, 6]]");
    assert_eq!(row(&df, "borrowed", 0), "[7, 8]");
    assert_eq!(row(&df, "bytes", 0), "[1, 2, 3, 4]");
    assert_eq!(
        df.column("digest").unwrap().get(0).unwrap(),
        AnyValue::Binary(&[0xde, 0xad, 0xbe, 0xef])
    );
    assert_eq!(row(&df, "venues", 0), r#"["lit", "dark"]"#);
    assert_eq!(row(&df, "optional_ids", 0), r#"["9", null]"#);
    assert_eq!(row(&df, "nothing", 0), "[]");
    assert_eq!(row(&df, "corners.y", 0), "[0.5, 1.5]");
    assert_eq!(row(&df, "y", 1), "[2.5, 3.5]");
}

// --- the length may be a parameter of the row type --------------------------------

#[derive(ToDataFrame)]
struct Grid<const N: usize> {
    id: u32,
    cells: [f64; N],
}

#[derive(ToDataFrame)]
struct Grids {
    small: Grid<2>,
    large: Grid<3>,
}

#[test]
fn an_array_length_may_be_a_parameter_of_the_row_type() {
    let rows = [Grids {
        small: Grid {
            id: 1,
            cells: [0.5, 1.5],
        },
        large: Grid {
            id: 2,
            cells: [2.5, 3.5, 4.5],
        },
    }];
    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(dtype(&df, "small.cells"), list(DataType::Float64));
    assert_eq!(row(&df, "small.cells", 0), "[0.5, 1.5]");
    assert_eq!(row(&df, "large.cells", 0), "[2.5, 3.5, 4.5]");

    // Rows of one length are a frame of their own.
    let direct = [Grid {
        id: 3,
        cells: [9.0; 4],
    }];
    let df = direct.as_slice().to_dataframe().unwrap();
    assert_eq!(row(&df, "cells", 0), "[9.0, 9.0, 9.0, 9.0]");
    assert_eq!(
        Grid::<7>::schema().unwrap()[1],
        ("cells".to_owned(), list(DataType::Float64))
    );
}

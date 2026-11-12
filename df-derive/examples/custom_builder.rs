// A hand-written `ColumnBuilder`, selected for one field with `with`.
//
// `Cents` stores a floating-point price as integer cents. It wraps the
// provided builder for `i64` columns and delegates everything but the
// conversion to it, which is the usual shape of a custom builder.

use df_derive::dataframe::builders::PrimitiveBuilder;
use df_derive::dataframe::{Bitmap, ColumnBuilder, Fields};
use df_derive::prelude::*;
use polars::prelude::{Column, PlSmallStr, PolarsResult};

struct Cents(PrimitiveBuilder<i64>);

impl ColumnBuilder<f64> for Cents {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        <PrimitiveBuilder<i64> as ColumnBuilder<i64>>::schema(name, out);
    }

    fn new() -> Self {
        Self(ColumnBuilder::<i64>::new())
    }

    fn reserve(&mut self, additional: usize) {
        ColumnBuilder::<i64>::reserve(&mut self.0, additional);
    }

    #[allow(clippy::cast_possible_truncation)]
    fn push(&mut self, value: Option<&f64>) -> PolarsResult<()> {
        let cents = value.map(|value| (value * 100.0).round() as i64);
        self.0.push(cents.as_ref())
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        ColumnBuilder::<i64>::finish(self.0, name, validity, out)
    }
}

#[derive(ToDataFrame)]
struct Quote {
    symbol: String,
    #[df_derive(with = "Cents")]
    last: f64,
    // The builder applies to the innermost value: this is a nullable column
    // and the next a list column, both of integer cents.
    #[df_derive(with = "Cents")]
    close: Option<f64>,
    #[df_derive(with = "Cents")]
    fills: Vec<f64>,
}

fn main() -> PolarsResult<()> {
    let quotes = vec![
        Quote {
            symbol: "AAPL".into(),
            last: 187.23,
            close: Some(186.4),
            fills: vec![187.2, 187.25],
        },
        Quote {
            symbol: "MSFT".into(),
            last: 411.61,
            close: None,
            fills: vec![],
        },
    ];

    let df = quotes.as_slice().to_dataframe()?;
    println!("{df}");
    for (name, dtype) in Quote::schema()? {
        println!("  {name}: {dtype:?}");
    }
    Ok(())
}

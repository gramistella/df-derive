// Types whose columns are not their fields: an enum spread over several
// columns, and a struct exported through a row that borrows from it.
//
// `via` names a row struct. Each value is converted to it with `From<&T>`
// and written straight into the columns, so no collection of rows is built.

use df_derive::prelude::*;
use polars::prelude::PolarsResult;

#[derive(ToDataFrame)]
#[df_derive(via = "ActionRow")]
enum Action {
    Dividend { amount: f64 },
    Split { numerator: u32, denominator: u32 },
}

#[derive(ToDataFrame)]
struct ActionRow {
    kind: &'static str,
    amount: Option<f64>,
    numerator: Option<u32>,
    denominator: Option<u32>,
}

impl From<&Action> for ActionRow {
    fn from(action: &Action) -> Self {
        match *action {
            Action::Dividend { amount } => Self {
                kind: "dividend",
                amount: Some(amount),
                numerator: None,
                denominator: None,
            },
            Action::Split {
                numerator,
                denominator,
            } => Self {
                kind: "split",
                amount: None,
                numerator: Some(numerator),
                denominator: Some(denominator),
            },
        }
    }
}

// The row borrows from the value it is built from. Its lifetime is written
// as `'_` in the attribute.
#[derive(ToDataFrame)]
#[df_derive(via = "InstrumentRow<'_>")]
struct Instrument {
    symbol: String,
    exchange: Option<String>,
}

#[derive(ToDataFrame)]
struct InstrumentRow<'a> {
    symbol: &'a str,
    exchange: Option<&'a str>,
    listed: bool,
}

impl<'a> From<&'a Instrument> for InstrumentRow<'a> {
    fn from(instrument: &'a Instrument) -> Self {
        Self {
            symbol: &instrument.symbol,
            exchange: instrument.exchange.as_deref(),
            listed: instrument.exchange.is_some(),
        }
    }
}

// A projected type nests like any other: here as a field and in a list.
#[derive(ToDataFrame)]
struct Holding {
    instrument: Instrument,
    history: Vec<Action>,
}

fn main() -> PolarsResult<()> {
    let actions = [
        Action::Dividend { amount: 0.24 },
        Action::Split {
            numerator: 4,
            denominator: 1,
        },
    ];
    println!("{}", actions.as_slice().to_dataframe()?);

    let holding = Holding {
        instrument: Instrument {
            symbol: "AAPL".into(),
            exchange: Some("XNAS".into()),
        },
        history: actions.into(),
    };
    println!("{}", holding.to_dataframe()?);
    Ok(())
}

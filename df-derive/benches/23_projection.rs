// `#[df_derive(via = "...")]` encodes a type through a row built from each
// value. The alternative it replaces is converting by hand: collect the rows
// into a `Vec` and encode that. The two are measured side by side, at the top
// level and as the elements of a list, with a row that copies its fields and
// a row that borrows from the value.
use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;

#[path = "support/mod.rs"]
mod bench_support;
use crate::bench_support::configure_criterion;
use df_derive::dataframe::ToDataFrameVec;

const N_ROWS: usize = 100_000;

#[derive(ToDataFrame, Clone)]
#[df_derive(via = "ActionRow")]
enum Action {
    Dividend {
        day: i32,
        amount: f64,
    },
    Split {
        day: i32,
        numerator: u32,
        denominator: u32,
    },
    CapitalGain {
        day: i32,
        gain: f64,
    },
}

#[derive(ToDataFrame)]
struct ActionRow {
    kind: &'static str,
    day: i32,
    amount: Option<f64>,
    numerator: Option<u32>,
    denominator: Option<u32>,
    gain: Option<f64>,
}

impl From<&Action> for ActionRow {
    fn from(action: &Action) -> Self {
        let empty = Self {
            kind: "",
            day: 0,
            amount: None,
            numerator: None,
            denominator: None,
            gain: None,
        };
        match *action {
            Action::Dividend { day, amount } => Self {
                kind: "dividend",
                day,
                amount: Some(amount),
                ..empty
            },
            Action::Split {
                day,
                numerator,
                denominator,
            } => Self {
                kind: "split",
                day,
                numerator: Some(numerator),
                denominator: Some(denominator),
                ..empty
            },
            Action::CapitalGain { day, gain } => Self {
                kind: "capital_gain",
                day,
                gain: Some(gain),
                ..empty
            },
        }
    }
}

#[derive(ToDataFrame, Clone)]
#[df_derive(via = "InstrumentRow<'_>")]
struct Instrument {
    symbol: String,
    exchange: Option<String>,
    lot: u32,
    tick: f64,
}

#[derive(ToDataFrame)]
struct InstrumentRow<'a> {
    symbol: &'a str,
    exchange: Option<&'a str>,
    listed: bool,
    lot: u32,
    tick: f64,
}

impl<'a> From<&'a Instrument> for InstrumentRow<'a> {
    fn from(instrument: &'a Instrument) -> Self {
        Self {
            symbol: &instrument.symbol,
            exchange: instrument.exchange.as_deref(),
            listed: instrument.exchange.is_some(),
            lot: instrument.lot,
            tick: instrument.tick,
        }
    }
}

#[derive(ToDataFrame, Clone)]
struct History {
    id: u64,
    actions: Vec<Action>,
}

#[derive(ToDataFrame)]
struct HistoryRow {
    id: u64,
    actions: Vec<ActionRow>,
}

fn action(i: usize) -> Action {
    let day = i32::try_from(i % 20_000).unwrap();
    match i % 3 {
        0 => Action::Dividend { day, amount: 0.25 },
        1 => Action::Split {
            day,
            numerator: 2,
            denominator: 1,
        },
        _ => Action::CapitalGain { day, gain: 1.5 },
    }
}

fn benchmark_projection(c: &mut Criterion) {
    let actions: Vec<Action> = (0..N_ROWS).map(action).collect();
    c.bench_function("projection/enum_via", |b| {
        b.iter(|| std::hint::black_box(&actions).to_dataframe().unwrap());
    });
    c.bench_function("projection/enum_by_hand", |b| {
        b.iter(|| {
            let rows: Vec<ActionRow> = std::hint::black_box(&actions)
                .iter()
                .map(ActionRow::from)
                .collect();
            rows.to_dataframe().unwrap()
        });
    });

    let instruments: Vec<Instrument> = (0..N_ROWS)
        .map(|i| Instrument {
            symbol: format!("SYM{:04}", i % 1_000),
            exchange: (i % 4 != 0).then(|| format!("X{:02}", i % 50)),
            lot: 100,
            tick: 0.01,
        })
        .collect();
    c.bench_function("projection/borrowed_via", |b| {
        b.iter(|| std::hint::black_box(&instruments).to_dataframe().unwrap());
    });
    c.bench_function("projection/borrowed_by_hand", |b| {
        b.iter(|| {
            let rows: Vec<InstrumentRow<'_>> = std::hint::black_box(&instruments)
                .iter()
                .map(InstrumentRow::from)
                .collect();
            rows.to_dataframe().unwrap()
        });
    });

    let histories: Vec<History> = (0..N_ROWS / 4)
        .map(|i| History {
            id: i as u64,
            actions: (0..i % 5).map(|k| action(i + k)).collect(),
        })
        .collect();
    c.bench_function("projection/list_via", |b| {
        b.iter(|| std::hint::black_box(&histories).to_dataframe().unwrap());
    });
    c.bench_function("projection/list_by_hand", |b| {
        b.iter(|| {
            let rows: Vec<HistoryRow> = std::hint::black_box(&histories)
                .iter()
                .map(|history| HistoryRow {
                    id: history.id,
                    actions: history.actions.iter().map(ActionRow::from).collect(),
                })
                .collect();
            rows.to_dataframe().unwrap()
        });
    });
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = benchmark_projection
}
criterion_main!(benches);

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

/// One declaration for a row of many numbers and for a type with the same
/// fields that is encoded through that row.
macro_rules! snapshot {
    ($($field:ident: $ty:ty),+ $(,)?) => {
        #[derive(ToDataFrame, Clone)]
        struct SnapshotRow {
            $($field: $ty,)+
        }

        #[derive(ToDataFrame, Clone)]
        #[df_derive(via = "SnapshotRow")]
        struct Snapshot {
            $($field: $ty,)+
        }

        impl From<&Snapshot> for SnapshotRow {
            fn from(snapshot: &Snapshot) -> Self {
                Self {
                    $($field: snapshot.$field,)+
                }
            }
        }
    };
}

snapshot! {
    id: u64,
    at: i64,
    bid: f64,
    ask: f64,
    last: f64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    vwap: f64,
    bid_size: u32,
    ask_size: u32,
    last_size: u32,
    volume: u64,
    trades: u32,
    turnover: f64,
    delta: f64,
    gamma: f64,
    theta: f64,
    vega: f64,
    rho: f64,
    implied: Option<f64>,
    open_interest: Option<u64>,
    strike: f64,
}

fn snapshot(i: usize) -> Snapshot {
    let n = u32::try_from(i % 1_000_000).unwrap();
    let price = f64::from(n).mul_add(0.01, 100.0);
    Snapshot {
        id: i as u64,
        at: 1_700_000_000 + i64::from(n),
        bid: price - 0.01,
        ask: price + 0.01,
        last: price,
        open: price - 1.0,
        high: price + 2.0,
        low: price - 2.0,
        close: price + 0.5,
        vwap: price + 0.1,
        bid_size: 100 + n % 7,
        ask_size: 200 + n % 5,
        last_size: 1 + n % 3,
        volume: 10_000 + u64::from(n),
        trades: n % 500,
        turnover: price * 1_000.0,
        delta: 0.5,
        gamma: 0.01,
        theta: -0.02,
        vega: 0.3,
        rho: 0.05,
        implied: (!i.is_multiple_of(7)).then_some(0.25),
        open_interest: (!i.is_multiple_of(5)).then_some(1_000 + u64::from(n)),
        strike: 100.0,
    }
}

/// A type whose row borrows a list from it.
#[derive(ToDataFrame, Clone)]
#[df_derive(via = "BasketRow<'_>")]
struct Basket {
    id: u64,
    weights: Vec<f64>,
}

#[derive(ToDataFrame)]
struct BasketRow<'a> {
    id: u64,
    weights: &'a [f64],
}

impl<'a> From<&'a Basket> for BasketRow<'a> {
    fn from(basket: &'a Basket) -> Self {
        Self {
            id: basket.id,
            weights: &basket.weights,
        }
    }
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

    let snapshots: Vec<Snapshot> = (0..N_ROWS).map(snapshot).collect();
    c.bench_function("projection/wide_via", |b| {
        b.iter(|| std::hint::black_box(&snapshots).to_dataframe().unwrap());
    });
    c.bench_function("projection/wide_by_hand", |b| {
        b.iter(|| {
            let rows: Vec<SnapshotRow> = std::hint::black_box(&snapshots)
                .iter()
                .map(SnapshotRow::from)
                .collect();
            rows.to_dataframe().unwrap()
        });
    });
    // The rows themselves, already converted: what the columns cost alone.
    let rows: Vec<SnapshotRow> = snapshots.iter().map(SnapshotRow::from).collect();
    c.bench_function("projection/wide_rows", |b| {
        b.iter(|| std::hint::black_box(&rows).to_dataframe().unwrap());
    });

    for (rows, len) in [(N_ROWS, 4_usize), (N_ROWS, 16), (N_ROWS / 4, 64)] {
        let baskets: Vec<Basket> = (0..rows)
            .map(|i| Basket {
                id: i as u64,
                weights: vec![0.5; len],
            })
            .collect();
        c.bench_function(&format!("projection/inner_list_{len}_via"), |b| {
            b.iter(|| std::hint::black_box(&baskets).to_dataframe().unwrap());
        });
        c.bench_function(&format!("projection/inner_list_{len}_by_hand"), |b| {
            b.iter(|| {
                let rows: Vec<BasketRow<'_>> = std::hint::black_box(&baskets)
                    .iter()
                    .map(BasketRow::from)
                    .collect();
                rows.to_dataframe().unwrap()
            });
        });
    }
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = benchmark_projection
}
criterion_main!(benches);

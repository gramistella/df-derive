//! Design gate: the released derive ("old") against hand-written impls of the
//! builder runtime ("new") on identical structs and data.
//!
//! `encode_struct!` writes what the new derive would emit: a builder struct
//! with one member per field that forwards `push` / `finish` field by field.
//! Before timing, every shape asserts that both paths produce equal frames.

use chrono::{DateTime, TimeZone, Utc};
use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;
use df_derive::dataframe::{ToDataFrame as _, ToDataFrameVec as _};
use df_derive_core::encode::{
    self, ColumnBuilder, DatetimeBuilder, DecimalBuilder, Encode, Fields, Ms, OptionBuilder, child,
};
use polars::prelude::{Column, DataFrame, PlSmallStr, PolarsResult};
use polars_arrow::bitmap::Bitmap;
use rust_decimal::Decimal;
use std::hint::black_box;
use std::time::Duration;

macro_rules! encode_struct {
    ($ty:ident, $builder:ident { $($field:ident : $fty:ty $(=> $bty:ty)?),* $(,)? }) => {
        struct $builder {
            $($field: encode_struct!(@builder $fty $(=> $bty)?),)*
        }

        impl ColumnBuilder<$ty> for $builder {
            fn schema(name: PlSmallStr, out: &mut Fields) {
                $(
                    <encode_struct!(@builder $fty $(=> $bty)?) as ColumnBuilder<$fty>>::schema(
                        child(&name, stringify!($field)),
                        out,
                    );
                )*
            }

            fn with_capacity(rows: usize) -> Self {
                Self {
                    $(
                        $field: <encode_struct!(@builder $fty $(=> $bty)?) as ColumnBuilder<$fty>>::with_capacity(rows),
                    )*
                }
            }

            #[inline(always)]
            fn push(&mut self, value: &$ty) -> PolarsResult<()> {
                $(ColumnBuilder::<$fty>::push(&mut self.$field, &value.$field)?;)*
                Ok(())
            }

            #[inline(always)]
            fn push_null(&mut self) {
                $(ColumnBuilder::<$fty>::push_null(&mut self.$field);)*
            }

            fn finish(
                self,
                name: PlSmallStr,
                validity: Option<&Bitmap>,
                out: &mut Vec<Column>,
            ) -> PolarsResult<()> {
                $(
                    ColumnBuilder::<$fty>::finish(
                        self.$field,
                        child(&name, stringify!($field)),
                        validity,
                        out,
                    )?;
                )*
                Ok(())
            }
        }

        impl Encode for $ty {
            type Builder = $builder;
        }
    };
    (@builder $fty:ty) => { <$fty as Encode>::Builder };
    (@builder $fty:ty => $bty:ty) => { $bty };
}

fn u(i: usize) -> u32 {
    u32::try_from(i).unwrap()
}

fn assert_same(shape: &str, old: &DataFrame, new: &DataFrame) {
    assert_eq!(old.schema(), new.schema(), "{shape}: schema differs");
    assert!(old.equals_missing(new), "{shape}: values differ");
}

fn pair<Old, New>(c: &mut Criterion, shape: &str, old: Old, new: New)
where
    Old: Fn() -> DataFrame,
    New: Fn() -> DataFrame,
{
    assert_same(shape, &old(), &new());
    let mut group = c.benchmark_group(shape);
    group.bench_function("old", |b| b.iter(|| black_box(old())));
    group.bench_function("new", |b| b.iter(|| black_box(new())));
    group.finish();
}

// --- flat numeric (bench 01) -------------------------------------------------

#[derive(ToDataFrame, Clone)]
struct Tick {
    ts: i64,
    price: f64,
    volume: u64,
    bid: f64,
    ask: f64,
    bid_size: u32,
}
encode_struct!(
    Tick,
    TickBuilder {
        ts: i64,
        price: f64,
        volume: u64,
        bid: f64,
        ask: f64,
        bid_size: u32
    }
);

fn ticks(n: usize) -> Vec<Tick> {
    (0..n)
        .map(|i| Tick {
            ts: 1_700_000_000 + i64::try_from(i).unwrap(),
            price: f64::from(u(i)).mul_add(0.001, 100.0),
            volume: 1_000 + (i as u64),
            bid: f64::from(u(i)).mul_add(0.001, 99.9),
            ask: f64::from(u(i)).mul_add(0.001, 100.1),
            bid_size: 10 + (u(i) % 100),
        })
        .collect()
}

// --- nested options with strings (bench 03) ----------------------------------

#[derive(ToDataFrame, Clone)]
struct Address {
    street: String,
    city: String,
    zip: String,
}
encode_struct!(
    Address,
    AddressBuilder {
        street: String,
        city: String,
        zip: String
    }
);

#[derive(ToDataFrame, Clone)]
struct Profile {
    age: i32,
    email: String,
    address: Option<Address>,
}
encode_struct!(Profile, ProfileBuilder { age: i32, email: String, address: Option<Address> });

#[derive(ToDataFrame, Clone)]
struct User {
    id: u64,
    name: String,
    profile: Option<Profile>,
}
encode_struct!(User, UserBuilder { id: u64, name: String, profile: Option<Profile> });

fn users(n: usize) -> Vec<User> {
    (0..n)
        .map(|i| User {
            id: i as u64,
            name: format!("user-{i}"),
            profile: if i % 3 == 0 {
                None
            } else {
                Some(Profile {
                    age: 18 + (i32::try_from(i).unwrap() % 60),
                    email: format!("user{i}@example.com"),
                    address: if i % 5 == 0 {
                        None
                    } else {
                        Some(Address {
                            street: format!("{i} Main St"),
                            city: "Metropolis".to_string(),
                            zip: format!("{:05}", i % 100_000),
                        })
                    },
                })
            },
        })
        .collect()
}

// --- one row holding a long Vec<struct> (bench 04) ---------------------------

#[derive(ToDataFrame)]
struct Quote {
    timestamp: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: u64,
}
encode_struct!(
    Quote,
    QuoteBuilder {
        timestamp: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: u64
    }
);

#[derive(ToDataFrame)]
struct MarketData {
    symbol: String,
    quotes: Vec<Quote>,
}
encode_struct!(MarketData, MarketDataBuilder { symbol: String, quotes: Vec<Quote> });

fn market_data(n: usize) -> MarketData {
    MarketData {
        symbol: "BENCH".to_string(),
        quotes: (0..n)
            .map(|i| {
                let f = f64::from(u(i));
                Quote {
                    timestamp: 1_700_000_000 + i64::try_from(i).unwrap(),
                    open: f.mul_add(0.1, 100.0),
                    high: f.mul_add(0.1, 102.0),
                    low: f.mul_add(0.1, 99.5),
                    close: f.mul_add(0.1, 101.0),
                    volume: 1000 + i as u64,
                }
            })
            .collect(),
    }
}

// --- wide optional scalars (bench 05) ----------------------------------------

#[derive(ToDataFrame, Clone)]
struct WideTick {
    ts: i64,
    price: Option<f64>,
    volume: Option<u64>,
    bid: Option<f64>,
    ask: Option<f64>,
    bid_size: Option<u32>,
    ask_size: Option<u32>,
    trade_id: Option<String>,
    venue: Option<String>,
    flag_a: Option<bool>,
    flag_b: Option<bool>,
}
encode_struct!(WideTick, WideTickBuilder {
    ts: i64,
    price: Option<f64>,
    volume: Option<u64>,
    bid: Option<f64>,
    ask: Option<f64>,
    bid_size: Option<u32>,
    ask_size: Option<u32>,
    trade_id: Option<String>,
    venue: Option<String>,
    flag_a: Option<bool>,
    flag_b: Option<bool>,
});

fn wide_ticks(n: usize) -> Vec<WideTick> {
    (0..n)
        .map(|i| WideTick {
            ts: 1_700_000_000 + i64::try_from(i).unwrap(),
            price: (i % 10 != 0).then(|| f64::from(u(i)).mul_add(0.01, 100.0)),
            volume: (i % 13 != 0).then(|| 1_000 + i as u64),
            bid: (i % 7 != 0).then(|| f64::from(u(i)).mul_add(0.01, 99.5)),
            ask: (i % 11 != 0).then(|| f64::from(u(i)).mul_add(0.01, 100.5)),
            bid_size: (i % 9 != 0).then(|| 10 + (u(i) % 100)),
            ask_size: (i % 8 != 0).then(|| 10 + (u(i) % 100)),
            trade_id: (i % 6 != 0).then(|| format!("T-{i:08}")),
            venue: (i % 5 != 0).then(|| "XNYS".to_string()),
            flag_a: (i % 4 != 0).then_some(i % 2 == 0),
            flag_b: (i % 3 != 0).then_some(i % 2 == 1),
        })
        .collect()
}

// --- strings (bench 10) ------------------------------------------------------

#[derive(ToDataFrame, Clone)]
struct StringRow {
    symbol: String,
    venue: String,
    side: String,
    user_id: String,
    note: String,
}
encode_struct!(
    StringRow,
    StringRowBuilder {
        symbol: String,
        venue: String,
        side: String,
        user_id: String,
        note: String
    }
);

#[derive(ToDataFrame, Clone)]
struct OptStringRow {
    symbol: Option<String>,
    venue: Option<String>,
    side: Option<String>,
    user_id: Option<String>,
    note: Option<String>,
}
encode_struct!(OptStringRow, OptStringRowBuilder {
    symbol: Option<String>,
    venue: Option<String>,
    side: Option<String>,
    user_id: Option<String>,
    note: Option<String>,
});

fn string_rows(n: usize) -> Vec<StringRow> {
    (0..n)
        .map(|i| StringRow {
            symbol: format!("SYM{:04}", i % 1_000),
            venue: format!("V{:02}", i % 100),
            side: if i % 2 == 0 { "BUY" } else { "SELL" }.to_string(),
            user_id: format!("user-{}", i % 5_000),
            note: format!("trade-{i}-some-context-payload"),
        })
        .collect()
}

fn opt_string_rows(n: usize) -> Vec<OptStringRow> {
    (0..n)
        .map(|i| OptStringRow {
            symbol: (i % 11 != 0).then(|| format!("SYM{:04}", i % 1_000)),
            venue: (i % 7 != 0).then(|| format!("V{:02}", i % 100)),
            side: (i % 5 != 0).then(|| if i % 2 == 0 { "BUY" } else { "SELL" }.to_string()),
            user_id: (i % 4 != 0).then(|| format!("user-{}", i % 5_000)),
            note: (i % 3 != 0).then(|| format!("trade-{i}-some-context-payload")),
        })
        .collect()
}

// --- Option<Vec<Option<struct>>> (bench 16) and Vec<Vec<struct>> (bench 17) ---

#[derive(ToDataFrame, Clone)]
struct Inner {
    code: u32,
    label: String,
    ratio: f64,
}
encode_struct!(
    Inner,
    InnerBuilder {
        code: u32,
        label: String,
        ratio: f64
    }
);

#[derive(ToDataFrame, Clone)]
struct Outer {
    id: u64,
    name: Option<String>,
    items: Option<Vec<Option<Inner>>>,
}
encode_struct!(Outer, OuterBuilder { id: u64, name: Option<String>, items: Option<Vec<Option<Inner>>> });

fn outers(n: usize) -> Vec<Outer> {
    (0..n)
        .map(|i| Outer {
            id: i as u64,
            name: (i % 7 != 0).then(|| format!("outer-{i}")),
            items: (i % 10 >= 3).then(|| {
                (0..i % 6)
                    .map(|k| {
                        ((k + i) % 5 != 0).then(|| Inner {
                            code: u(i * 10 + k),
                            label: format!("lbl-{i}-{k}"),
                            ratio: f64::from(u(k)).mul_add(0.125, 1.0),
                        })
                    })
                    .collect()
            }),
        })
        .collect()
}

#[derive(ToDataFrame, Clone)]
struct Container {
    id: u64,
    items: Vec<Vec<Inner>>,
}
encode_struct!(Container, ContainerBuilder { id: u64, items: Vec<Vec<Inner>> });

fn containers(n: usize) -> Vec<Container> {
    (0..n)
        .map(|i| Container {
            id: i as u64,
            items: (0..i % 6)
                .map(|j| {
                    (0..(i + j) % 6)
                        .map(|k| Inner {
                            code: u(i * 100 + j * 10 + k),
                            label: format!("lbl-{i}-{j}-{k}"),
                            ratio: f64::from(u(k)).mul_add(0.125, 1.0),
                        })
                        .collect()
                })
                .collect(),
        })
        .collect()
}

// --- datetime and decimal (bench 13) -----------------------------------------

#[derive(ToDataFrame, Clone)]
struct TimedTick {
    #[df_derive(time_unit = "ms")]
    ts: DateTime<Utc>,
    #[df_derive(time_unit = "ms")]
    ts_opt: Option<DateTime<Utc>>,
    #[df_derive(decimal(precision = 18, scale = 6))]
    price: Decimal,
    #[df_derive(decimal(precision = 18, scale = 6))]
    price_opt: Option<Decimal>,
    volume: u64,
}
encode_struct!(TimedTick, TimedTickBuilder {
    ts: DateTime<Utc> => DatetimeBuilder<Ms>,
    ts_opt: Option<DateTime<Utc>> => OptionBuilder<DatetimeBuilder<Ms>>,
    price: Decimal => DecimalBuilder<18, 6>,
    price_opt: Option<Decimal> => OptionBuilder<DecimalBuilder<18, 6>>,
    volume: u64,
});

fn timed_ticks(n: usize) -> Vec<TimedTick> {
    let base = Utc
        .timestamp_millis_opt(1_700_000_000_000)
        .single()
        .unwrap();
    (0..n)
        .map(|i| {
            let k = i64::try_from(i).unwrap();
            let ts = base + chrono::Duration::milliseconds(k * 100);
            TimedTick {
                ts,
                ts_opt: (i % 7 != 0).then_some(ts),
                price: Decimal::new(1_000_000 + k * 13, 6),
                price_opt: (i % 5 != 0).then(|| Decimal::new(2_000_000 + k * 11, 6)),
                volume: 1_000 + (i as u64),
            }
        })
        .collect()
}

// --- primitive and string lists ----------------------------------------------

#[derive(ToDataFrame, Clone)]
struct Lists {
    id: u64,
    xs: Vec<f64>,
    tags: Vec<String>,
    grid: Vec<Vec<i32>>,
}
encode_struct!(Lists, ListsBuilder { id: u64, xs: Vec<f64>, tags: Vec<String>, grid: Vec<Vec<i32>> });

fn lists(n: usize) -> Vec<Lists> {
    (0..n)
        .map(|i| Lists {
            id: i as u64,
            xs: (0..16).map(|k| f64::from(u(i + k)) * 0.5).collect(),
            tags: (0..i % 4).map(|k| format!("tag-{k}")).collect(),
            grid: (0..4)
                .map(|r| {
                    (0..4)
                        .map(|k| i32::try_from((i + r * 4 + k) % 1000).unwrap())
                        .collect()
                })
                .collect(),
        })
        .collect()
}

// --- harness -----------------------------------------------------------------

#[allow(clippy::too_many_lines)]
fn gate(c: &mut Criterion) {
    let data = ticks(100_000);
    pair(
        c,
        "flat_numeric",
        || data.as_slice().to_dataframe().unwrap(),
        || encode::to_dataframe(&data).unwrap(),
    );

    let one = &data[0];
    pair(
        c,
        "single_row",
        || one.to_dataframe().unwrap(),
        || encode::to_dataframe([one]).unwrap(),
    );

    let data = users(50_000);
    pair(
        c,
        "nested_option",
        || data.as_slice().to_dataframe().unwrap(),
        || encode::to_dataframe(&data).unwrap(),
    );

    let data = market_data(100_000);
    pair(
        c,
        "vec_struct_one_row",
        || data.to_dataframe().unwrap(),
        || encode::to_dataframe([&data]).unwrap(),
    );

    let data = wide_ticks(50_000);
    pair(
        c,
        "wide_options",
        || data.as_slice().to_dataframe().unwrap(),
        || encode::to_dataframe(&data).unwrap(),
    );

    let data = string_rows(100_000);
    pair(
        c,
        "strings_required",
        || data.as_slice().to_dataframe().unwrap(),
        || encode::to_dataframe(&data).unwrap(),
    );

    let data = opt_string_rows(100_000);
    pair(
        c,
        "strings_optional",
        || data.as_slice().to_dataframe().unwrap(),
        || encode::to_dataframe(&data).unwrap(),
    );

    let data = outers(30_000);
    pair(
        c,
        "option_vec_option_struct",
        || data.as_slice().to_dataframe().unwrap(),
        || encode::to_dataframe(&data).unwrap(),
    );

    let data = containers(10_000);
    pair(
        c,
        "vec_vec_struct",
        || data.as_slice().to_dataframe().unwrap(),
        || encode::to_dataframe(&data).unwrap(),
    );

    let data = timed_ticks(100_000);
    pair(
        c,
        "datetime_decimal",
        || data.as_slice().to_dataframe().unwrap(),
        || encode::to_dataframe(&data).unwrap(),
    );

    let data = lists(50_000);
    pair(
        c,
        "primitive_lists",
        || data.as_slice().to_dataframe().unwrap(),
        || encode::to_dataframe(&data).unwrap(),
    );
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(5))
        .noise_threshold(0.02)
        .confidence_level(0.99);
    targets = gate
}
criterion_main!(benches);

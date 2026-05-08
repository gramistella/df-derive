use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;

#[path = "support/mod.rs"]
mod bench_support;
#[path = "../tests/common.rs"]
mod core;
use crate::bench_support::configure_criterion;
use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};

use chrono::{NaiveDate, NaiveTime};

const N_ROWS: usize = 100_000;

#[derive(ToDataFrame, Clone)]
struct Sample {
    nd: NaiveDate,
    nd_opt: Option<NaiveDate>,
    nd_vec: Vec<NaiveDate>,

    nt: NaiveTime,

    cd: chrono::Duration,
    sd: std::time::Duration,

    #[df_derive(time_unit = "ms")]
    cd_ms: chrono::Duration,
}

fn generate() -> Vec<Sample> {
    let epoch = NaiveDate::from_ymd_opt(2000, 1, 1).unwrap();
    (0..N_ROWS)
        .map(|i| {
            let date = epoch + chrono::Duration::days(i64::try_from(i).unwrap() % 10_000);
            let secs = (i % 86_400) as u32;
            let time = NaiveTime::from_num_seconds_from_midnight_opt(secs, 0).unwrap();
            Sample {
                nd: date,
                nd_opt: if i % 5 == 0 { None } else { Some(date) },
                nd_vec: vec![date, date, date],
                nt: time,
                cd: chrono::Duration::nanoseconds(i64::try_from(i).unwrap()),
                sd: std::time::Duration::from_nanos(i as u64),
                cd_ms: chrono::Duration::milliseconds(i64::try_from(i).unwrap()),
            }
        })
        .collect()
}

fn bench_to_dataframe(c: &mut Criterion) {
    let data = generate();

    c.bench_function("naive_and_duration_columnar", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&data).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });

    let single = data[0].clone();
    c.bench_function("naive_and_duration_single_row", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&single).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = bench_to_dataframe
}
criterion_main!(benches);

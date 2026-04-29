use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;
use std::time::Duration;

#[path = "../tests/common.rs"]
mod core;
use crate::core::dataframe::ToDataFrameVec;

const N_ROWS_REQUIRED: usize = 100_000;
const N_ROWS_OPTIONAL: usize = 100_000;

#[derive(ToDataFrame, Clone)]
struct StringRowRequired {
    symbol: String,
    venue: String,
    side: String,
    user_id: String,
    note: String,
}

#[derive(ToDataFrame, Clone)]
struct StringRowOptional {
    symbol: Option<String>,
    venue: Option<String>,
    side: Option<String>,
    user_id: Option<String>,
    note: Option<String>,
}

fn generate_required() -> Vec<StringRowRequired> {
    (0..N_ROWS_REQUIRED)
        .map(|i| StringRowRequired {
            symbol: format!("SYM{:04}", i % 1_000),
            venue: format!("V{:02}", i % 100),
            side: if i % 2 == 0 { "BUY" } else { "SELL" }.to_string(),
            user_id: format!("user-{}", i % 5_000),
            note: format!("trade-{i}-some-context-payload"),
        })
        .collect()
}

fn generate_optional() -> Vec<StringRowOptional> {
    (0..N_ROWS_OPTIONAL)
        .map(|i| StringRowOptional {
            symbol: if i % 11 == 0 {
                None
            } else {
                Some(format!("SYM{:04}", i % 1_000))
            },
            venue: if i % 7 == 0 {
                None
            } else {
                Some(format!("V{:02}", i % 100))
            },
            side: if i % 5 == 0 {
                None
            } else {
                Some(if i % 2 == 0 { "BUY" } else { "SELL" }.to_string())
            },
            user_id: if i % 4 == 0 {
                None
            } else {
                Some(format!("user-{}", i % 5_000))
            },
            note: if i % 3 == 0 {
                None
            } else {
                Some(format!("trade-{i}-some-context-payload"))
            },
        })
        .collect()
}

fn benchmark_string_columns(c: &mut Criterion) {
    let required = generate_required();
    c.bench_function("string_columns_required", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&required).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });

    let optional = generate_optional();
    c.bench_function("string_columns_optional", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&optional).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });
}

fn configure_criterion() -> Criterion {
    Criterion::default()
        .sample_size(100)
        .warm_up_time(Duration::from_secs(5))
        .measurement_time(Duration::from_secs(15))
        .nresamples(200_000)
        .noise_threshold(0.02)
        .confidence_level(0.99)
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = benchmark_string_columns
}
criterion_main!(benches);

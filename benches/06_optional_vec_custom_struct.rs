use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;

#[path = "../tests/common.rs"]
mod core;
use crate::core::dataframe::ToDataFrame;
use crate::core::dataframe::ToDataFrameVec;

const N_USERS: usize = 30_000;

#[derive(ToDataFrame, Clone)]
struct Holding {
    symbol: String,
    shares: f64,
    avg_cost: f64,
}

#[derive(ToDataFrame, Clone)]
struct Portfolio {
    name: String,
    holdings: Option<Vec<Holding>>,
}

#[derive(ToDataFrame, Clone)]
struct Investor {
    id: u64,
    portfolio: Portfolio,
}

fn generate_investors() -> Vec<Investor> {
    (0..N_USERS)
        .map(|i| {
            let holdings = if i % 4 == 0 {
                None
            } else {
                Some(
                    (0..=(i % 5))
                        .map(|k| Holding {
                            symbol: format!("SYM{}{}", i % 50, k),
                            shares: f64::from(u32::try_from(k).unwrap()).mul_add(2.5, 10.0),
                            avg_cost: f64::from(u32::try_from(k).unwrap()).mul_add(1.25, 100.0),
                        })
                        .collect(),
                )
            };
            Investor {
                id: i as u64,
                portfolio: Portfolio {
                    name: format!("P{i}"),
                    holdings,
                },
            }
        })
        .collect()
}

fn benchmark_optional_vec_custom_struct(c: &mut Criterion) {
    let data = generate_investors();

    c.bench_function("optional_vec_custom_struct_conversion", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&data).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });
}

criterion_group!(benches, benchmark_optional_vec_custom_struct);
criterion_main!(benches);

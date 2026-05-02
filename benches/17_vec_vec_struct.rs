use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;

#[path = "support/mod.rs"]
mod bench_support;
#[path = "../tests/common.rs"]
mod core;
use crate::bench_support::configure_criterion;
use crate::core::dataframe::ToDataFrameVec;

const N_OUTER: usize = 10_000;

#[derive(ToDataFrame, Clone)]
struct Inner {
    code: u32,
    label: String,
    ratio: f64,
}

#[derive(ToDataFrame, Clone)]
struct Container {
    id: u64,
    items: Vec<Vec<Inner>>,
}

fn generate_containers() -> Vec<Container> {
    (0..N_OUTER)
        .map(|i| {
            let outer_len = i % 6;
            let items = (0..outer_len)
                .map(|j| {
                    let inner_len = (i + j) % 6;
                    (0..inner_len)
                        .map(|k| Inner {
                            code: u32::try_from(i * 100 + j * 10 + k).unwrap(),
                            label: format!("lbl-{i}-{j}-{k}"),
                            ratio: f64::from(u32::try_from(k).unwrap()).mul_add(0.125, 1.0),
                        })
                        .collect()
                })
                .collect();
            Container {
                id: i as u64,
                items,
            }
        })
        .collect()
}

fn benchmark_vec_vec_struct(c: &mut Criterion) {
    let data = generate_containers();

    c.bench_function("vec_vec_struct_conversion", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&data).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = benchmark_vec_vec_struct
}
criterion_main!(benches);

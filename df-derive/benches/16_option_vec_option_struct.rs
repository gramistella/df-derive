use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;

#[path = "support/mod.rs"]
mod bench_support;
#[path = "../tests/common.rs"]
mod core;
use crate::bench_support::configure_criterion;
use crate::core::dataframe::ToDataFrameVec;

const N_OUTER: usize = 30_000;

#[derive(ToDataFrame, Clone)]
struct Inner {
    code: u32,
    label: String,
    ratio: f64,
}

#[derive(ToDataFrame, Clone)]
struct Outer {
    id: u64,
    name: Option<String>,
    items: Option<Vec<Option<Inner>>>,
}

fn generate_outers() -> Vec<Outer> {
    (0..N_OUTER)
        .map(|i| {
            let items = if i % 10 < 3 {
                None
            } else {
                let len = i % 6;
                Some(
                    (0..len)
                        .map(|k| {
                            if (k + i) % 5 == 0 {
                                None
                            } else {
                                Some(Inner {
                                    code: u32::try_from(i * 10 + k).unwrap(),
                                    label: format!("lbl-{i}-{k}"),
                                    ratio: f64::from(u32::try_from(k).unwrap()).mul_add(0.125, 1.0),
                                })
                            }
                        })
                        .collect(),
                )
            };
            Outer {
                id: i as u64,
                name: if i % 7 == 0 {
                    None
                } else {
                    Some(format!("outer-{i}"))
                },
                items,
            }
        })
        .collect()
}

fn benchmark_option_vec_option_struct(c: &mut Criterion) {
    let data = generate_outers();

    c.bench_function("option_vec_option_struct_conversion", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&data).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = benchmark_option_vec_option_struct
}
criterion_main!(benches);

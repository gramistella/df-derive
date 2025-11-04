use criterion::{Criterion, criterion_group, criterion_main};
use std::time::Duration;
use df_derive::ToDataFrame;

#[path = "../tests/common.rs"]
mod core;
use crate::core::dataframe::ToDataFrame;
use crate::core::dataframe::ToDataFrameVec;

const N_CONTAINERS: usize = 30_000;

#[derive(ToDataFrame, Clone)]
struct Item {
    id: u32,
    name: String,
}

#[derive(ToDataFrame, Clone)]
struct Container {
    id: i32,
    // A vec of optional primitives
    primitive_items: Vec<Option<i32>>,
    // A vec of optional custom structs
    custom_items: Vec<Option<Item>>,
    // An optional vec of optional primitives
    opt_vec_opt_primitive: Option<Vec<Option<i32>>>,
}

fn generate_containers() -> Vec<Container> {
    (0..N_CONTAINERS)
        .map(|i| {
            let primitive_items: Vec<Option<i32>> = (0..(i % 7 + 3))
                .map(|k| {
                    if (k + i) % 3 == 0 {
                        None
                    } else {
                        Some(i32::try_from(k).unwrap() + 1)
                    }
                })
                .collect();

            let custom_items: Vec<Option<Item>> = (0..(i % 5 + 2))
                .map(|k| {
                    if (k + 2 * i) % 4 == 0 {
                        None
                    } else {
                        Some(Item {
                            id: 1000
                                + (u32::try_from(i).unwrap() * 10)
                                + (u32::try_from(k).unwrap()),
                            name: format!("item-{i}-{k}"),
                        })
                    }
                })
                .collect();

            let opt_vec_opt_primitive = if i % 2 == 0 {
                Some(
                    (0..=(i % 6))
                        .map(|k| {
                            if k % 2 == 1 {
                                None
                            } else {
                                Some(i32::try_from(k).unwrap() * 2)
                            }
                        })
                        .collect(),
                )
            } else {
                None
            };

            Container {
                id: i32::try_from(i).unwrap(),
                primitive_items,
                custom_items,
                opt_vec_opt_primitive,
            }
        })
        .collect()
}

fn benchmark_complex_wrappers(c: &mut Criterion) {
    let data = generate_containers();

    c.bench_function("complex_wrappers_conversion", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&data).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });
}
fn configure_criterion() -> Criterion {
    Criterion::default()
        .sample_size(150)
        .warm_up_time(Duration::from_secs(8))
        .measurement_time(Duration::from_secs(20))
        .nresamples(200_000)
        .noise_threshold(0.02)
        .confidence_level(0.99)
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = benchmark_complex_wrappers
}
criterion_main!(benches);

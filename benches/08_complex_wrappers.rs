use criterion::{Criterion, black_box, criterion_group, criterion_main};
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
                        Some((k as i32) + 1)
                    }
                })
                .collect();

            let custom_items: Vec<Option<Item>> = (0..(i % 5 + 2))
                .map(|k| {
                    if (k + 2 * i) % 4 == 0 {
                        None
                    } else {
                        Some(Item {
                            id: (1000 + (i as u32) * 10 + (k as u32)),
                            name: format!("item-{}-{}", i, k),
                        })
                    }
                })
                .collect();

            let opt_vec_opt_primitive = if i % 2 == 0 {
                Some(
                    (0..(i % 6 + 1))
                        .map(|k| {
                            if k % 2 == 1 {
                                None
                            } else {
                                Some((k as i32) * 2)
                            }
                        })
                        .collect(),
                )
            } else {
                None
            };

            Container {
                id: i as i32,
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
            let df = black_box(&data).to_dataframe().unwrap();
            black_box(df)
        })
    });
}

criterion_group!(benches, benchmark_complex_wrappers);
criterion_main!(benches);

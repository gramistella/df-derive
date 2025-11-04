use criterion::{Criterion, criterion_group, criterion_main};
use std::time::Duration;
use df_derive::ToDataFrame;

#[path = "../tests/common.rs"]
mod core;
use crate::core::dataframe::ToDataFrame;
use crate::core::dataframe::ToDataFrameVec;

const N_USERS: usize = 20_000;
const N_ORDERS_PER_USER: usize = 5;

#[derive(ToDataFrame, Clone)]
struct Item {
    sku: String,
    price: f64,
    qty: u32,
}

#[derive(ToDataFrame, Clone)]
struct Order {
    order_id: u64,
    items: Vec<Item>,
}

#[derive(ToDataFrame, Clone)]
struct User {
    user_id: u64,
    orders: Vec<Order>,
}

fn generate_users() -> Vec<User> {
    (0..N_USERS)
        .map(|u| {
            let orders: Vec<Order> = (0..N_ORDERS_PER_USER)
                .map(|k| {
                    let base = (u * 100 + k) as u64;
                    let items: Vec<Item> = (0..(k % 7 + 3))
                        .map(|i| Item {
                            sku: format!("SKU-{u}-{k}-{i}"),
                            price: f64::from(u32::try_from(i).unwrap()).mul_add(0.5, 10.0),
                            qty: 1 + (u32::try_from(i).unwrap() % 4),
                        })
                        .collect();
                    Order {
                        order_id: base,
                        items,
                    }
                })
                .collect();
            User {
                user_id: u as u64,
                orders,
            }
        })
        .collect()
}

fn benchmark_deep_nesting_vec(c: &mut Criterion) {
    let users = generate_users();

    c.bench_function("deep_nesting_vec_conversion", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&users).to_dataframe().unwrap();
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
    targets = benchmark_deep_nesting_vec
}
criterion_main!(benches);

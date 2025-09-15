use criterion::{Criterion, black_box, criterion_group, criterion_main};
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
                            sku: format!("SKU-{}-{}-{}", u, k, i),
                            price: 10.0 + (i as f64) * 0.5,
                            qty: 1 + (i as u32 % 4),
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
            let df = black_box(&users).to_dataframe().unwrap();
            black_box(df)
        })
    });
}

criterion_group!(benches, benchmark_deep_nesting_vec);
criterion_main!(benches);

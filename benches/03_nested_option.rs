use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;

#[path = "../tests/common.rs"]
mod core;
use crate::core::dataframe::ToDataFrame;
use crate::core::dataframe::ToDataFrameVec;

const N_USERS: usize = 50_000;

#[derive(ToDataFrame, Clone)]
struct Address {
    street: String,
    city: String,
    zip: String,
}

#[derive(ToDataFrame, Clone)]
struct Profile {
    age: i32,
    email: String,
    address: Option<Address>,
}

#[derive(ToDataFrame, Clone)]
struct User {
    id: u64,
    name: String,
    profile: Option<Profile>,
}

fn generate_users() -> Vec<User> {
    (0..N_USERS)
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

fn benchmark_nested_option(c: &mut Criterion) {
    let users = generate_users();

    c.bench_function("nested_option_conversion", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&users).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });
}

criterion_group!(benches, benchmark_nested_option);
criterion_main!(benches);

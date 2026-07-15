use std::hint::black_box;

use df_derive::ToDataFrame;
use gungraun::prelude::*;

#[path = "support/tuple_replay_boundary.rs"]
mod tuple_replay_boundary;

use crate::tuple_replay_boundary::{
    TupleReplayBoundary, TupleReplayBoundaryMinusOne, make_tuple_replay_boundary,
    make_tuple_replay_boundary_minus_one,
};
use df_derive::dataframe::{Columnar, ToDataFrameVec};

const N_NUMERIC_ROWS: usize = 10_000;
const N_NESTED_USERS: usize = 5_000;
const N_STRING_ROWS: usize = 10_000;
const N_BINARY_ROWS: usize = 2_000;
const TINY_LEN: usize = 8;
const MEDIUM_LEN: usize = 1024;

#[derive(ToDataFrame)]
struct Tick {
    ts: i64,
    price: f64,
    volume: u64,
    bid: f64,
    ask: f64,
    bid_size: u32,
}

#[derive(ToDataFrame)]
struct Address {
    street: String,
    city: String,
    zip: String,
}

#[derive(ToDataFrame)]
struct Profile {
    age: i32,
    email: String,
    address: Option<Address>,
}

#[derive(ToDataFrame)]
struct User {
    id: u64,
    name: String,
    profile: Option<Profile>,
}

#[derive(ToDataFrame)]
struct StringRowRequired {
    symbol: String,
    venue: String,
    side: String,
    user_id: String,
    note: String,
}

#[derive(ToDataFrame)]
struct StringRowOptional {
    symbol: Option<String>,
    venue: Option<String>,
    side: Option<String>,
    user_id: Option<String>,
    note: Option<String>,
}

#[derive(ToDataFrame)]
struct BinaryTinyRow {
    #[df_derive(as_binary)]
    bytes: Vec<u8>,
}

#[derive(ToDataFrame)]
struct BinaryMediumRow {
    #[df_derive(as_binary)]
    bytes: Vec<u8>,
}

#[derive(ToDataFrame)]
struct FlatScalarControl {
    a: i64,
    b: i64,
    c: i64,
    d: i64,
    e: i64,
    f: i64,
    g: i64,
    h: i64,
}

#[derive(ToDataFrame)]
struct TupleScalarRow {
    values: (i64, i64, i64, i64, i64, i64, i64, i64),
}

#[derive(ToDataFrame)]
struct WideTupleScalarRow {
    t0: (i64, i64, i64, i64),
    t1: (i64, i64, i64, i64),
    t2: (i64, i64, i64, i64),
    t3: (i64, i64, i64, i64),
    t4: (i64, i64, i64, i64),
    t5: (i64, i64, i64, i64),
    t6: (i64, i64, i64, i64),
    t7: (i64, i64, i64, i64),
}

#[derive(ToDataFrame)]
struct ListQuad {
    a: i64,
    b: i64,
    c: i64,
    d: i64,
}

#[derive(ToDataFrame)]
struct NestedStructListControl {
    items: Vec<ListQuad>,
}

#[derive(ToDataFrame)]
struct NestedTupleReplayBoundary {
    nested: TupleReplayBoundary,
}

#[derive(ToDataFrame)]
struct TupleListRow {
    items: Vec<(i64, i64, i64, i64)>,
}

#[derive(ToDataFrame)]
struct VecBoolRow {
    id: u64,
    items: Vec<bool>,
}

#[derive(ToDataFrame)]
struct VecOptBoolRow {
    id: u64,
    items: Vec<Option<bool>>,
}

#[derive(ToDataFrame)]
struct VecI32Row {
    id: u64,
    items: Vec<i32>,
}

#[derive(ToDataFrame)]
struct VecOptI32Row {
    id: u64,
    items: Vec<Option<i32>>,
}

#[derive(ToDataFrame)]
struct VecVecBoolRow {
    id: u64,
    items: Vec<Vec<bool>>,
}

#[derive(ToDataFrame)]
struct VecVecI32Row {
    id: u64,
    items: Vec<Vec<i32>>,
}

#[derive(ToDataFrame)]
struct VecVecOptI32Row {
    id: u64,
    items: Vec<Vec<Option<i32>>>,
}

fn convert_rows<T>(rows: Vec<T>) -> (usize, usize)
where
    [T]: ToDataFrameVec,
{
    let df = black_box(rows.as_slice()).to_dataframe().unwrap();
    let shape = black_box(df).shape();

    // Criterion keeps benchmark fixtures alive outside each iteration; do the
    // same here so the one-shot count is focused on conversion, not fixture drop.
    std::mem::forget(rows);

    shape
}

fn convert_rows_streaming<T>(rows: Vec<T>) -> (usize, usize)
where
    T: Columnar,
{
    let df = <T as Columnar>::encode(black_box(rows.iter())).unwrap();
    let shape = black_box(df).shape();

    std::mem::forget(rows);

    shape
}

fn generate_ticks() -> Vec<Tick> {
    (0..N_NUMERIC_ROWS)
        .map(|i| Tick {
            ts: 1_700_000_000 + i64::try_from(i).unwrap(),
            price: f64::from(u32::try_from(i).unwrap()).mul_add(0.001, 100.0),
            volume: 1_000 + (i as u64),
            bid: f64::from(u32::try_from(i).unwrap()).mul_add(0.001, 99.9),
            ask: f64::from(u32::try_from(i).unwrap()).mul_add(0.001, 100.1),
            bid_size: 10 + (u32::try_from(i).unwrap() % 100),
        })
        .collect()
}

fn generate_users() -> Vec<User> {
    (0..N_NESTED_USERS)
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

fn generate_required_strings() -> Vec<StringRowRequired> {
    (0..N_STRING_ROWS)
        .map(|i| StringRowRequired {
            symbol: format!("SYM{:04}", i % 1_000),
            venue: format!("V{:02}", i % 100),
            side: if i % 2 == 0 { "BUY" } else { "SELL" }.to_string(),
            user_id: format!("user-{}", i % 5_000),
            note: format!("trade-{i}-some-context-payload"),
        })
        .collect()
}

fn generate_optional_strings() -> Vec<StringRowOptional> {
    (0..N_STRING_ROWS)
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

fn make_payload(seed: usize, len: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(len);
    for i in 0..len {
        buf.push(u8::try_from((seed.wrapping_add(i)) & 0xff).unwrap());
    }
    buf
}

fn generate_binary_tiny() -> Vec<BinaryTinyRow> {
    (0..N_BINARY_ROWS)
        .map(|i| BinaryTinyRow {
            bytes: make_payload(i, TINY_LEN),
        })
        .collect()
}

fn generate_binary_medium() -> Vec<BinaryMediumRow> {
    (0..N_BINARY_ROWS)
        .map(|i| BinaryMediumRow {
            bytes: make_payload(i, MEDIUM_LEN),
        })
        .collect()
}

fn generate_flat_scalar_control() -> Vec<FlatScalarControl> {
    (0..N_NUMERIC_ROWS)
        .map(|i| {
            let base = i64::try_from(i).unwrap() * 8;
            FlatScalarControl {
                a: base,
                b: base + 1,
                c: base + 2,
                d: base + 3,
                e: base + 4,
                f: base + 5,
                g: base + 6,
                h: base + 7,
            }
        })
        .collect()
}

fn generate_tuple_scalar_rows() -> Vec<TupleScalarRow> {
    (0..N_NUMERIC_ROWS)
        .map(|i| {
            let base = i64::try_from(i).unwrap() * 8;
            TupleScalarRow {
                values: (
                    base,
                    base + 1,
                    base + 2,
                    base + 3,
                    base + 4,
                    base + 5,
                    base + 6,
                    base + 7,
                ),
            }
        })
        .collect()
}

const fn scalar_quad(base: i64) -> (i64, i64, i64, i64) {
    (base, base + 1, base + 2, base + 3)
}

fn generate_wide_tuple_scalar_rows() -> Vec<WideTupleScalarRow> {
    (0..N_NUMERIC_ROWS)
        .map(|i| {
            let base = i64::try_from(i).unwrap() * 32;
            WideTupleScalarRow {
                t0: scalar_quad(base),
                t1: scalar_quad(base + 4),
                t2: scalar_quad(base + 8),
                t3: scalar_quad(base + 12),
                t4: scalar_quad(base + 16),
                t5: scalar_quad(base + 20),
                t6: scalar_quad(base + 24),
                t7: scalar_quad(base + 28),
            }
        })
        .collect()
}

fn list_value(row: usize, item: usize, lane: i64) -> i64 {
    i64::try_from(row).unwrap() * 100 + i64::try_from(item).unwrap() * 10 + lane
}

fn generate_nested_struct_list_control() -> Vec<NestedStructListControl> {
    (0..N_NUMERIC_ROWS)
        .map(|row| NestedStructListControl {
            items: (0..(row % 7))
                .map(|item| ListQuad {
                    a: list_value(row, item, 0),
                    b: list_value(row, item, 1),
                    c: list_value(row, item, 2),
                    d: list_value(row, item, 3),
                })
                .collect(),
        })
        .collect()
}

fn generate_nested_tuple_replay_boundary() -> Vec<NestedTupleReplayBoundary> {
    make_tuple_replay_boundary(N_NUMERIC_ROWS)
        .into_iter()
        .map(|nested| NestedTupleReplayBoundary { nested })
        .collect()
}

fn generate_tuple_list_rows() -> Vec<TupleListRow> {
    (0..N_NUMERIC_ROWS)
        .map(|row| TupleListRow {
            items: (0..(row % 7))
                .map(|item| {
                    (
                        list_value(row, item, 0),
                        list_value(row, item, 1),
                        list_value(row, item, 2),
                        list_value(row, item, 3),
                    )
                })
                .collect(),
        })
        .collect()
}

fn generate_vec_bool_rows() -> Vec<VecBoolRow> {
    (0..N_NUMERIC_ROWS)
        .map(|i| VecBoolRow {
            id: i as u64,
            items: (0..(i % 7 + 3)).map(|k| (k + i) % 2 == 0).collect(),
        })
        .collect()
}

fn generate_vec_opt_bool_rows() -> Vec<VecOptBoolRow> {
    (0..N_NUMERIC_ROWS)
        .map(|i| VecOptBoolRow {
            id: i as u64,
            items: (0..(i % 7 + 3))
                .map(|k| {
                    if (k + i) % 5 == 0 {
                        None
                    } else {
                        Some((k + i) % 2 == 0)
                    }
                })
                .collect(),
        })
        .collect()
}

fn generate_vec_i32_rows() -> Vec<VecI32Row> {
    (0..N_NUMERIC_ROWS)
        .map(|i| VecI32Row {
            id: i as u64,
            items: (0..(i % 7 + 3))
                .map(|k| i32::try_from(i * 10 + k).unwrap())
                .collect(),
        })
        .collect()
}

fn generate_vec_opt_i32_rows() -> Vec<VecOptI32Row> {
    (0..N_NUMERIC_ROWS)
        .map(|i| VecOptI32Row {
            id: i as u64,
            items: (0..(i % 7 + 3))
                .map(|k| {
                    if (i + k) % 5 == 0 {
                        None
                    } else {
                        Some(i32::try_from(i * 10 + k).unwrap())
                    }
                })
                .collect(),
        })
        .collect()
}

fn generate_vec_vec_bool_rows() -> Vec<VecVecBoolRow> {
    (0..N_NUMERIC_ROWS)
        .map(|i| VecVecBoolRow {
            id: i as u64,
            items: (0..(i % 5 + 2))
                .map(|j| (0..(j % 4 + 2)).map(|k| (i + j + k) % 2 == 0).collect())
                .collect(),
        })
        .collect()
}

fn nested_i32_value(row: usize, list: usize, item: usize) -> i32 {
    i32::try_from(row).unwrap() * 10 + i32::try_from(list).unwrap() + i32::try_from(item).unwrap()
}

fn generate_vec_vec_i32_rows() -> Vec<VecVecI32Row> {
    (0..N_NUMERIC_ROWS)
        .map(|i| VecVecI32Row {
            id: i as u64,
            items: (0..(i % 5 + 2))
                .map(|j| {
                    (0..(j % 4 + 2))
                        .map(|k| nested_i32_value(i, j, k))
                        .collect()
                })
                .collect(),
        })
        .collect()
}

fn generate_vec_vec_opt_i32_rows() -> Vec<VecVecOptI32Row> {
    (0..N_NUMERIC_ROWS)
        .map(|i| VecVecOptI32Row {
            id: i as u64,
            items: (0..(i % 5 + 2))
                .map(|j| {
                    (0..(j % 4 + 2))
                        .map(|k| {
                            if (i + j + k) % 5 == 0 {
                                None
                            } else {
                                Some(nested_i32_value(i, j, k))
                            }
                        })
                        .collect()
                })
                .collect(),
        })
        .collect()
}

#[library_benchmark]
#[bench::top_level_vec(generate_ticks())]
fn bench_top_level_vec(rows: Vec<Tick>) -> (usize, usize) {
    convert_rows(rows)
}

#[library_benchmark]
#[bench::nested_option(generate_users())]
fn bench_nested_option(rows: Vec<User>) -> (usize, usize) {
    convert_rows(rows)
}

#[library_benchmark]
#[bench::string_columns_required(generate_required_strings())]
fn bench_string_columns_required(rows: Vec<StringRowRequired>) -> (usize, usize) {
    convert_rows(rows)
}

#[library_benchmark]
#[bench::string_columns_optional(generate_optional_strings())]
fn bench_string_columns_optional(rows: Vec<StringRowOptional>) -> (usize, usize) {
    convert_rows(rows)
}

#[library_benchmark]
#[bench::as_binary_tiny_inline(generate_binary_tiny())]
fn bench_as_binary_tiny(rows: Vec<BinaryTinyRow>) -> (usize, usize) {
    convert_rows(rows)
}

#[library_benchmark]
#[bench::as_binary_medium_out_of_line(generate_binary_medium())]
fn bench_as_binary_medium(rows: Vec<BinaryMediumRow>) -> (usize, usize) {
    convert_rows(rows)
}

#[library_benchmark]
#[bench::flat_scalar_control(generate_flat_scalar_control())]
fn bench_flat_scalar_control(rows: Vec<FlatScalarControl>) -> (usize, usize) {
    convert_rows(rows)
}

#[library_benchmark]
#[bench::flat_scalar_control_streaming(generate_flat_scalar_control())]
fn bench_flat_scalar_control_streaming(rows: Vec<FlatScalarControl>) -> (usize, usize) {
    convert_rows_streaming(rows)
}

#[library_benchmark]
#[bench::tuple_scalar_grouped(generate_tuple_scalar_rows())]
fn bench_tuple_scalar_grouped(rows: Vec<TupleScalarRow>) -> (usize, usize) {
    convert_rows(rows)
}

#[library_benchmark]
#[bench::tuple_replay_boundary_minus_one(make_tuple_replay_boundary_minus_one(N_NUMERIC_ROWS))]
fn bench_tuple_replay_boundary_minus_one(rows: Vec<TupleReplayBoundaryMinusOne>) -> (usize, usize) {
    convert_rows(rows)
}

#[library_benchmark]
#[bench::tuple_replay_boundary(make_tuple_replay_boundary(N_NUMERIC_ROWS))]
fn bench_tuple_replay_boundary(rows: Vec<TupleReplayBoundary>) -> (usize, usize) {
    convert_rows(rows)
}

#[library_benchmark]
#[bench::wide_tuple_scalar_replayed(generate_wide_tuple_scalar_rows())]
fn bench_wide_tuple_scalar_replayed(rows: Vec<WideTupleScalarRow>) -> (usize, usize) {
    convert_rows(rows)
}

#[library_benchmark]
#[bench::wide_tuple_scalar_replayed_streaming(generate_wide_tuple_scalar_rows())]
fn bench_wide_tuple_scalar_replayed_streaming(rows: Vec<WideTupleScalarRow>) -> (usize, usize) {
    convert_rows_streaming(rows)
}

#[library_benchmark]
#[bench::nested_struct_list_control(generate_nested_struct_list_control())]
fn bench_nested_struct_list_control(rows: Vec<NestedStructListControl>) -> (usize, usize) {
    convert_rows(rows)
}

#[library_benchmark]
#[bench::nested_tuple_replay_boundary(generate_nested_tuple_replay_boundary())]
fn bench_nested_tuple_replay_boundary(rows: Vec<NestedTupleReplayBoundary>) -> (usize, usize) {
    convert_rows(rows)
}

#[library_benchmark]
#[bench::tuple_list_grouped(generate_tuple_list_rows())]
fn bench_tuple_list_grouped(rows: Vec<TupleListRow>) -> (usize, usize) {
    convert_rows(rows)
}

#[library_benchmark]
#[bench::vec_bool(generate_vec_bool_rows())]
fn bench_vec_bool(rows: Vec<VecBoolRow>) -> (usize, usize) {
    convert_rows(rows)
}

#[library_benchmark]
#[bench::vec_opt_bool(generate_vec_opt_bool_rows())]
fn bench_vec_opt_bool(rows: Vec<VecOptBoolRow>) -> (usize, usize) {
    convert_rows(rows)
}

#[library_benchmark]
#[bench::vec_i32(generate_vec_i32_rows())]
fn bench_vec_i32(rows: Vec<VecI32Row>) -> (usize, usize) {
    convert_rows(rows)
}

#[library_benchmark]
#[bench::vec_opt_i32(generate_vec_opt_i32_rows())]
fn bench_vec_opt_i32(rows: Vec<VecOptI32Row>) -> (usize, usize) {
    convert_rows(rows)
}

#[library_benchmark]
#[bench::vec_vec_bool(generate_vec_vec_bool_rows())]
fn bench_vec_vec_bool(rows: Vec<VecVecBoolRow>) -> (usize, usize) {
    convert_rows(rows)
}

#[library_benchmark]
#[bench::vec_vec_i32(generate_vec_vec_i32_rows())]
fn bench_vec_vec_i32(rows: Vec<VecVecI32Row>) -> (usize, usize) {
    convert_rows(rows)
}

#[library_benchmark]
#[bench::vec_vec_i32_streaming(generate_vec_vec_i32_rows())]
fn bench_vec_vec_i32_streaming(rows: Vec<VecVecI32Row>) -> (usize, usize) {
    convert_rows_streaming(rows)
}

#[library_benchmark]
#[bench::vec_vec_opt_i32(generate_vec_vec_opt_i32_rows())]
fn bench_vec_vec_opt_i32(rows: Vec<VecVecOptI32Row>) -> (usize, usize) {
    convert_rows(rows)
}

library_benchmark_group!(
    name = instruction_counts,
    benchmarks = [
        bench_top_level_vec,
        bench_nested_option,
        bench_string_columns_required,
        bench_string_columns_optional,
        bench_as_binary_tiny,
        bench_as_binary_medium,
        bench_flat_scalar_control,
        bench_flat_scalar_control_streaming,
        bench_tuple_scalar_grouped,
        bench_tuple_replay_boundary_minus_one,
        bench_tuple_replay_boundary,
        bench_wide_tuple_scalar_replayed,
        bench_wide_tuple_scalar_replayed_streaming,
        bench_nested_struct_list_control,
        bench_nested_tuple_replay_boundary,
        bench_tuple_list_grouped
    ]
);

library_benchmark_group!(
    name = primitive_list_instruction_counts,
    benchmarks = [
        bench_vec_bool,
        bench_vec_opt_bool,
        bench_vec_i32,
        bench_vec_opt_i32,
        bench_vec_vec_bool,
        bench_vec_vec_i32,
        bench_vec_vec_i32_streaming,
        bench_vec_vec_opt_i32
    ]
);

main!(
    config = LibraryBenchmarkConfig::default()
        .envs([("POLARS_MAX_THREADS", "1"), ("RAYON_NUM_THREADS", "1")]),
    library_benchmark_groups = [instruction_counts, primitive_list_instruction_counts]
);

// Bench harness for `#[df_derive(as_binary)]` (Finding #1).
//
// Three shapes side-by-side at 100k rows:
//   1. `Vec<u8>` baseline — the default `List(UInt8)` schema. The runtime
//      path needs the polars `dtype-u8` feature, which the workspace does
//      not enable; the harness defines and references the baseline so it
//      keeps building, but only `as_binary_*` shapes run end-to-end.
//   2. `Vec<u8>` with `as_binary` — tiny payloads (8 bytes/row, BinaryView
//      inlines).
//   3. `Vec<u8>` with `as_binary` — medium payloads (1 KB/row, BinaryView
//      out-of-line).
use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;

#[path = "support/mod.rs"]
mod bench_support;
#[path = "../tests/common.rs"]
mod core;
use crate::bench_support::configure_criterion;
use crate::core::dataframe::ToDataFrameVec;

const N_ROWS: usize = 100_000;
const TINY_LEN: usize = 8;
const MEDIUM_LEN: usize = 1024;

#[derive(ToDataFrame, Clone)]
struct ListU8Row {
    bytes: Vec<u8>,
}

#[derive(ToDataFrame, Clone)]
struct BinaryTinyRow {
    #[df_derive(as_binary)]
    bytes: Vec<u8>,
}

#[derive(ToDataFrame, Clone)]
struct BinaryMediumRow {
    #[df_derive(as_binary)]
    bytes: Vec<u8>,
}

fn make_payload(seed: usize, len: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(len);
    for i in 0..len {
        buf.push(u8::try_from((seed.wrapping_add(i)) & 0xff).unwrap());
    }
    buf
}

fn benchmark_as_binary(c: &mut Criterion) {
    let list_u8: Vec<ListU8Row> = (0..N_ROWS)
        .map(|i| ListU8Row {
            bytes: make_payload(i, TINY_LEN),
        })
        .collect();
    let binary_tiny: Vec<BinaryTinyRow> = (0..N_ROWS)
        .map(|i| BinaryTinyRow {
            bytes: make_payload(i, TINY_LEN),
        })
        .collect();
    let binary_medium: Vec<BinaryMediumRow> = (0..N_ROWS)
        .map(|i| BinaryMediumRow {
            bytes: make_payload(i, MEDIUM_LEN),
        })
        .collect();

    c.bench_function("as_binary_tiny_inline", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&binary_tiny).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });

    c.bench_function("as_binary_medium_out_of_line", |b| {
        b.iter(|| {
            let df = std::hint::black_box(&binary_medium).to_dataframe().unwrap();
            std::hint::black_box(df)
        });
    });

    // Reference the `list_u8` data so the compile-only baseline isn't
    // dropped by dead-code elimination — keeps the harness honest about
    // including the baseline shape even though we don't run it.
    std::hint::black_box(&list_u8);
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = benchmark_as_binary
}
criterion_main!(benches);

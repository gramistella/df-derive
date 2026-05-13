// Focused cost-model benchmark for wide tuple and nested scalar shapes.
//
// The flat named baseline is the shape the columnar emitter handles in one
// row loop. Tuple-heavy and nested-heavy variants have the same logical
// column count so regressions from extra projection / nested scans are easier
// to see in Criterion output.

use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;

#[path = "support/mod.rs"]
mod bench_support;
#[path = "../tests/common.rs"]
mod core;
use crate::bench_support::configure_criterion;
use crate::core::dataframe::ToDataFrameVec;

const N_ROWS: usize = 100_000;

#[derive(ToDataFrame, Clone)]
struct Quad {
    a: i64,
    b: i64,
    c: i64,
    d: i64,
}

#[derive(ToDataFrame, Clone)]
struct FlatThirtyTwo {
    f00: i64,
    f01: i64,
    f02: i64,
    f03: i64,
    f04: i64,
    f05: i64,
    f06: i64,
    f07: i64,
    f08: i64,
    f09: i64,
    f10: i64,
    f11: i64,
    f12: i64,
    f13: i64,
    f14: i64,
    f15: i64,
    f16: i64,
    f17: i64,
    f18: i64,
    f19: i64,
    f20: i64,
    f21: i64,
    f22: i64,
    f23: i64,
    f24: i64,
    f25: i64,
    f26: i64,
    f27: i64,
    f28: i64,
    f29: i64,
    f30: i64,
    f31: i64,
}

#[derive(ToDataFrame, Clone)]
struct TupleEightByFour {
    t0: (i64, i64, i64, i64),
    t1: (i64, i64, i64, i64),
    t2: (i64, i64, i64, i64),
    t3: (i64, i64, i64, i64),
    t4: (i64, i64, i64, i64),
    t5: (i64, i64, i64, i64),
    t6: (i64, i64, i64, i64),
    t7: (i64, i64, i64, i64),
}

#[derive(ToDataFrame, Clone)]
struct NestedEightByFour {
    n0: Quad,
    n1: Quad,
    n2: Quad,
    n3: Quad,
    n4: Quad,
    n5: Quad,
    n6: Quad,
    n7: Quad,
}

fn row_value(row: usize, offset: i64) -> i64 {
    i64::try_from(row).unwrap() + offset
}

fn quad(row: usize, base: i64) -> Quad {
    Quad {
        a: row_value(row, base),
        b: row_value(row, base + 1),
        c: row_value(row, base + 2),
        d: row_value(row, base + 3),
    }
}

fn tuple_quad(row: usize, base: i64) -> (i64, i64, i64, i64) {
    (
        row_value(row, base),
        row_value(row, base + 1),
        row_value(row, base + 2),
        row_value(row, base + 3),
    )
}

#[allow(clippy::too_many_lines)]
fn flat_row(row: usize) -> FlatThirtyTwo {
    FlatThirtyTwo {
        f00: row_value(row, 0),
        f01: row_value(row, 1),
        f02: row_value(row, 2),
        f03: row_value(row, 3),
        f04: row_value(row, 4),
        f05: row_value(row, 5),
        f06: row_value(row, 6),
        f07: row_value(row, 7),
        f08: row_value(row, 8),
        f09: row_value(row, 9),
        f10: row_value(row, 10),
        f11: row_value(row, 11),
        f12: row_value(row, 12),
        f13: row_value(row, 13),
        f14: row_value(row, 14),
        f15: row_value(row, 15),
        f16: row_value(row, 16),
        f17: row_value(row, 17),
        f18: row_value(row, 18),
        f19: row_value(row, 19),
        f20: row_value(row, 20),
        f21: row_value(row, 21),
        f22: row_value(row, 22),
        f23: row_value(row, 23),
        f24: row_value(row, 24),
        f25: row_value(row, 25),
        f26: row_value(row, 26),
        f27: row_value(row, 27),
        f28: row_value(row, 28),
        f29: row_value(row, 29),
        f30: row_value(row, 30),
        f31: row_value(row, 31),
    }
}

fn tuple_row(row: usize) -> TupleEightByFour {
    TupleEightByFour {
        t0: tuple_quad(row, 0),
        t1: tuple_quad(row, 4),
        t2: tuple_quad(row, 8),
        t3: tuple_quad(row, 12),
        t4: tuple_quad(row, 16),
        t5: tuple_quad(row, 20),
        t6: tuple_quad(row, 24),
        t7: tuple_quad(row, 28),
    }
}

fn nested_row(row: usize) -> NestedEightByFour {
    NestedEightByFour {
        n0: quad(row, 0),
        n1: quad(row, 4),
        n2: quad(row, 8),
        n3: quad(row, 12),
        n4: quad(row, 16),
        n5: quad(row, 20),
        n6: quad(row, 24),
        n7: quad(row, 28),
    }
}

fn make_flat() -> Vec<FlatThirtyTwo> {
    (0..N_ROWS).map(flat_row).collect()
}

fn make_tuple_heavy() -> Vec<TupleEightByFour> {
    (0..N_ROWS).map(tuple_row).collect()
}

fn make_nested_heavy() -> Vec<NestedEightByFour> {
    (0..N_ROWS).map(nested_row).collect()
}

fn bench_cost_model_passes(c: &mut Criterion) {
    let flat = make_flat();
    let tuple_heavy = make_tuple_heavy();
    let nested_heavy = make_nested_heavy();

    let mut group = c.benchmark_group("cost_model_passes");
    group.bench_function("flat_32_scalar_fields", |b| {
        b.iter(|| std::hint::black_box(&flat).to_dataframe().unwrap());
    });
    group.bench_function("tuple_8x4_scalar_elements", |b| {
        b.iter(|| std::hint::black_box(&tuple_heavy).to_dataframe().unwrap());
    });
    group.bench_function("nested_8x4_scalar_fields", |b| {
        b.iter(|| std::hint::black_box(&nested_heavy).to_dataframe().unwrap());
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = bench_cost_model_passes
}
criterion_main!(benches);

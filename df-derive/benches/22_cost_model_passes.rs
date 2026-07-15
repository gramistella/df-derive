// Focused cost-model benchmark for wide tuple and nested scalar shapes.
//
// The flat named baseline is the shape the columnar emitter handles in one
// row loop. Tuple-heavy and nested-heavy variants have the same logical
// column count so regressions from extra projection / nested scans are easier
// to see in Criterion output.

use criterion::{Criterion, criterion_group, criterion_main};
use df_derive::ToDataFrame;
use rust_decimal::Decimal;

#[path = "support/mod.rs"]
mod bench_support;
#[path = "support/scalar_policy_boundaries.rs"]
mod scalar_policy_boundaries;
#[path = "support/tuple_replay_boundary.rs"]
mod tuple_replay_boundary;
use crate::bench_support::configure_criterion;
use crate::scalar_policy_boundaries::{
    FlatThirtyTwoScalarFields, SeventeenScalarTupleTerminals, SixteenOneElementTupleFields,
    make_flat_thirty_two_scalar_fields, make_seventeen_scalar_tuple_terminals,
    make_sixteen_one_element_tuple_fields,
};
use crate::tuple_replay_boundary::{
    TupleReplayBoundary, make_tuple_replay_boundary, make_tuple_replay_boundary_minus_one,
};
use df_derive::dataframe::{Columnar, ToDataFrameVec};

const N_ROWS: usize = 100_000;

#[derive(ToDataFrame, Clone)]
struct Quad {
    a: i64,
    b: i64,
    c: i64,
    d: i64,
}

#[derive(ToDataFrame)]
struct NestedTupleReplayBoundary {
    nested: TupleReplayBoundary,
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
#[allow(clippy::type_complexity)]
struct MixedTupleThirtyTwo {
    values: (
        (Decimal, i64, i64, i64, i64, i64, i64, i64),
        (i64, i64, i64, i64, i64, i64, i64, i64),
        (i64, i64, i64, i64, i64, i64, i64, i64),
        (i64, i64, i64, i64, i64, i64, i64, i64),
    ),
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

fn tuple_octet(row: usize, base: i64) -> (i64, i64, i64, i64, i64, i64, i64, i64) {
    (
        row_value(row, base),
        row_value(row, base + 1),
        row_value(row, base + 2),
        row_value(row, base + 3),
        row_value(row, base + 4),
        row_value(row, base + 5),
        row_value(row, base + 6),
        row_value(row, base + 7),
    )
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

fn mixed_tuple_row(row: usize) -> MixedTupleThirtyTwo {
    let first = tuple_octet(row, 0);
    MixedTupleThirtyTwo {
        values: (
            (
                Decimal::from(row_value(row, 0)),
                first.1,
                first.2,
                first.3,
                first.4,
                first.5,
                first.6,
                first.7,
            ),
            tuple_octet(row, 8),
            tuple_octet(row, 16),
            tuple_octet(row, 24),
        ),
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

fn make_tuple_heavy() -> Vec<TupleEightByFour> {
    (0..N_ROWS).map(tuple_row).collect()
}

fn make_mixed_tuple_heavy() -> Vec<MixedTupleThirtyTwo> {
    (0..N_ROWS).map(mixed_tuple_row).collect()
}

fn make_nested_heavy() -> Vec<NestedEightByFour> {
    (0..N_ROWS).map(nested_row).collect()
}

fn make_nested_tuple_replay() -> Vec<NestedTupleReplayBoundary> {
    make_tuple_replay_boundary(N_ROWS)
        .into_iter()
        .map(|nested| NestedTupleReplayBoundary { nested })
        .collect()
}

fn bench_cost_model_passes(c: &mut Criterion) {
    let flat = make_flat_thirty_two_scalar_fields(N_ROWS);
    let sixteen_one_element_tuple_fields = make_sixteen_one_element_tuple_fields(N_ROWS);
    let seventeen_scalar_tuple_terminals = make_seventeen_scalar_tuple_terminals(N_ROWS);
    let tuple_heavy = make_tuple_heavy();
    let tuple_replay_boundary_minus_one = make_tuple_replay_boundary_minus_one(N_ROWS);
    let tuple_replay_boundary = make_tuple_replay_boundary(N_ROWS);
    let mixed_tuple_heavy = make_mixed_tuple_heavy();
    let nested_heavy = make_nested_heavy();
    let nested_tuple_replay = make_nested_tuple_replay();

    let mut group = c.benchmark_group("cost_model_passes");
    group.bench_function("flat_32_scalar_fields", |b| {
        b.iter(|| std::hint::black_box(&flat).to_dataframe().unwrap());
    });
    group.bench_function("flat_32_scalar_fields_streaming", |b| {
        b.iter(|| {
            <FlatThirtyTwoScalarFields as Columnar>::encode(std::hint::black_box(flat.iter()))
                .unwrap()
        });
    });
    group.bench_function("tuple_16_one_element_fields", |b| {
        b.iter(|| {
            std::hint::black_box(&sixteen_one_element_tuple_fields)
                .to_dataframe()
                .unwrap()
        });
    });
    group.bench_function("tuple_16_one_element_fields_streaming", |b| {
        b.iter(|| {
            <SixteenOneElementTupleFields as Columnar>::encode(std::hint::black_box(
                sixteen_one_element_tuple_fields.iter(),
            ))
            .unwrap()
        });
    });
    group.bench_function("tuple_17_scalar_terminals", |b| {
        b.iter(|| {
            std::hint::black_box(&seventeen_scalar_tuple_terminals)
                .to_dataframe()
                .unwrap()
        });
    });
    group.bench_function("tuple_17_scalar_terminals_streaming", |b| {
        b.iter(|| {
            <SeventeenScalarTupleTerminals as Columnar>::encode(std::hint::black_box(
                seventeen_scalar_tuple_terminals.iter(),
            ))
            .unwrap()
        });
    });
    group.bench_function("tuple_8x4_scalar_elements", |b| {
        b.iter(|| std::hint::black_box(&tuple_heavy).to_dataframe().unwrap());
    });
    group.bench_function("tuple_8x4_scalar_elements_streaming", |b| {
        b.iter(|| {
            <TupleEightByFour as Columnar>::encode(std::hint::black_box(tuple_heavy.iter()))
                .unwrap()
        });
    });
    group.bench_function("tuple_replay_boundary_minus_one", |b| {
        b.iter(|| {
            std::hint::black_box(&tuple_replay_boundary_minus_one)
                .to_dataframe()
                .unwrap()
        });
    });
    group.bench_function("tuple_replay_boundary", |b| {
        b.iter(|| {
            std::hint::black_box(&tuple_replay_boundary)
                .to_dataframe()
                .unwrap()
        });
    });
    group.bench_function("tuple_mixed_1_decimal_31_scalar_elements", |b| {
        b.iter(|| {
            std::hint::black_box(&mixed_tuple_heavy)
                .to_dataframe()
                .unwrap()
        });
    });
    group.bench_function("nested_8x4_scalar_fields", |b| {
        b.iter(|| std::hint::black_box(&nested_heavy).to_dataframe().unwrap());
    });
    group.bench_function("nested_tuple_replay_boundary", |b| {
        b.iter(|| {
            std::hint::black_box(&nested_tuple_replay)
                .to_dataframe()
                .unwrap()
        });
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = bench_cost_model_passes
}
criterion_main!(benches);

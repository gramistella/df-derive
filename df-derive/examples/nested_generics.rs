//! Demonstrates a generic struct used as a nested field.
//!
//! A `#[derive(ToDataFrame)]` type with a type parameter can be instantiated
//! at concrete types (themselves derived) and embedded in another derived
//! struct. Each instantiation flattens with dot notation, just like a
//! non-generic nested struct, and `Vec<Generic<T>>` becomes one
//! `List<...>` column per inner schema column. This shows two pieces fitting
//! together: (1) generic structs derive correctly when used standalone,
//! (2) those instantiations compose under wrapper layers when nested into
//! another derive.
//!
//! The macro injects `T: ToDataFrame + Columnar` bounds on every type
//! parameter (no `Clone` — bulk emitters borrow from `&T`), so any concrete
//! instantiation must be derivable. Using already-derived nested structs as
//! the type arguments is the idiomatic fit; using primitives requires
//! providing manual `ToDataFrame + Columnar` impls for those primitives,
//! which the existing `quickstart` example does not do because it isn't
//! generic.
//!
//! Uses the default `df-derive` facade runtime.

use df_derive::ToDataFrame;
use df_derive::dataframe::ToDataFrameVec as _;

// Two concrete payloads to instantiate the generic with. Both derive
// `ToDataFrame`, which the macro requires for any concrete `T`.
#[derive(ToDataFrame, Clone)]
struct IntPayload {
    timestamp: i64,
    seq: u32,
}

#[derive(ToDataFrame, Clone)]
struct StringPayload {
    name: String,
    note: String,
}

// Generic carrier struct. The macro injects `T: ToDataFrame + Columnar` on
// its impl blocks; `#[derive(Clone)]` adds its own `T: Clone` bound. So
// `Generic<P>` derives transparently for any `P` that satisfies all three.
#[derive(ToDataFrame, Clone)]
struct Generic<T> {
    value: T,
    label: String,
}

// Outer struct that embeds two distinct instantiations of `Generic<T>`:
//   - `inner: Generic<IntPayload>` — flattens to
//     `inner.value.timestamp`, `inner.value.seq`, `inner.label`.
//   - `list: Vec<Generic<StringPayload>>` — each inner field becomes a
//     List column: `list.value.name`, `list.value.note`, `list.label`.
#[derive(ToDataFrame, Clone)]
struct Outer {
    id: u32,
    inner: Generic<IntPayload>,
    list: Vec<Generic<StringPayload>>,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let rows = vec![
        Outer {
            id: 1,
            inner: Generic {
                value: IntPayload {
                    timestamp: 1_700_000_000,
                    seq: 1,
                },
                label: "alpha".into(),
            },
            list: vec![
                Generic {
                    value: StringPayload {
                        name: "first".into(),
                        note: "x".into(),
                    },
                    label: "list-a".into(),
                },
                Generic {
                    value: StringPayload {
                        name: "second".into(),
                        note: "y".into(),
                    },
                    label: "list-b".into(),
                },
            ],
        },
        Outer {
            id: 2,
            inner: Generic {
                value: IntPayload {
                    timestamp: 1_700_000_100,
                    seq: 2,
                },
                label: "beta".into(),
            },
            list: vec![],
        },
    ];

    println!("Outer (batch):\n{}", rows.as_slice().to_dataframe()?);

    let schema = <Outer as df_derive::dataframe::ToDataFrame>::schema()?;
    println!("\nSchema (generic instantiations flatten with dot notation):");
    for (name, dtype) in schema {
        println!("  {name}: {dtype:?}");
    }

    Ok(())
}

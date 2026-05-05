//! Shared support module for criterion benches.
//!
//! Each `[[bench]]` target compiles as its own binary with no shared crate
//! scope, so this file is pulled into every bench via
//! `#[path = "support/mod.rs"] mod bench_support;`. The file lives in a
//! sub-directory so cargo's bench auto-discovery does not pick it up as a
//! standalone target. It owns the criterion timing config shared by every
//! bench.
//!
//! The runtime-trait module (`tests/common.rs`) is pulled into every bench
//! as a top-level `core` module so the proc-macro-generated
//! `crate::core::dataframe::*` paths resolve.

use criterion::Criterion;
use std::time::Duration;

#[must_use]
pub fn configure_criterion() -> Criterion {
    Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .noise_threshold(0.02)
        .confidence_level(0.99)
}

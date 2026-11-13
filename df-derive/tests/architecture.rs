// Architecture fixtures embed small downstream crates as raw strings; keeping
// those snippets readable is more useful than linting them as production code.
#![allow(clippy::missing_const_for_fn, clippy::needless_raw_string_hashes)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn package_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn repo_root() -> PathBuf {
    package_root()
        .parent()
        .expect("facade crate lives under the workspace root")
        .to_path_buf()
}

fn toml_path(path: &Path) -> String {
    path.display().to_string().replace('\\', "\\\\")
}

fn write_fixture_file(root: &Path, rel_path: &str, contents: &str) {
    let path = root.join(rel_path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create fixture file parent");
    }
    fs::write(path, contents).expect("write fixture file");
}

fn write_fixture(
    name: &str,
    manifest: &str,
    main_rs: &str,
    extra_files: &[(&str, &str)],
) -> PathBuf {
    let root = repo_root();
    let fixture_root = root.join("target").join("architecture-fixtures").join(name);
    if fixture_root.exists() {
        fs::remove_dir_all(&fixture_root).expect("remove stale fixture");
    }
    fs::create_dir_all(fixture_root.join("src")).expect("create fixture src");
    write_fixture_file(&fixture_root, "Cargo.toml", manifest);
    write_fixture_file(&fixture_root, "src/main.rs", main_rs);
    for (rel_path, contents) in extra_files {
        write_fixture_file(&fixture_root, rel_path, contents);
    }
    fixture_root
}

fn check_fixture_with_files(
    name: &str,
    manifest: &str,
    main_rs: &str,
    extra_files: &[(&str, &str)],
) {
    check_fixture_with_files_and_args(name, manifest, main_rs, extra_files, &[]);
}

/// Runs `cargo check` on a fixture and returns what it printed and how it
/// ended.
fn run_check(
    name: &str,
    manifest: &str,
    main_rs: &str,
    extra_files: &[(&str, &str)],
    cargo_args: &[&str],
) -> Output {
    let root = repo_root();
    let fixture_root = write_fixture(name, manifest, main_rs, extra_files);

    Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
        .arg("check")
        .arg("--quiet")
        .arg("--manifest-path")
        .arg(fixture_root.join("Cargo.toml"))
        .args(cargo_args)
        .env(
            "CARGO_TARGET_DIR",
            root.join("target").join("architecture-fixtures-target"),
        )
        .output()
        .expect("run cargo check")
}

fn check_fixture_with_files_and_args(
    name: &str,
    manifest: &str,
    main_rs: &str,
    extra_files: &[(&str, &str)],
    cargo_args: &[&str],
) {
    let output = run_check(name, manifest, main_rs, extra_files, cargo_args);
    assert!(
        output.status.success(),
        "fixture `{name}` failed\n\nstdout:\n{}\n\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

/// Checks a fixture that must not compile and returns the compiler's errors.
fn failing_fixture(name: &str, manifest: &str, main_rs: &str) -> String {
    let output = run_check(name, manifest, main_rs, &[], &[]);
    assert!(!output.status.success(), "fixture `{name}` compiled");
    String::from_utf8(output.stderr).expect("cargo check stderr is valid UTF-8")
}

fn check_fixture(name: &str, manifest: &str, main_rs: &str) {
    check_fixture_with_files(name, manifest, main_rs, &[]);
}

fn cargo_tree_fixture_with_files(
    name: &str,
    manifest: &str,
    main_rs: &str,
    extra_files: &[(&str, &str)],
    cargo_args: &[&str],
) -> String {
    let root = repo_root();
    let fixture_root = write_fixture(name, manifest, main_rs, extra_files);

    let output = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
        .arg("tree")
        .arg("--quiet")
        .arg("--manifest-path")
        .arg(fixture_root.join("Cargo.toml"))
        .args(cargo_args)
        .env(
            "CARGO_TARGET_DIR",
            root.join("target").join("architecture-fixtures-target"),
        )
        .output()
        .expect("run cargo tree");

    assert!(
        output.status.success(),
        "fixture `{name}` cargo tree failed\n\nstdout:\n{}\n\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    String::from_utf8(output.stdout).expect("cargo tree stdout is valid UTF-8")
}

fn polars_deps() -> &'static str {
    r#"
polars = { version = "0.55", features = ["timezones", "dtype-decimal", "dtype-date", "dtype-datetime", "dtype-time", "dtype-duration"] }
"#
}

#[test]
fn facade_default_runtime_works_without_attributes() {
    let root = package_root();
    let manifest = format!(
        r#"
[package]
name = "facade-default-runtime"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
df-derive = {{ path = "{}" }}
{}
"#,
        toml_path(root),
        polars_deps(),
    );

    check_fixture(
        "facade-default-runtime",
        &manifest,
        r#"
use df_derive::prelude::*;

#[derive(ToDataFrame)]
struct Trade {
    symbol: String,
    price: f64,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let rows = vec![
        Trade { symbol: "AAPL".into(), price: 187.23 },
        Trade { symbol: "MSFT".into(), price: 411.61 },
    ];
    let df = rows.as_slice().to_dataframe()?;
    assert_eq!(df.shape(), (2, 2));
    Ok(())
}
"#,
    );
}

#[test]
fn generated_code_works_without_try_from_in_downstream_prelude() {
    let root = package_root();
    let manifest = format!(
        r#"
[package]
name = "edition-2018-no-try-from-prelude"
version = "0.0.0"
edition = "2018"
publish = false

[workspace]

[dependencies]
df-derive = {{ path = "{}" }}
{}
"#,
        toml_path(root),
        polars_deps(),
    );

    check_fixture(
        "edition-2018-no-try-from-prelude",
        &manifest,
        r#"
use df_derive::prelude::*;

#[derive(ToDataFrame)]
struct Row {
    elapsed: std::time::Duration,
    values: Vec<i32>,
    flags: Vec<bool>,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let rows = vec![Row {
        elapsed: std::time::Duration::from_nanos(7),
        values: vec![1, 2, 3],
        flags: vec![true, false, true],
    }];

    let df = rows.as_slice().to_dataframe()?;
    assert_eq!(df.shape(), (1, 3));
    Ok(())
}
"#,
    );
}

#[test]
fn scalar_derive_compiles_with_downstream_deny_warnings() {
    let root = package_root();
    let manifest = format!(
        r#"
[package]
name = "scalar-deny-warnings"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
df-derive = {{ path = "{}" }}
{}
"#,
        toml_path(root),
        polars_deps(),
    );

    check_fixture(
        "scalar-deny-warnings",
        &manifest,
        r#"
#![deny(warnings)]

use df_derive::prelude::*;

#[derive(ToDataFrame)]
struct Row {
    value: i32,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let rows = [Row { value: 7 }];
    let df = rows.as_slice().to_dataframe()?;
    assert_eq!(df.shape(), (1, 1));
    Ok(())
}
"#,
    );
}

/// A crate that depends on the facade with the given features and on nothing
/// else of ours: its manifest, with `dependencies` added after the facade.
fn feature_manifest(name: &str, features: &[&str], dependencies: &str) -> String {
    let features = features
        .iter()
        .map(|feature| format!("\"{feature}\""))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        r#"
[package]
name = "{name}"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
df-derive = {{ path = "{}", default-features = false, features = [{features}] }}
polars = {{ version = "0.55", default-features = false }}
{dependencies}
"#,
        toml_path(package_root()),
    )
}

/// The features of the `polars` crate that one of our features turns on.
const TIED_POLARS_FEATURES: [&str; 5] = [
    "dtype-date",
    "dtype-datetime",
    "dtype-time",
    "timezones",
    "dtype-decimal",
];

/// Checks which of the tied Polars features a set of our features turns on,
/// and returns the feature tree for further checks.
fn assert_polars_features(name: &str, features: &[&str], expected: &[&str]) -> String {
    let tree = cargo_tree_fixture_with_files(
        name,
        &feature_manifest(name, features, ""),
        "fn main() {}",
        &[],
        &["--edges", "features"],
    );
    assert!(
        tree.contains("df-derive-core v"),
        "feature tree should include the core crate\n\n{tree}"
    );
    assert!(
        !tree.contains("df-derive-core feature \"default\""),
        "facade default-features = false must not enable core defaults\n\n{tree}"
    );
    for feature in TIED_POLARS_FEATURES {
        assert_eq!(
            tree.contains(&format!("polars feature \"{feature}\"")),
            expected.contains(&feature),
            "polars feature `{feature}` with df-derive features {features:?}\n\n{tree}"
        );
    }
    tree
}

#[test]
fn each_feature_turns_on_the_polars_features_it_needs_and_no_other() {
    let dates = ["dtype-date", "dtype-datetime", "dtype-time", "timezones"];

    let tree = assert_polars_features("features-none", &[], &[]);
    assert!(
        !tree.contains("rust_decimal v"),
        "rust_decimal should not appear in the resolved dependency tree\n\n{tree}"
    );
    for feature in ["rust_decimal", "decimal", "chrono"] {
        assert!(
            !tree.contains(&format!("df-derive-core feature \"{feature}\"")),
            "core feature `{feature}` should stay disabled\n\n{tree}"
        );
    }

    let tree = assert_polars_features("features-chrono", &["chrono"], &dates);
    assert!(
        !tree.contains("rust_decimal v"),
        "`chrono` should not bring rust_decimal in\n\n{tree}"
    );

    let tree = assert_polars_features("features-decimal", &["decimal"], &["dtype-decimal"]);
    assert!(
        !tree.contains("rust_decimal v"),
        "`decimal` alone should not bring rust_decimal in\n\n{tree}"
    );

    let tree = assert_polars_features(
        "features-rust-decimal",
        &["rust_decimal"],
        &["dtype-decimal"],
    );
    assert!(
        tree.contains("rust_decimal v"),
        "`rust_decimal` should bring rust_decimal in\n\n{tree}"
    );
}

// The four fixtures below build the facade with one feature set each, as a
// downstream crate would. A feature set that is never built on its own can
// stop compiling without anything noticing: every other test here runs with
// all features on.

#[test]
fn without_features_the_standard_library_types_are_covered() {
    check_fixture(
        "features-none-build",
        &feature_manifest("features-none-build", &[], ""),
        r#"
use df_derive::prelude::*;

#[derive(ToDataFrame)]
struct Row {
    id: u64,
    small: i8,
    wide: u128,
    label: String,
    scores: Vec<Option<f64>>,
    elapsed: std::time::Duration,
    #[df_derive(time_unit = "ms")]
    waited: Option<std::time::Duration>,
    #[df_derive(as_binary)]
    payload: Vec<u8>,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let rows = vec![Row {
        id: 1,
        small: -1,
        wide: 1,
        label: "a".into(),
        scores: vec![Some(0.5), None],
        elapsed: std::time::Duration::from_nanos(7),
        waited: None,
        payload: vec![1, 2, 3],
    }];
    let df = rows.as_slice().to_dataframe()?;
    assert_eq!(df.shape(), (1, 8));
    Ok(())
}
"#,
    );
}

#[test]
fn chrono_alone_covers_dates_and_times() {
    check_fixture(
        "features-chrono-build",
        &feature_manifest(
            "features-chrono-build",
            &["chrono"],
            r#"chrono = { version = "0.4", default-features = false }"#,
        ),
        r#"
use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, TimeDelta, Utc};
use df_derive::prelude::*;

#[derive(ToDataFrame)]
struct Row {
    at: DateTime<Utc>,
    #[df_derive(time_unit = "ns")]
    seen: Option<NaiveDateTime>,
    day: NaiveDate,
    time: NaiveTime,
    took: Vec<TimeDelta>,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let at = DateTime::<Utc>::UNIX_EPOCH;
    let rows = vec![Row {
        at,
        seen: Some(at.naive_utc()),
        day: at.date_naive(),
        time: at.time(),
        took: vec![TimeDelta::seconds(1)],
    }];
    let df = rows.as_slice().to_dataframe()?;
    assert_eq!(df.shape(), (1, 5));
    Ok(())
}
"#,
    );
}

#[test]
fn rust_decimal_alone_covers_decimals() {
    check_fixture(
        "features-rust-decimal-build",
        &feature_manifest(
            "features-rust-decimal-build",
            &["rust_decimal"],
            r#"rust_decimal = { version = "1.42", default-features = false }"#,
        ),
        r#"
use df_derive::prelude::*;
use rust_decimal::Decimal;

#[derive(ToDataFrame)]
struct Row {
    price: Decimal,
    #[df_derive(decimal(precision = 18, scale = 6))]
    fee: Option<Decimal>,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let rows = vec![Row { price: Decimal::new(12345, 2), fee: None }];
    let df = rows.as_slice().to_dataframe()?;
    assert_eq!(df.shape(), (1, 2));
    Ok(())
}
"#,
    );
}

#[test]
fn decimal_alone_covers_a_decimal_type_of_ones_own() {
    check_fixture(
        "features-decimal-build",
        &feature_manifest("features-decimal-build", &["decimal"], ""),
        r#"
use df_derive::prelude::*;

/// An amount in hundredths.
struct Cents(i64);

impl Decimal128Encode for Cents {
    fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128> {
        // Exact at two decimals or more. A coarser scale would have to round.
        let shift = target_scale.checked_sub(2)?;
        i128::from(self.0).checked_mul(10_i128.checked_pow(shift)?)
    }
}

#[derive(ToDataFrame)]
struct Row {
    #[df_derive(decimal(precision = 20, scale = 2))]
    amount: Cents,
    #[df_derive(decimal(precision = 24, scale = 4))]
    fees: Vec<Option<Cents>>,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let rows = vec![Row { amount: Cents(12345), fees: vec![Some(Cents(5)), None] }];
    let df = rows.as_slice().to_dataframe()?;
    assert_eq!(df.shape(), (1, 2));
    Ok(())
}
"#,
    );
}

#[test]
fn a_type_that_needs_a_feature_is_reported_with_the_feature() {
    // The fixture has `chrono` and `rust_decimal` of its own, and df-derive
    // without the features that cover their types.
    let errors = failing_fixture(
        "features-none-missing",
        &feature_manifest(
            "features-none-missing",
            &[],
            r#"chrono = { version = "0.4", default-features = false }
rust_decimal = { version = "1.42", default-features = false }"#,
        ),
        r#"
use df_derive::prelude::*;

#[derive(ToDataFrame)]
struct Row {
    at: chrono::DateTime<chrono::Utc>,
    price: rust_decimal::Decimal,
}

fn main() {}
"#,
    );
    for expected in [
        "`DateTime<Utc>` cannot be stored in a DataFrame column",
        "df-derive's `chrono` feature is off: turn it on if `DateTime<Utc>` is a `chrono` type",
        "`Decimal` cannot be stored in a DataFrame column",
        "df-derive's `rust_decimal` feature is off: turn it on if `Decimal` is `rust_decimal::Decimal`",
    ] {
        assert!(
            errors.contains(expected),
            "missing `{expected}` in:\n\n{errors}"
        );
    }
}

#[test]
fn core_runtime_with_crate_attribute_enables_numeric_dtype_features() {
    let root = repo_root();
    let manifest = format!(
        r#"
[package]
name = "core-runtime-dtype-features"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
df-derive-core = {{ path = "{}" }}
df-derive-macros = {{ path = "{}" }}
polars = {{ version = "0.55", default-features = false }}
"#,
        toml_path(&root.join("df-derive-core")),
        toml_path(&root.join("df-derive-macros")),
    );

    check_fixture(
        "core-runtime-dtype-features",
        &manifest,
        r#"
use df_derive_core::dataframe::{ToDataFrame as _, ToDataFrameVec as _};
use df_derive_macros::ToDataFrame;
use polars::prelude::{DataType, PolarsResult};

#[derive(ToDataFrame)]
#[df_derive(crate = "::df_derive_core")]
struct Row {
    i8_v: i8,
    i16_v: i16,
    i128_v: i128,
    u8_v: u8,
    u16_v: u16,
    u128_v: u128,
}

fn main() -> PolarsResult<()> {
    let schema = Row::schema()?;
    assert_eq!(
        schema,
        vec![
            ("i8_v".to_string(), DataType::Int8),
            ("i16_v".to_string(), DataType::Int16),
            ("i128_v".to_string(), DataType::Int128),
            ("u8_v".to_string(), DataType::UInt8),
            ("u16_v".to_string(), DataType::UInt16),
            ("u128_v".to_string(), DataType::UInt128),
        ]
    );

    let rows = vec![Row {
        i8_v: -8,
        i16_v: -16,
        i128_v: -128,
        u8_v: 8,
        u16_v: 16,
        u128_v: 128,
    }];
    let df = rows.as_slice().to_dataframe()?;
    assert_eq!(df.shape(), (1, 6));
    Ok(())
}
"#,
    );
}

#[test]
fn crate_attribute_targets_a_crate_that_re_exports_the_runtime() {
    let root = package_root();
    let wrapper_manifest = format!(
        r#"
[package]
name = "model-kit"
version = "0.0.0"
edition = "2024"
publish = false

[dependencies]
df-derive = {{ path = "{}" }}
"#,
        toml_path(root),
    );
    let manifest = format!(
        r#"
[package]
name = "re-exported-runtime"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
model-kit = {{ path = "model-kit" }}
{}
"#,
        polars_deps(),
    );

    // The downstream crate depends only on `model-kit`, never on `df-derive`.
    // Its derives name the runtime through `model_kit::dataframe`, and its
    // rows nest a type that `model-kit` derived with the default path.
    check_fixture_with_files(
        "re-exported-runtime",
        &manifest,
        r#"
use model_kit::dataframe::ToDataFrameVec as _;
use model_kit::{Money, ToDataFrame};

#[derive(ToDataFrame)]
#[df_derive(crate = "::model_kit")]
struct Trade {
    symbol: String,
    notional: Option<Money>,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let rows = vec![Trade {
        symbol: "AAPL".into(),
        notional: Some(Money { amount: 1.5, currency: "USD".into() }),
    }];
    let df = rows.as_slice().to_dataframe()?;
    let names: Vec<_> = df.get_column_names().into_iter().map(|name| name.as_str()).collect();
    assert_eq!(names, ["symbol", "notional.amount", "notional.currency"]);
    Ok(())
}
"#,
        &[
            ("model-kit/Cargo.toml", &wrapper_manifest),
            (
                "model-kit/src/lib.rs",
                r#"
pub use df_derive::{ToDataFrame, dataframe};

#[derive(ToDataFrame)]
pub struct Money {
    pub amount: f64,
    pub currency: String,
}
"#,
            ),
        ],
    );
}

#[test]
fn renamed_facade_dependency_works_with_the_crate_attribute() {
    let root = package_root();
    let manifest = format!(
        r#"
[package]
name = "renamed-facade"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
dfd = {{ package = "df-derive", path = "{}" }}
pl = {{ package = "polars", version = "0.55", default-features = false }}
"#,
        toml_path(root),
    );

    // Generated code reaches Polars through the runtime's own re-export, so a
    // renamed or absent direct `polars` dependency does not matter.
    check_fixture(
        "renamed-facade",
        &manifest,
        r#"
use dfd::prelude::*;

#[derive(ToDataFrame)]
#[df_derive(crate = "::dfd")]
struct Row {
    id: u32,
    tags: Vec<String>,
    note: Option<String>,
}

fn main() -> pl::prelude::PolarsResult<()> {
    let rows = vec![Row { id: 1, tags: vec!["a".into()], note: None }];
    let df = rows.as_slice().to_dataframe()?;
    assert_eq!(df.shape(), (1, 3));
    Ok(())
}
"#,
    );
}

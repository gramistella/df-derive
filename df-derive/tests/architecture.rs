// Architecture fixtures embed small downstream crates as raw strings; keeping
// those snippets readable is more useful than linting them as production code.
#![allow(clippy::missing_const_for_fn, clippy::needless_raw_string_hashes)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

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

fn check_fixture_with_files_and_args(
    name: &str,
    manifest: &str,
    main_rs: &str,
    extra_files: &[(&str, &str)],
    cargo_args: &[&str],
) {
    let root = repo_root();
    let fixture_root = write_fixture(name, manifest, main_rs, extra_files);

    let output = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
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
        .expect("run cargo check");

    assert!(
        output.status.success(),
        "fixture `{name}` failed\n\nstdout:\n{}\n\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
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

#[test]
fn facade_default_features_false_does_not_enable_core_rust_decimal() {
    let root = package_root();
    let manifest = format!(
        r#"
[package]
name = "facade-no-default-features"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
df-derive = {{ path = "{}", default-features = false }}
"#,
        toml_path(root),
    );

    let tree = cargo_tree_fixture_with_files(
        "facade-no-default-features",
        &manifest,
        "fn main() {}",
        &[],
        &["--edges", "features"],
    );

    assert!(
        tree.contains("df-derive v"),
        "feature tree should include the facade crate\n\n{tree}"
    );
    assert!(
        tree.contains("df-derive-core v"),
        "feature tree should include the core crate\n\n{tree}"
    );
    assert!(
        !tree.contains("df-derive feature \"rust_decimal\""),
        "facade rust_decimal feature should stay disabled\n\n{tree}"
    );
    assert!(
        !tree.contains("df-derive-core feature \"default\""),
        "facade default-features = false must not enable core defaults\n\n{tree}"
    );
    assert!(
        !tree.contains("df-derive-core feature \"rust_decimal\""),
        "facade default-features = false must not enable core rust_decimal\n\n{tree}"
    );
    assert!(
        !tree.contains("rust_decimal v"),
        "rust_decimal should not appear in the resolved dependency tree\n\n{tree}"
    );
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

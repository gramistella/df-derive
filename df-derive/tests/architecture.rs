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

fn check_fixture(name: &str, manifest: &str, main_rs: &str) {
    let root = repo_root();
    let fixture_root = root.join("target").join("architecture-fixtures").join(name);
    if fixture_root.exists() {
        fs::remove_dir_all(&fixture_root).expect("remove stale fixture");
    }
    fs::create_dir_all(fixture_root.join("src")).expect("create fixture src");
    fs::write(fixture_root.join("Cargo.toml"), manifest).expect("write fixture manifest");
    fs::write(fixture_root.join("src").join("main.rs"), main_rs).expect("write fixture main");

    let output = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
        .arg("check")
        .arg("--quiet")
        .arg("--manifest-path")
        .arg(fixture_root.join("Cargo.toml"))
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

fn polars_deps() -> &'static str {
    r#"
polars = { version = "0.53.0", features = ["timezones", "dtype-decimal", "dtype-date", "dtype-time", "dtype-duration"] }
polars-arrow = "0.53.0"
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
fn macros_direct_with_core_runtime_works_without_facade() {
    let root = repo_root();
    let manifest = format!(
        r#"
[package]
name = "macros-direct-core"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
df-derive-core = {{ path = "{}" }}
df-derive-macros = {{ path = "{}" }}
{}
"#,
        toml_path(&root.join("df-derive-core")),
        toml_path(&root.join("df-derive-macros")),
        polars_deps(),
    );

    check_fixture(
        "macros-direct-core",
        &manifest,
        r#"
use df_derive_core::dataframe::{ToDataFrame as _, ToDataFrameVec as _};
use df_derive_macros::ToDataFrame;

#[derive(ToDataFrame)]
struct Row {
    id: u32,
    label: String,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let row = Row { id: 7, label: "ok".into() };
    let df = row.to_dataframe()?;
    assert_eq!(df.shape(), (1, 2));

    let rows = vec![row];
    let batch = rows.as_slice().to_dataframe()?;
    assert_eq!(batch.shape(), (1, 2));
    Ok(())
}
"#,
    );
}

#[test]
fn renamed_facade_and_polars_dependencies_are_respected() {
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
pl = {{ package = "polars", version = "0.53.0", features = ["timezones", "dtype-decimal", "dtype-date", "dtype-time", "dtype-duration"] }}
pa = {{ package = "polars-arrow", version = "0.53.0" }}
"#,
        toml_path(root),
    );

    check_fixture(
        "renamed-facade",
        &manifest,
        r#"
use dfd::prelude::*;

#[derive(ToDataFrame)]
struct Row {
    id: u32,
    values: Vec<i64>,
}

fn main() -> pl::prelude::PolarsResult<()> {
    let rows = vec![Row { id: 1, values: vec![10, 20] }];
    let df = rows.as_slice().to_dataframe()?;
    assert_eq!(df.shape(), (1, 2));
    Ok(())
}
"#,
    );
}

#[test]
fn local_fallback_works_without_facade_or_core_dependencies() {
    let root = repo_root();
    let manifest = format!(
        r#"
[package]
name = "local-fallback"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
df-derive-macros = {{ path = "{}" }}
{}
"#,
        toml_path(&root.join("df-derive-macros")),
        polars_deps(),
    );

    check_fixture(
        "local-fallback",
        &manifest,
        r#"
use df_derive_macros::ToDataFrame;
use crate::core::dataframe::ToDataFrame as _;

mod core {
    pub mod dataframe {
        use polars::prelude::{DataFrame, DataType, PolarsResult};

        pub trait ToDataFrame {
            fn to_dataframe(&self) -> PolarsResult<DataFrame>;
            fn empty_dataframe() -> PolarsResult<DataFrame>;
            fn schema() -> PolarsResult<Vec<(String, DataType)>>;
        }

        pub trait Columnar: Sized {
            fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame> {
                let refs: Vec<&Self> = items.iter().collect();
                Self::columnar_from_refs(&refs)
            }

            fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame>;
        }

        pub trait Decimal128Encode {
            fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128>;
        }
    }
}

#[derive(ToDataFrame)]
struct Local {
    id: u32,
    name: String,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let df = Local { id: 1, name: "local".into() }.to_dataframe()?;
    assert_eq!(df.shape(), (1, 2));
    Ok(())
}
"#,
    );
}

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

fn paft_like_runtime_lib() -> &'static str {
    r#"
pub use df_derive_macros::ToDataFrame;

pub mod dataframe {
    use polars::prelude::{DataFrame, DataType, PolarsResult};

    #[doc(hidden)]
    pub mod __private {
        pub use polars;
        pub use pa as polars_arrow;
    }

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

    pub trait ToDataFrameVec {
        fn to_dataframe(&self) -> PolarsResult<DataFrame>;
    }

    impl<T> ToDataFrameVec for [T]
    where
        T: Columnar + ToDataFrame,
    {
        fn to_dataframe(&self) -> PolarsResult<DataFrame> {
            if self.is_empty() {
                return <T as ToDataFrame>::empty_dataframe();
            }
            <T as Columnar>::columnar_to_dataframe(self)
        }
    }

    pub trait Decimal128Encode {
        fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128>;
    }
}
"#
}

fn polars_deps() -> &'static str {
    r#"
polars = { version = "0.53.0", features = ["timezones", "dtype-decimal", "dtype-date", "dtype-datetime", "dtype-time", "dtype-duration"] }
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
fn facade_runtime_wins_over_paft_dependencies() {
    let root = package_root();
    let manifest = format!(
        r#"
[package]
name = "facade-wins-over-paft"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
df-derive = {{ path = "{}" }}
paft = {{ path = "paft" }}
paft-utils = {{ path = "paft-utils" }}
{}
"#,
        toml_path(root),
        polars_deps(),
    );

    check_fixture_with_files(
        "facade-wins-over-paft",
        &manifest,
        r#"
use df_derive::prelude::*;

#[derive(ToDataFrame)]
struct Row {
    id: u32,
}

fn assert_facade_runtime<T: df_derive::dataframe::ToDataFrame + df_derive::dataframe::Columnar>() {}

fn main() -> polars::prelude::PolarsResult<()> {
    assert_facade_runtime::<Row>();
    let df = Row { id: 1 }.to_dataframe()?;
    assert_eq!(df.shape(), (1, 1));
    Ok(())
}
"#,
        &[
            (
                "paft/Cargo.toml",
                r#"
[package]
name = "paft"
version = "0.0.0"
edition = "2024"
publish = false
"#,
            ),
            (
                "paft/src/lib.rs",
                r#"
pub mod dataframe {
    pub trait ToDataFrame {
        fn incompatible_paft_marker(&self);
    }

    pub trait Columnar {}
    pub trait Decimal128Encode {}
}
"#,
            ),
            (
                "paft-utils/Cargo.toml",
                r#"
[package]
name = "paft-utils"
version = "0.0.0"
edition = "2024"
publish = false
"#,
            ),
            (
                "paft-utils/src/lib.rs",
                r#"
pub mod dataframe {
    pub trait ToDataFrame {
        fn incompatible_paft_utils_marker(&self);
    }

    pub trait Columnar {}
    pub trait Decimal128Encode {}
}
"#,
            ),
        ],
    );
}

#[test]
fn core_runtime_wins_over_paft_utils_dependency() {
    let root = repo_root();
    let manifest = format!(
        r#"
[package]
name = "core-wins-over-paft-utils"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
df-derive-core = {{ path = "{}" }}
df-derive-macros = {{ path = "{}" }}
paft-utils = {{ path = "paft-utils" }}
{}
"#,
        toml_path(&root.join("df-derive-core")),
        toml_path(&root.join("df-derive-macros")),
        polars_deps(),
    );

    check_fixture_with_files(
        "core-wins-over-paft-utils",
        &manifest,
        r#"
use df_derive_core::dataframe::{ToDataFrame as _, ToDataFrameVec as _};
use df_derive_macros::ToDataFrame;

#[derive(ToDataFrame)]
struct Row {
    id: u32,
}

fn assert_core_runtime<T: df_derive_core::dataframe::ToDataFrame + df_derive_core::dataframe::Columnar>() {}

fn main() -> polars::prelude::PolarsResult<()> {
    assert_core_runtime::<Row>();
    let rows = vec![Row { id: 1 }];
    let df = rows.as_slice().to_dataframe()?;
    assert_eq!(df.shape(), (1, 1));
    Ok(())
}
"#,
        &[
            (
                "paft-utils/Cargo.toml",
                r#"
[package]
name = "paft-utils"
version = "0.0.0"
edition = "2024"
publish = false
"#,
            ),
            (
                "paft-utils/src/lib.rs",
                r#"
pub mod dataframe {
    pub trait ToDataFrame {
        fn incompatible_paft_utils_marker(&self);
    }

    pub trait Columnar {}
    pub trait Decimal128Encode {}
}
"#,
            ),
        ],
    );
}

#[test]
fn macros_direct_with_paft_utils_runtime_works_without_facade_or_core() {
    let root = repo_root();
    let manifest = format!(
        r#"
[package]
name = "macros-direct-paft-utils"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
df-derive-macros = {{ path = "{}" }}
paft-utils = {{ path = "paft-utils" }}
{}
"#,
        toml_path(&root.join("df-derive-macros")),
        polars_deps(),
    );

    check_fixture_with_files(
        "macros-direct-paft-utils",
        &manifest,
        r#"
use df_derive_macros::ToDataFrame;
use paft_utils::dataframe::{ToDataFrame as _, ToDataFrameVec as _};

#[derive(ToDataFrame)]
struct Row {
    id: u32,
    label: String,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let row = Row { id: 7, label: "paft-utils".into() };
    let df = row.to_dataframe()?;
    assert_eq!(df.shape(), (1, 2));

    let rows = vec![row];
    let batch = rows.as_slice().to_dataframe()?;
    assert_eq!(batch.shape(), (1, 2));
    Ok(())
}
"#,
        &[
            (
                "paft-utils/Cargo.toml",
                r#"
[package]
name = "paft-utils"
version = "0.0.0"
edition = "2024"
publish = false

[dependencies]
polars = { version = "0.53.0", default-features = false }
polars-arrow = { version = "0.53.0", default-features = false }
"#,
            ),
            (
                "paft-utils/src/lib.rs",
                r#"
pub mod dataframe {
    use polars::prelude::{DataFrame, DataType, PolarsResult};

    #[doc(hidden)]
    pub mod __private {
        pub use polars;
        pub use polars_arrow;
    }

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

    pub trait ToDataFrameVec {
        fn to_dataframe(&self) -> PolarsResult<DataFrame>;
    }

    impl<T> ToDataFrameVec for [T]
    where
        T: Columnar + ToDataFrame,
    {
        fn to_dataframe(&self) -> PolarsResult<DataFrame> {
            if self.is_empty() {
                return <T as ToDataFrame>::empty_dataframe();
            }
            <T as Columnar>::columnar_to_dataframe(self)
        }
    }

    pub trait Decimal128Encode {
        fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128>;
    }
}
"#,
            ),
        ],
    );
}

#[test]
fn paft_package_examples_use_library_runtime_path() {
    let root = repo_root();
    let manifest = format!(
        r#"
[package]
name = "paft"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
df-derive-macros = {{ path = "{}" }}
polars = {{ version = "0.53.0", features = ["timezones", "dtype-decimal", "dtype-date", "dtype-datetime", "dtype-time", "dtype-duration"] }}
pa = {{ package = "polars-arrow", version = "0.53.0" }}

[[example]]
name = "unannotated"
"#,
        toml_path(&root.join("df-derive-macros")),
    );

    check_fixture_with_files_and_args(
        "paft-package-example-runtime",
        &manifest,
        "fn main() {}",
        &[
            ("src/lib.rs", paft_like_runtime_lib()),
            (
                "examples/unannotated.rs",
                r#"
#[derive(Clone, paft::ToDataFrame)]
struct Row {
    id: u32,
}

fn assert_runtime<T: paft::dataframe::ToDataFrame + paft::dataframe::Columnar>() {}

fn main() -> polars::prelude::PolarsResult<()> {
    assert_runtime::<Row>();
    let df = paft::dataframe::ToDataFrame::to_dataframe(&Row { id: 1 })?;
    assert_eq!(df.shape(), (1, 1));
    Ok(())
}
"#,
            ),
        ],
        &["--example", "unannotated"],
    );
}

#[test]
fn paft_utils_package_examples_use_library_runtime_path() {
    let root = repo_root();
    let manifest = format!(
        r#"
[package]
name = "paft-utils"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
df-derive-macros = {{ path = "{}" }}
polars = {{ version = "0.53.0", features = ["timezones", "dtype-decimal", "dtype-date", "dtype-datetime", "dtype-time", "dtype-duration"] }}
pa = {{ package = "polars-arrow", version = "0.53.0" }}

[[example]]
name = "unannotated"
"#,
        toml_path(&root.join("df-derive-macros")),
    );

    check_fixture_with_files_and_args(
        "paft-utils-package-example-runtime",
        &manifest,
        "fn main() {}",
        &[
            ("src/lib.rs", paft_like_runtime_lib()),
            (
                "examples/unannotated.rs",
                r#"
#[derive(Clone, paft_utils::ToDataFrame)]
struct Row {
    id: u32,
}

fn assert_runtime<T: paft_utils::dataframe::ToDataFrame + paft_utils::dataframe::Columnar>() {}

fn main() -> polars::prelude::PolarsResult<()> {
    assert_runtime::<Row>();
    let df = paft_utils::dataframe::ToDataFrame::to_dataframe(&Row { id: 1 })?;
    assert_eq!(df.shape(), (1, 1));
    Ok(())
}
"#,
            ),
        ],
        &["--example", "unannotated"],
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
fn core_runtime_enables_supported_numeric_dtype_features_without_downstream_polars_flags() {
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
polars = {{ version = "0.53.0", default-features = false }}
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
fn renamed_facade_and_polars_dependency_is_respected_without_direct_arrow() {
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
pl = {{ package = "polars", version = "0.53.0", features = ["timezones", "dtype-decimal", "dtype-date", "dtype-datetime", "dtype-time", "dtype-duration"] }}
time_crate = {{ package = "chrono", version = "0.4.41" }}
"#,
        toml_path(root),
    );

    check_fixture(
        "renamed-facade",
        &manifest,
        r#"
use dfd::prelude::*;
use time_crate::{NaiveDate, NaiveTime};

#[derive(ToDataFrame)]
struct Row {
    id: u32,
    values: Vec<i64>,
    day: NaiveDate,
    at: NaiveTime,
}

fn main() -> pl::prelude::PolarsResult<()> {
    let rows = vec![Row {
        id: 1,
        values: vec![10, 20],
        day: NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
        at: NaiveTime::from_hms_opt(12, 34, 56).unwrap(),
    }];
    let df = rows.as_slice().to_dataframe()?;
    assert_eq!(df.shape(), (1, 4));
    Ok(())
}
"#,
    );
}

#[test]
fn explicit_scalar_custom_runtime_works_without_direct_arrow() {
    let root = repo_root();
    let manifest = format!(
        r#"
[package]
name = "explicit-scalar-custom-runtime-no-arrow"
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
        "explicit-scalar-custom-runtime-no-arrow",
        &manifest,
        r#"
use df_derive_macros::ToDataFrame;

mod runtime {
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

#[derive(ToDataFrame)]
#[df_derive(
    trait = "crate::runtime::ToDataFrame",
    columnar = "crate::runtime::Columnar"
)]
struct Row {
    id: u32,
    qty: i64,
    active: bool,
    price: f64,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let rows = vec![
        Row { id: 1, qty: 10, active: true, price: 9.5 },
        Row { id: 2, qty: 20, active: false, price: 19.25 },
    ];

    let single = runtime::ToDataFrame::to_dataframe(&rows[0])?;
    assert_eq!(single.shape(), (1, 4));

    let batch = runtime::Columnar::columnar_to_dataframe(rows.as_slice())?;
    assert_eq!(batch.shape(), (2, 4));
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
polars = {{ version = "0.53.0", features = ["timezones", "dtype-decimal", "dtype-date", "dtype-datetime", "dtype-time", "dtype-duration"] }}
pa = {{ package = "polars-arrow", version = "0.53.0" }}
"#,
        toml_path(&root.join("df-derive-macros")),
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

        #[doc(hidden)]
        pub mod __private {
            pub use polars;
            pub use pa as polars_arrow;
        }

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

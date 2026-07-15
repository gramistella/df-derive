// Architecture fixtures embed small downstream crates as raw strings; keeping
// those snippets readable is more useful than linting them as production code.
#![allow(
    clippy::missing_const_for_fn,
    clippy::needless_raw_string_hashes,
    clippy::too_many_lines
)]

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

fn paft_like_runtime_lib() -> String {
    [
        r#"
pub use df_derive_macros::ToDataFrame;

pub mod dataframe {
    use std::sync::Arc;

    use polars::prelude::{
        Column, DataFrame, DataType, PlSmallStr, PolarsResult, Schema, SchemaRef, polars_err,
    };

    #[doc(hidden)]
    pub mod __private {
        pub use polars;
        pub use pa as polars_arrow;
    }
"#,
        runtime_traits_source(),
        "}\n",
    ]
    .concat()
}

fn runtime_traits_source() -> &'static str {
    r#"
    pub struct EncodedBatch {
        height: usize,
        columns: Vec<Column>,
    }

    impl EncodedBatch {
        pub fn into_columns(self) -> Vec<Column> {
            self.columns
        }

        fn into_dataframe(self) -> PolarsResult<DataFrame> {
            DataFrame::new(self.height, self.columns)
        }
    }

    pub struct ColumnSink {
        schema: SchemaRef,
        columns: Vec<Column>,
        producer: &'static str,
    }

    #[must_use]
    pub struct ColumnSlot<'a> {
        columns: &'a mut Vec<Column>,
        producer: &'static str,
        index: usize,
        expected_name: &'a PlSmallStr,
        expected_dtype: &'a DataType,
    }

    impl ColumnSlot<'_> {
        #[inline]
        pub const fn name(&self) -> &PlSmallStr {
            self.expected_name
        }

        #[inline]
        pub const fn dtype(&self) -> &DataType {
            self.expected_dtype
        }

        #[inline]
        pub fn commit(self, column: Column) -> PolarsResult<()> {
            if column.name() != self.expected_name {
                return Err(polars_err!(
                    ComputeError:
                    "df-derive: ColumnarSpec for {} returned column `{}` at index {}, expected `{}`",
                    self.producer,
                    column.name(),
                    self.index,
                    self.expected_name,
                ));
            }
            if column.dtype() != self.expected_dtype {
                return Err(polars_err!(
                    ComputeError:
                    "df-derive: ColumnarSpec for {} returned dtype {:?} for column `{}` at index {}, expected {:?}",
                    self.producer,
                    column.dtype(),
                    column.name(),
                    self.index,
                    self.expected_dtype,
                ));
            }
            self.columns.push(column);
            Ok(())
        }
    }

    impl ColumnSink {
        fn new(schema: SchemaRef, producer: &'static str) -> Self {
            let columns = Vec::with_capacity(schema.len());
            Self {
                schema,
                columns,
                producer,
            }
        }

        #[inline]
        pub fn next_slot(&mut self) -> PolarsResult<ColumnSlot<'_>> {
            let Self { schema, columns, producer } = self;
            let index = columns.len();
            let Some((expected_name, expected_dtype)) = schema.get_at_index(index) else {
                return Err(polars_err!(
                    ComputeError:
                    "df-derive: ColumnarSpec for {} requested column slot at index {}, exceeding schema width {}",
                    producer,
                    index,
                    schema.len(),
                ));
            };
            Ok(ColumnSlot {
                columns,
                producer,
                index,
                expected_name,
                expected_dtype,
            })
        }

        fn finish(self, height: usize) -> PolarsResult<EncodedBatch> {
            if self.columns.len() != self.schema.len() {
                return Err(polars_err!(
                    ComputeError:
                    "df-derive: ColumnarSpec for {} returned schema width {}, expected {}",
                    self.producer,
                    self.columns.len(),
                    self.schema.len(),
                ));
            }

            for (index, column) in self.columns.iter().enumerate() {
                if column.len() != height {
                    return Err(polars_err!(
                        ComputeError:
                        "df-derive: ColumnarSpec for {} returned height {} for column `{}` at index {}, expected {}",
                        self.producer,
                        column.len(),
                        column.name(),
                        index,
                        height,
                    ));
                }
            }

            Ok(EncodedBatch {
                height,
                columns: self.columns,
            })
        }
    }

    /// Iterator boundary for generated encoders that may revisit yielded rows.
    ///
    /// # Safety
    ///
    /// `yielded` must equal the number of successful `next` calls. After
    /// `enable_replay`, `replay` must yield those items exactly once in source
    /// order. Generated exact-capacity writes rely on this contract.
    pub unsafe trait RowCursor: Iterator {
        type Replay<'cursor>: Iterator<Item = Self::Item>
        where
            Self: 'cursor;

        fn enable_replay(&mut self, capacity: usize);

        fn replay(&self) -> Self::Replay<'_>;

        fn yielded(&self) -> usize;
    }

    struct StreamingCursor<I>
    where
        I: Iterator,
    {
        inner: I,
        yielded: usize,
    }

    impl<I> StreamingCursor<I>
    where
        I: Iterator,
    {
        fn new(inner: I) -> Self {
            Self {
                inner,
                yielded: 0,
            }
        }
    }

    impl<I> Iterator for StreamingCursor<I>
    where
        I: Iterator,
        I::Item: Copy,
    {
        type Item = I::Item;

        fn next(&mut self) -> Option<Self::Item> {
            let item = self.inner.next()?;
            self.yielded += 1;
            Some(item)
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            self.inner.size_hint()
        }
    }

    unsafe impl<I> RowCursor for StreamingCursor<I>
    where
        I: Iterator,
        I::Item: Copy,
    {
        type Replay<'cursor>
            = std::iter::Empty<I::Item>
        where
            Self: 'cursor;

        fn enable_replay(&mut self, _capacity: usize) {
            panic!("df-derive: compact streaming cursor does not support row replay");
        }

        fn replay(&self) -> Self::Replay<'_> {
            std::iter::empty()
        }

        fn yielded(&self) -> usize {
            self.yielded
        }
    }

    struct ReplayStreamingCursor<I>
    where
        I: Iterator,
    {
        inner: I,
        yielded: usize,
        replay: Option<Vec<I::Item>>,
    }

    impl<I> ReplayStreamingCursor<I>
    where
        I: Iterator,
    {
        fn new(inner: I) -> Self {
            Self {
                inner,
                yielded: 0,
                replay: None,
            }
        }
    }

    impl<I> Iterator for ReplayStreamingCursor<I>
    where
        I: Iterator,
        I::Item: Copy,
    {
        type Item = I::Item;

        fn next(&mut self) -> Option<Self::Item> {
            let item = self.inner.next()?;
            match &mut self.replay {
                Some(replay) => replay.push(item),
                None => self.yielded += 1,
            }
            Some(item)
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            self.inner.size_hint()
        }
    }

    unsafe impl<I> RowCursor for ReplayStreamingCursor<I>
    where
        I: Iterator,
        I::Item: Copy,
    {
        type Replay<'cursor>
            = std::iter::Copied<std::slice::Iter<'cursor, I::Item>>
        where
            Self: 'cursor;

        fn enable_replay(&mut self, capacity: usize) {
            assert!(
                self.yielded == 0 && self.replay.is_none(),
                "df-derive: row replay must be enabled before iteration",
            );
            self.replay = Some(Vec::with_capacity(capacity));
        }

        fn replay(&self) -> Self::Replay<'_> {
            self.replay
                .as_ref()
                .expect("df-derive: row replay requested before it was enabled")
                .iter()
                .copied()
        }

        fn yielded(&self) -> usize {
            self.replay.as_ref().map_or(self.yielded, Vec::len)
        }
    }

    struct SliceCursor<'row, T> {
        original: &'row [T],
        remaining: std::slice::Iter<'row, T>,
    }

    impl<'row, T> SliceCursor<'row, T> {
        fn new(rows: &'row [T]) -> Self {
            Self {
                original: rows,
                remaining: rows.iter(),
            }
        }
    }

    impl<'row, T> Iterator for SliceCursor<'row, T> {
        type Item = &'row T;

        fn next(&mut self) -> Option<Self::Item> {
            self.remaining.next()
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            self.remaining.size_hint()
        }
    }

    unsafe impl<'row, T> RowCursor for SliceCursor<'row, T> {
        type Replay<'cursor>
            = std::slice::Iter<'row, T>
        where
            Self: 'cursor;

        fn enable_replay(&mut self, _capacity: usize) {}

        fn replay(&self) -> Self::Replay<'_> {
            let original: &'row [T] = self.original;
            original[..self.yielded()].iter()
        }

        fn yielded(&self) -> usize {
            self.original.len() - self.remaining.len()
        }
    }

    struct RefSliceCursor<'slice, 'row, T> {
        original: &'slice [&'row T],
        remaining: std::iter::Copied<std::slice::Iter<'slice, &'row T>>,
    }

    impl<'slice, 'row, T> RefSliceCursor<'slice, 'row, T> {
        fn new(rows: &'slice [&'row T]) -> Self {
            Self {
                original: rows,
                remaining: rows.iter().copied(),
            }
        }
    }

    impl<'row, T> Iterator for RefSliceCursor<'_, 'row, T> {
        type Item = &'row T;

        fn next(&mut self) -> Option<Self::Item> {
            self.remaining.next()
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            self.remaining.size_hint()
        }
    }

    unsafe impl<'slice, 'row, T> RowCursor for RefSliceCursor<'slice, 'row, T> {
        type Replay<'cursor>
            = std::iter::Copied<std::slice::Iter<'slice, &'row T>>
        where
            Self: 'cursor;

        fn enable_replay(&mut self, _capacity: usize) {}

        fn replay(&self) -> Self::Replay<'_> {
            let original: &'slice [&'row T] = self.original;
            original[..self.yielded()].iter().copied()
        }

        fn yielded(&self) -> usize {
            self.original.len() - self.remaining.len()
        }
    }

    pub trait ColumnarSpec: Sized {
        const REQUIRES_ROW_REPLAY: bool = false;

        fn build_schema() -> PolarsResult<SchemaRef>;

        fn encode_columns<'a, I>(
            rows: &mut I,
            sink: &mut ColumnSink,
        ) -> PolarsResult<()>
        where
            Self: 'a,
            I: RowCursor<Item = &'a Self>;
    }

    #[allow(
        clippy::inline_always,
        reason = "the cursor type must specialize away before entering generated row loops"
    )]
    #[inline(always)]
    fn encode_streaming_cursor<'row, T, C>(
        mut rows: C,
        sink: &mut ColumnSink,
    ) -> PolarsResult<usize>
    where
        T: ColumnarSpec + 'row,
        C: RowCursor<Item = &'row T>,
    {
        T::encode_columns(&mut rows, sink)?;
        rows.by_ref().for_each(drop);
        Ok(rows.yielded())
    }

    pub trait Columnar: ColumnarSpec {
        fn encode_batch<'a, R>(rows: R) -> PolarsResult<EncodedBatch>
        where
            Self: 'a,
            R: IntoIterator<Item = &'a Self>,
        {
            let schema = <Self as ColumnarSpec>::build_schema()?;
            let mut sink = ColumnSink::new(schema, std::any::type_name::<Self>());
            let rows = rows.into_iter();
            let height = if <Self as ColumnarSpec>::REQUIRES_ROW_REPLAY {
                encode_streaming_cursor::<Self, _>(ReplayStreamingCursor::new(rows), &mut sink)?
            } else {
                encode_streaming_cursor::<Self, _>(StreamingCursor::new(rows), &mut sink)?
            };

            sink.finish(height)
        }

        fn encode_slice(rows: &[Self]) -> PolarsResult<EncodedBatch> {
            let schema = <Self as ColumnarSpec>::build_schema()?;
            let mut sink = ColumnSink::new(schema, std::any::type_name::<Self>());
            let mut rows_cursor = SliceCursor::new(rows);

            <Self as ColumnarSpec>::encode_columns(&mut rows_cursor, &mut sink)?;

            sink.finish(rows.len())
        }

        fn encode_ref_batch<'row>(rows: &[&'row Self]) -> PolarsResult<EncodedBatch>
        where
            Self: 'row,
        {
            let schema = <Self as ColumnarSpec>::build_schema()?;
            let mut sink = ColumnSink::new(schema, std::any::type_name::<Self>());
            let mut rows_cursor = RefSliceCursor::new(rows);

            <Self as ColumnarSpec>::encode_columns(&mut rows_cursor, &mut sink)?;
            rows_cursor.by_ref().for_each(drop);

            sink.finish(rows.len())
        }

        fn encode<'a, R>(rows: R) -> PolarsResult<DataFrame>
        where
            Self: 'a,
            R: IntoIterator<Item = &'a Self>,
        {
            Self::encode_batch(rows)?.into_dataframe()
        }
    }

    impl<T: ColumnarSpec> Columnar for T {}

    pub trait ToDataFrame: Columnar {
        fn to_dataframe(&self) -> PolarsResult<DataFrame> {
            Self::encode_slice(std::slice::from_ref(self))?.into_dataframe()
        }

        fn empty_dataframe() -> PolarsResult<DataFrame> {
            Self::encode_slice(&[])?.into_dataframe()
        }

        fn schema() -> PolarsResult<SchemaRef> {
            <Self as ColumnarSpec>::build_schema()
        }
    }

    impl<T: Columnar> ToDataFrame for T {}

    pub trait ToDataFrameVec {
        fn to_dataframe(&self) -> PolarsResult<DataFrame>;
    }

    impl<T> ToDataFrameVec for [T]
    where
        T: Columnar,
    {
        fn to_dataframe(&self) -> PolarsResult<DataFrame> {
            <T as Columnar>::encode_slice(self)?.into_dataframe()
        }
    }

    impl ColumnarSpec for () {
        fn build_schema() -> PolarsResult<SchemaRef> {
            Ok(Arc::new(Schema::default()))
        }

        fn encode_columns<'a, I>(
            _rows: &mut I,
            _sink: &mut ColumnSink,
        ) -> PolarsResult<()>
        where
            Self: 'a,
            I: RowCursor<Item = &'a Self>,
        {
            Ok(())
        }
    }

    pub trait Decimal128Encode {
        fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128>;
    }
"#
}

fn paft_utils_runtime_lib() -> String {
    [
        r#"
pub mod dataframe {
    use std::sync::Arc;

    use polars::prelude::{
        Column, DataFrame, DataType, PlSmallStr, PolarsResult, Schema, SchemaRef, polars_err,
    };

    #[doc(hidden)]
    pub mod __private {
        pub use polars;
        pub use polars_arrow;
    }
"#,
        runtime_traits_source(),
        "}\n",
    ]
    .concat()
}

fn explicit_runtime_module_source() -> String {
    [
        r#"
mod runtime {
    use std::sync::Arc;

    use polars::prelude::{
        Column, DataFrame, DataType, PlSmallStr, PolarsResult, Schema, SchemaRef, polars_err,
    };
"#,
        runtime_traits_source(),
        "}\n",
    ]
    .concat()
}

fn local_runtime_module_source() -> String {
    [
        r#"
mod core {
    pub mod dataframe {
        use std::sync::Arc;

        use polars::prelude::{
            Column, DataFrame, DataType, PlSmallStr, PolarsResult, Schema, SchemaRef, polars_err,
        };

        #[doc(hidden)]
        pub mod __private {
            pub use polars;
            pub use pa as polars_arrow;
        }
"#,
        runtime_traits_source(),
        "    }\n}\n",
    ]
    .concat()
}

fn polars_deps() -> &'static str {
    r#"
polars = { version = "0.54", features = ["timezones", "dtype-decimal", "dtype-date", "dtype-datetime", "dtype-time", "dtype-duration"] }
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

    let paft_utils_runtime = paft_utils_runtime_lib();

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
polars = { version = "0.54", default-features = false }
polars-arrow = { version = "0.54", default-features = false }
"#,
            ),
            ("paft-utils/src/lib.rs", paft_utils_runtime.as_str()),
        ],
    );
}

#[test]
fn paft_package_examples_use_library_runtime_path() {
    let root = repo_root();
    let runtime = paft_like_runtime_lib();
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
polars = {{ version = "0.54", features = ["timezones", "dtype-decimal", "dtype-date", "dtype-datetime", "dtype-time", "dtype-duration"] }}
pa = {{ package = "polars-arrow", version = "0.54" }}

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
            ("src/lib.rs", runtime.as_str()),
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
    let runtime = paft_like_runtime_lib();
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
polars = {{ version = "0.54", features = ["timezones", "dtype-decimal", "dtype-date", "dtype-datetime", "dtype-time", "dtype-duration"] }}
pa = {{ package = "polars-arrow", version = "0.54" }}

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
            ("src/lib.rs", runtime.as_str()),
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
fn explicit_core_runtime_path_uses_hidden_dependency_reexports() {
    let root = repo_root();
    let manifest = format!(
        r#"
[package]
name = "explicit-core-runtime-hidden-deps"
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
        "explicit-core-runtime-hidden-deps",
        &manifest,
        r#"
use df_derive_core::dataframe::{ToDataFrame as _, ToDataFrameVec as _};
use df_derive_macros::ToDataFrame;

#[derive(ToDataFrame)]
#[df_derive(
    trait = "df_derive_core::dataframe::ToDataFrame",
    columnar = "df_derive_core::dataframe::Columnar"
)]
struct Row {
    labels: Vec<String>,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let rows = vec![Row {
        labels: vec!["bid".to_owned(), "ask".to_owned()],
    }];
    let df = rows.as_slice().to_dataframe()?;
    assert_eq!(df.shape(), (1, 1));
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
polars = {{ version = "0.54", default-features = false }}
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
        schema
            .iter()
            .map(|(name, dtype)| (name.as_str(), dtype.clone()))
            .collect::<Vec<_>>(),
        vec![
            ("i8_v", DataType::Int8),
            ("i16_v", DataType::Int16),
            ("i128_v", DataType::Int128),
            ("u8_v", DataType::UInt8),
            ("u16_v", DataType::UInt16),
            ("u128_v", DataType::UInt128),
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
pl = {{ package = "polars", version = "0.54", features = ["timezones", "dtype-decimal", "dtype-date", "dtype-datetime", "dtype-time", "dtype-duration"] }}
time_crate = {{ package = "chrono", version = "0.4" }}
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

#[derive(ToDataFrame)]
#[df_derive(trait = "dfd::dataframe::ToDataFrame")]
struct ExplicitRuntimeRow {
    labels: Vec<String>,
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

    let explicit_rows = vec![ExplicitRuntimeRow {
        labels: vec!["bid".to_owned(), "ask".to_owned()],
    }];
    let explicit_df = explicit_rows.as_slice().to_dataframe()?;
    assert_eq!(explicit_df.shape(), (1, 1));
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

    let runtime = explicit_runtime_module_source();
    let main_rs = [
        r#"
use df_derive_macros::ToDataFrame;
"#,
        runtime.as_str(),
        r#"
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

    let batch = runtime::Columnar::encode(rows.as_slice())?;
    assert_eq!(batch.shape(), (2, 4));
    Ok(())
}
"#,
    ]
    .concat();

    check_fixture(
        "explicit-scalar-custom-runtime-no-arrow",
        &manifest,
        &main_rs,
    );
}

#[test]
fn explicit_custom_runtime_decimal_tuple_does_not_need_reference_encode_impl() {
    let root = repo_root();
    let manifest = format!(
        r#"
[package]
name = "explicit-custom-runtime-decimal-tuple"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
df-derive-macros = {{ path = "{}" }}
polars = {{ version = "0.54", default-features = false, features = ["dtype-decimal"] }}
polars-arrow = {{ version = "0.54", default-features = false }}
rust_decimal = "1.42"
"#,
        toml_path(&root.join("df-derive-macros")),
    );

    let runtime = explicit_runtime_module_source();
    let main_rs = [
        r#"
use df_derive_macros::ToDataFrame;
use polars::prelude::{AnyValue, DataFrame, DataType, PolarsResult};
use rust_decimal::Decimal;
"#,
        runtime.as_str(),
        r#"
impl runtime::Decimal128Encode for Decimal {
    fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128> {
        let source_scale = self.scale();
        if source_scale > target_scale {
            return None;
        }
        self.mantissa().checked_mul(10i128.pow(target_scale - source_scale))
    }
}

#[derive(ToDataFrame, Clone)]
#[df_derive(
    trait = "crate::runtime::ToDataFrame",
    columnar = "crate::runtime::Columnar",
    decimal128_encode = "crate::runtime::Decimal128Encode",
)]
struct Row {
    maybe: Option<(Decimal,)>,
}

fn main() -> PolarsResult<()> {
    let rows = vec![
        Row {
            maybe: Some((Decimal::new(123, 2),)),
        },
        Row { maybe: None },
    ];
    let df = runtime::Columnar::encode(rows.as_slice())?;
    assert_eq!(df.shape(), (2, 1));
    assert_eq!(df.column("maybe.field_0")?.dtype(), &DataType::Decimal(38, 10));
    assert_eq!(
        df.column("maybe.field_0")?.get(0)?,
        AnyValue::Decimal(12_300_000_000, 38, 10),
    );
    assert_eq!(df.column("maybe.field_0")?.get(1)?, AnyValue::Null);
    Ok(())
}
"#,
    ]
    .concat();

    check_fixture("explicit-custom-runtime-decimal-tuple", &manifest, &main_rs);
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
polars = {{ version = "0.54", features = ["timezones", "dtype-decimal", "dtype-date", "dtype-datetime", "dtype-time", "dtype-duration"] }}
pa = {{ package = "polars-arrow", version = "0.54" }}
"#,
        toml_path(&root.join("df-derive-macros")),
    );

    let runtime = local_runtime_module_source();
    let main_rs = [
        r#"
use df_derive_macros::ToDataFrame;
use crate::core::dataframe::ToDataFrame as _;
"#,
        runtime.as_str(),
        r#"
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
    ]
    .concat();

    check_fixture("local-fallback", &manifest, &main_rs);
}

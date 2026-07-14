//! User-facing facade for deriving Polars `DataFrame` conversions.
//!
//! Most users should depend on this crate, import the prelude, and derive
//! `ToDataFrame` without any runtime-path attributes:
//!
//! ```toml
//! [dependencies]
//! df-derive = "0.4"
//! polars = "0.54"
//! ```
//!
//! The default facade hides the macro's `polars-arrow` implementation
//! dependency behind `df_derive::dataframe`; custom runtimes still need to
//! provide their own compatible direct dependencies.
//!
//! ```ignore
//! use df_derive::prelude::*;
//!
//! #[derive(ToDataFrame)]
//! struct Trade {
//!     symbol: String,
//!     price: f64,
//!     size: u64,
//! }
//! ```
//!
//! The derive macro targets [`dataframe`] by default, which is re-exported
//! from `df-derive-core`. Power users can depend on `df-derive-macros`
//! directly or use `#[df_derive(trait = "...")]`,
//! `#[df_derive(columnar = "...")]`, and
//! `#[df_derive(decimal128_encode = "...")]` to target a custom runtime.
//! Explicit paths back to `df_derive::dataframe::ToDataFrame` or
//! `df_derive_core::dataframe::ToDataFrame` still use the default runtime's
//! hidden dependency re-exports.

// `polars` pulls a wide transitive dependency tree where multiple resolved
// versions are unavoidable. `clippy::multiple_crate_versions` is part of the
// `clippy::cargo` group `just lint` enables; allow it here so linting stays
// focused on this crate's own code.
#![allow(clippy::multiple_crate_versions)]

pub use df_derive_core::dataframe;
pub use df_derive_macros::ToDataFrame;

/// Common imports for normal users.
///
/// This includes the derive macro and the runtime traits. The trait
/// `ToDataFrame` is also exported as `ToDataFrameTrait` for code that wants
/// an unambiguous type-namespace name.
pub mod prelude {
    pub use crate::ToDataFrame;
    pub use crate::dataframe::{
        Columnar, Decimal128Encode, ToDataFrame, ToDataFrame as ToDataFrameTrait, ToDataFrameVec,
    };
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use crate::ToDataFrame;

    struct CountedRows<'a, T> {
        rows: &'a [T],
        into_iter_calls: &'a Cell<usize>,
    }

    impl<'a, T> IntoIterator for CountedRows<'a, T> {
        type Item = &'a T;
        type IntoIter = std::slice::Iter<'a, T>;

        fn into_iter(self) -> Self::IntoIter {
            self.into_iter_calls
                .set(self.into_iter_calls.get().saturating_add(1));
            self.rows.iter()
        }
    }

    #[derive(ToDataFrame)]
    struct SelfCrateRow {
        id: u32,
        label: String,
    }

    #[derive(ToDataFrame)]
    struct OptionalStringRow {
        label: Option<String>,
        flag: Option<bool>,
    }

    #[derive(Clone, Copy, ToDataFrame)]
    struct EmptyRow {}

    #[test]
    fn derive_uses_facade_runtime_inside_facade_crate() -> polars::prelude::PolarsResult<()> {
        use crate::dataframe::{Columnar as _, ToDataFrame as _, ToDataFrameVec as _};

        let row = SelfCrateRow {
            id: 1,
            label: "facade".to_owned(),
        };
        let single = row.to_dataframe()?;
        assert_eq!(single.shape(), (1, 2));

        let rows = [
            row,
            SelfCrateRow {
                id: 2,
                label: "self".to_owned(),
            },
        ];
        let batch = rows.as_slice().to_dataframe()?;
        assert_eq!(batch.shape(), (2, 2));

        let refs: Vec<&SelfCrateRow> = rows.iter().collect();
        let borrowed = SelfCrateRow::encode(refs.iter().copied())?;
        assert!(batch.equals(&borrowed));

        let into_iter_calls = Cell::new(0);
        let counted = SelfCrateRow::encode(CountedRows {
            rows: rows.as_slice(),
            into_iter_calls: &into_iter_calls,
        })?;
        assert!(batch.equals(&counted));
        assert_eq!(into_iter_calls.get(), 1);

        let filtered = SelfCrateRow::encode(rows.iter().filter(|row| row.id == 2))?;
        assert_eq!(filtered.shape(), (1, 2));
        assert_eq!(filtered.column("id")?.u32()?.get(0), Some(2));

        let optional_rows = [
            OptionalStringRow {
                label: Some("first".to_owned()),
                flag: Some(true),
            },
            OptionalStringRow {
                label: None,
                flag: None,
            },
            OptionalStringRow {
                label: Some("third".to_owned()),
                flag: Some(false),
            },
        ];
        let exact_optional = OptionalStringRow::encode(&optional_rows)?;
        let zero_lower_bound = OptionalStringRow::encode(optional_rows.iter().filter(|_| true))?;
        assert!(exact_optional.equals_missing(&zero_lower_bound));

        let positive_underestimate =
            ::core::iter::once(&optional_rows[0]).chain(optional_rows[1..].iter().filter(|_| true));
        assert_eq!(positive_underestimate.size_hint(), (1, Some(3)));
        let positive_underestimate = OptionalStringRow::encode(positive_underestimate)?;
        assert!(exact_optional.equals_missing(&positive_underestimate));

        let no_empty_rows: &[EmptyRow] = &[];
        assert_eq!(EmptyRow::encode(no_empty_rows)?.shape(), (0, 0));
        assert_eq!(EmptyRow::encode(&[EmptyRow {}][..])?.shape(), (1, 0));
        let empty_rows = [EmptyRow {}, EmptyRow {}, EmptyRow {}];
        let filtered_empty_rows = empty_rows
            .iter()
            .enumerate()
            .filter_map(|(index, row)| (index != 1).then_some(row));
        assert_eq!(EmptyRow::encode(filtered_empty_rows)?.shape(), (2, 0));

        let units = [(), (), (), ()];
        assert_eq!(
            <() as crate::dataframe::Columnar>::encode(units.as_slice())?.shape(),
            (4, 0),
        );

        let empty = SelfCrateRow::empty_dataframe()?;
        assert_eq!(SelfCrateRow::schema()?.as_ref(), empty.schema().as_ref());

        Ok(())
    }
}

use df_derive::ToDataFrame;
use polars::prelude::{DataFrame, DataType, PolarsResult};

// == SETUP 1: Use the shared `common` module for default traits ==
#[path = "../common.rs"]
mod common;
use common::dataframe as paft_traits; // Alias for clarity

// == SETUP 2: Define a completely separate custom trait ==
mod my_traits {
    use super::*; // Access PolarsResult, etc.

    // This is our "custom" trait.
    pub trait MyToDataFrame {
        fn to_dataframe(&self) -> PolarsResult<DataFrame>;
        fn empty_dataframe() -> PolarsResult<DataFrame>;
        fn schema() -> PolarsResult<Vec<(&'static str, DataType)>>;
    }

    /// Internal columnar trait mirrored from the main crate. Implemented by the derive macro.
    pub trait Columnar: Sized {
        /// # Errors
        /// Returns an error if `DataFrame` construction fails.
        fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame>;
    }

    // Extension trait for slices of types implementing our custom traits.
    pub trait MyToDataFrameVec {
        fn to_dataframe(&self) -> PolarsResult<DataFrame>;
    }

    impl<T> MyToDataFrameVec for [T]
    where
        T: Columnar + MyToDataFrame,
    {
        fn to_dataframe(&self) -> PolarsResult<DataFrame> {
            if self.is_empty() {
                return <T as MyToDataFrame>::empty_dataframe();
            }
            <T as Columnar>::columnar_to_dataframe(self)
        }
    }
}

// == STRUCT A: Uses the default path ==
// Note: The derive macro will find `paft` or `paft-core` first if they are dependencies.
// Since they aren't here, it will fall back to a path like `crate::core::dataframe`.
// The `#[df_derive]` attribute allows us to override this to use our test traits.
// To test the default logic properly, we must tell it to use our `common` module.
#[derive(ToDataFrame)]
#[df_derive(trait = "paft_traits::ToDataFrame")]
struct DefaultPath {
    id: i32,
}

// == STRUCT B: Uses the custom path via the #[df_derive] attribute ==
#[derive(ToDataFrame)]
#[df_derive(trait = "my_traits::MyToDataFrame")]
struct CustomPath {
    name: String,
}

// == STRUCT C: Uses explicit custom paths for both traits ==
#[derive(ToDataFrame)]
#[df_derive(trait = "my_traits::MyToDataFrame", columnar = "my_traits::Columnar")]
struct ExplicitPath {
    value: f64,
}

fn main() {
    println!("--- Verifying side-by-side trait implementations ---");

    // == TEST A: Verify the struct using the default path ==
    let default_instance = DefaultPath { id: 1 };
    // This line ONLY compiles if `impl paft_traits::ToDataFrame for DefaultPath` was generated.
    let df_default = paft_traits::ToDataFrame::to_dataframe(&default_instance).unwrap();
    assert_eq!(df_default.shape(), (1, 1));
    assert_eq!(df_default.get_column_names(), &["id"]);
    println!("✅ Default path (single) implementation works.");

    // == TEST B: Verify the struct using the custom path ==
    let custom_instance = CustomPath { name: "test".into() };
    // This line ONLY compiles if `impl my_traits::MyToDataFrame for CustomPath` was generated.
    let df_custom = my_traits::MyToDataFrame::to_dataframe(&custom_instance).unwrap();
    assert_eq!(df_custom.shape(), (1, 1));
    assert_eq!(df_custom.get_column_names(), &["name"]);
    println!("✅ Custom path (single) implementation works.");

    // This tests the auto-inferred `Columnar` trait path.
    use my_traits::MyToDataFrameVec;
    let custom_vec = vec![CustomPath { name: "a".into() }, CustomPath { name: "b".into() }];
    let df_custom_vec = custom_vec.as_slice().to_dataframe().unwrap();
    assert_eq!(df_custom_vec.shape(), (2, 1));
    assert_eq!(df_custom_vec.get_column_names(), &["name"]);
    println!("✅ Custom path (columnar) implementation works.");

    // == TEST C: Verify the struct using explicit custom paths ==
    let explicit_instance = ExplicitPath { value: 3.14 };
    // This line ONLY compiles if `impl my_traits::MyToDataFrame for ExplicitPath` was generated.
    let df_explicit = my_traits::MyToDataFrame::to_dataframe(&explicit_instance).unwrap();
    assert_eq!(df_explicit.shape(), (1, 1));
    assert_eq!(df_explicit.get_column_names(), &["value"]);
    println!("✅ Explicit path (single) implementation works.");
    
    // This tests the explicitly provided `Columnar` trait path.
    let explicit_vec = vec![ExplicitPath { value: 1.0 }, ExplicitPath { value: 2.0 }];
    let df_explicit_vec = explicit_vec.as_slice().to_dataframe().unwrap();
    assert_eq!(df_explicit_vec.shape(), (2, 1));
    assert_eq!(df_explicit_vec.get_column_names(), &["value"]);
    println!("✅ Explicit path (columnar) implementation works.");

    println!("\n✅ Side-by-side test successful!");
}
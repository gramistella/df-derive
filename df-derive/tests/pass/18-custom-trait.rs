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
        fn schema() -> PolarsResult<Vec<(String, DataType)>>;
    }

    /// Internal columnar trait mirrored from the main crate. Implemented by the derive macro.
    pub trait Columnar: Sized {
        /// # Errors
        /// Returns an error if `DataFrame` construction fails.
        fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame> {
            let refs: Vec<&Self> = items.iter().collect();
            Self::columnar_from_refs(&refs)
        }
        /// # Errors
        /// Returns an error if `DataFrame` construction fails.
        fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame>;
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

// == STRUCT A: Uses an explicit runtime path ==
// The `#[df_derive]` attribute lets us target a non-default runtime. The
// default facade/core/local fallback discovery is covered by integration
// tests in `tests/architecture.rs`.
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

#[derive(ToDataFrame)]
#[df_derive(trait = "my_traits::MyToDataFrame", columnar = "my_traits::Columnar")]
struct CustomInner {
    value: i32,
}

#[derive(ToDataFrame)]
#[df_derive(trait = "my_traits::MyToDataFrame", columnar = "my_traits::Columnar")]
struct CustomOuter {
    inner: CustomInner,
    inners: Vec<CustomInner>,
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

    // == TEST D: Verify nested schema/empty generation uses the explicit trait path ==
    let nested_schema = <CustomOuter as my_traits::MyToDataFrame>::schema().unwrap();
    assert_eq!(nested_schema.len(), 2);
    assert_eq!(nested_schema[0].0, "inner.value");
    assert_eq!(nested_schema[0].1, DataType::Int32);
    assert_eq!(nested_schema[1].0, "inners.value");
    assert_eq!(nested_schema[1].1, DataType::List(Box::new(DataType::Int32)));

    let empty_nested = <CustomOuter as my_traits::MyToDataFrame>::empty_dataframe().unwrap();
    assert_eq!(empty_nested.shape(), (0, 2));
    assert_eq!(empty_nested.get_column_names(), &["inner.value", "inners.value"]);

    let nested_instance = CustomOuter {
        inner: CustomInner { value: 1 },
        inners: vec![CustomInner { value: 10 }, CustomInner { value: 20 }],
    };
    let nested_df = my_traits::MyToDataFrame::to_dataframe(&nested_instance).unwrap();
    assert_eq!(nested_df.shape(), (1, 2));
    assert_eq!(nested_df.get_column_names(), &["inner.value", "inners.value"]);
    println!("✅ Custom path nested schema/empty implementation works.");

    println!("\n✅ Side-by-side test successful!");
}

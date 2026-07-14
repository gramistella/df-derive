use std::sync::Arc;

use df_derive::ToDataFrame;
use df_derive::dataframe::ToDataFrame as _;

#[derive(ToDataFrame)]
struct ConcreteSchemaRow {
    value: i64,
}

#[derive(ToDataFrame)]
struct GenericSchemaRow<T> {
    #[df_derive(as_str)]
    value: T,
}

#[test]
fn concrete_schema_reuses_the_cached_allocation() {
    let first = ConcreteSchemaRow::schema().unwrap();
    let second = ConcreteSchemaRow::schema().unwrap();

    assert!(Arc::ptr_eq(&first, &second));
}

#[test]
fn generic_schema_is_built_per_monomorphized_call() {
    let first = GenericSchemaRow::<String>::schema().unwrap();
    let second = GenericSchemaRow::<String>::schema().unwrap();
    let borrowed = GenericSchemaRow::<&str>::schema().unwrap();

    assert_eq!(first, second);
    assert_eq!(first, borrowed);
    assert!(!Arc::ptr_eq(&first, &second));
}

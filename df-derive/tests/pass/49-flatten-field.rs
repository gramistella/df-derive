use std::sync::Arc;

use df_derive::ToDataFrame;
use df_derive::dataframe::ToDataFrame;

#[derive(ToDataFrame, Clone)]
struct Inner {
    value: i32,
    label: String,
}

#[derive(ToDataFrame, Clone)]
struct GenericOuter<T> {
    #[df_derive(flatten)]
    payload: T,
    #[df_derive(flatten(prefix = "boxed"))]
    boxed: Box<Inner>,
    #[df_derive(flatten(prefix = "arc"))]
    arc: Arc<Inner>,
}

fn main() {
    let schema = GenericOuter::<Inner>::schema().unwrap();
    let names: Vec<_> = schema.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "value",
            "label",
            "boxed.value",
            "boxed.label",
            "arc.value",
            "arc.label",
        ],
    );
}

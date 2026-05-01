// Regression test: the derive must not force `T: Clone` on a generic
// payload type. Previously `impl_parts_with_bounds` pushed
// `::core::clone::Clone` onto every type parameter unconditionally, even
// though only one slow primitive-vec fallback ever cloned an element.
// That fallback now borrows from the for-loop binding directly, so the
// macro can be applied to a struct whose generic argument is not `Clone`.

use df_derive::ToDataFrame;
use polars::prelude::*;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::{Columnar, ToDataFrame, ToDataFrameVec};

// Nested-path payload: implements `ToDataFrame` + `Columnar`, deliberately
// NOT `Clone`. Used as the generic argument for fields without a transform
// (which route through the nested-struct strategy).
#[derive(Debug)]
struct NoClonePayload {
    value: i64,
}

impl ToDataFrame for NoClonePayload {
    fn to_dataframe(&self) -> PolarsResult<DataFrame> {
        DataFrame::new_infer_height(vec![Series::new("value".into(), &[self.value]).into()])
    }
    fn empty_dataframe() -> PolarsResult<DataFrame> {
        DataFrame::new_infer_height(vec![
            Series::new_empty("value".into(), &DataType::Int64).into(),
        ])
    }
    fn schema() -> PolarsResult<Vec<(String, DataType)>> {
        Ok(vec![("value".to_string(), DataType::Int64)])
    }
}

impl Columnar for NoClonePayload {
    fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame> {
        let vals: Vec<i64> = items.iter().map(|i| i.value).collect();
        DataFrame::new_infer_height(vec![Series::new("value".into(), &vals).into()])
    }
}

// Primitive-path payload: also NOT `Clone`, also implements `AsRef<str>` so
// it can be used with `#[df_derive(as_str)]`. The `as_str` transform routes
// the field through the primitive strategy, which was the path containing
// the `(*elem).clone()` that this test locks in as gone.
#[derive(Debug)]
struct NoCloneTag {
    label: String,
}

impl AsRef<str> for NoCloneTag {
    fn as_ref(&self) -> &str {
        &self.label
    }
}

impl ToDataFrame for NoCloneTag {
    fn to_dataframe(&self) -> PolarsResult<DataFrame> {
        DataFrame::new_infer_height(vec![
            Series::new("label".into(), &[self.label.as_str()]).into(),
        ])
    }
    fn empty_dataframe() -> PolarsResult<DataFrame> {
        DataFrame::new_infer_height(vec![
            Series::new_empty("label".into(), &DataType::String).into(),
        ])
    }
    fn schema() -> PolarsResult<Vec<(String, DataType)>> {
        Ok(vec![("label".to_string(), DataType::String)])
    }
}

impl Columnar for NoCloneTag {
    fn columnar_from_refs(items: &[&Self]) -> PolarsResult<DataFrame> {
        let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();
        DataFrame::new_infer_height(vec![Series::new("label".into(), &labels).into()])
    }
}

// Generic struct exercising the nested path across the wrapper shapes the
// derive supports. No `T: Clone` bound declared here, and the derive must
// not synthesize one.
#[derive(ToDataFrame)]
struct NestedHolder<T> {
    id: u32,
    payload: T,
    optional: Option<T>,
    listed: Vec<T>,
    optional_list: Option<Vec<T>>,
    list_optional: Vec<Option<T>>,
    list_list: Vec<Vec<T>>,
}

// Same wrapper shapes, but every generic field carries `as_str` so it
// routes through the primitive strategy. The deep-tail shapes
// (`Vec<Option<T>>`, `Vec<Vec<T>>`) hit the recursive fallback that used to
// clone each element. `T: AsRef<str>` is required by the `as_str` attribute
// itself (validated by a per-field const-fn assert in the derive); `Clone`
// is deliberately absent to lock in the fix.
#[derive(ToDataFrame)]
struct PrimitiveHolder<T>
where
    T: AsRef<str>,
{
    id: u32,
    #[df_derive(as_str)]
    payload: T,
    #[df_derive(as_str)]
    optional: Option<T>,
    #[df_derive(as_str)]
    listed: Vec<T>,
    #[df_derive(as_str)]
    optional_list: Option<Vec<T>>,
    #[df_derive(as_str)]
    list_optional: Vec<Option<T>>,
    #[df_derive(as_str)]
    list_list: Vec<Vec<T>>,
}

fn main() {
    test_nested_path_no_clone();
    test_primitive_path_no_clone();
    println!("All non-Clone generic tests passed!");
}

fn test_nested_path_no_clone() {
    let h: NestedHolder<NoClonePayload> = NestedHolder {
        id: 1,
        payload: NoClonePayload { value: 10 },
        optional: Some(NoClonePayload { value: 20 }),
        listed: vec![NoClonePayload { value: 30 }, NoClonePayload { value: 40 }],
        optional_list: Some(vec![NoClonePayload { value: 50 }]),
        list_optional: vec![Some(NoClonePayload { value: 60 }), None],
        list_list: vec![vec![NoClonePayload { value: 70 }], vec![]],
    };

    let schema = NestedHolder::<NoClonePayload>::schema().unwrap();
    let names: Vec<&str> = schema.iter().map(|(n, _)| n.as_str()).collect();
    assert!(names.contains(&"id"));
    assert!(names.contains(&"payload.value"));
    assert!(names.contains(&"optional.value"));
    assert!(names.contains(&"listed.value"));

    let df = h.to_dataframe().unwrap();
    assert_eq!(df.shape().0, 1);
    assert_eq!(df.column("id").unwrap().get(0).unwrap(), AnyValue::UInt32(1));
    assert_eq!(
        df.column("payload.value").unwrap().get(0).unwrap(),
        AnyValue::Int64(10)
    );

    // Slice path goes through Columnar::columnar_from_refs.
    let items = vec![h];
    let batch = items.as_slice().to_dataframe().unwrap();
    assert_eq!(batch.shape().0, 1);

    // Empty slice round-trips through empty_dataframe.
    let empty: &[NestedHolder<NoClonePayload>] = &[];
    let _ = empty.to_dataframe().unwrap();
}

fn test_primitive_path_no_clone() {
    let h: PrimitiveHolder<NoCloneTag> = PrimitiveHolder {
        id: 1,
        payload: NoCloneTag {
            label: "p".to_string(),
        },
        optional: Some(NoCloneTag {
            label: "o".to_string(),
        }),
        listed: vec![
            NoCloneTag {
                label: "a".to_string(),
            },
            NoCloneTag {
                label: "b".to_string(),
            },
        ],
        optional_list: Some(vec![NoCloneTag {
            label: "c".to_string(),
        }]),
        list_optional: vec![
            Some(NoCloneTag {
                label: "d".to_string(),
            }),
            None,
            Some(NoCloneTag {
                label: "e".to_string(),
            }),
        ],
        list_list: vec![
            vec![
                NoCloneTag {
                    label: "f".to_string(),
                },
                NoCloneTag {
                    label: "g".to_string(),
                },
            ],
            vec![],
            vec![NoCloneTag {
                label: "h".to_string(),
            }],
        ],
    };

    let df = h.to_dataframe().unwrap();
    assert_eq!(df.shape().0, 1);
    assert_eq!(
        df.column("payload").unwrap().get(0).unwrap(),
        AnyValue::String("p")
    );

    // The deep-tail shapes are the ones that previously required `T: Clone`.
    let list_optional_av = df.column("list_optional").unwrap().get(0).unwrap();
    if let AnyValue::List(s) = list_optional_av {
        let vals: Vec<Option<&str>> = s.str().unwrap().into_iter().collect();
        assert_eq!(vals, vec![Some("d"), None, Some("e")]);
    } else {
        panic!("expected list for list_optional");
    }

    let list_list_av = df.column("list_list").unwrap().get(0).unwrap();
    if let AnyValue::List(outer) = list_list_av {
        let inner_lists: Vec<_> = outer.list().unwrap().into_iter().collect();
        assert_eq!(inner_lists.len(), 3);
        let row0: Vec<Option<&str>> = inner_lists[0]
            .as_ref()
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(row0, vec![Some("f"), Some("g")]);
        assert_eq!(inner_lists[1].as_ref().unwrap().len(), 0);
    } else {
        panic!("expected outer list for list_list");
    }

    // Slice path through the columnar entry point.
    let items = vec![h];
    let batch = items.as_slice().to_dataframe().unwrap();
    assert_eq!(batch.shape().0, 1);
}

// The derive must not require `T: Clone` of a generic payload. An early
// version added `Clone` to every type parameter, because one code path
// cloned an element. Values are encoded through references, so a struct
// whose generic argument is not `Clone` has to derive.

use df_derive::ToDataFrame;
use polars::prelude::*;
use df_derive::dataframe::{ToDataFrame, ToDataFrameVec};

// A nested-struct payload: derives `ToDataFrame`, deliberately NOT `Clone`.
// Used as the generic argument of fields that have no attribute.
#[derive(Debug)]
#[derive(ToDataFrame)]
struct NoClonePayload {
    value: i64,
}

// A string payload: also NOT `Clone`, and `AsRef<str>` so that it works with
// `#[df_derive(as_str)]`. That attribute is where the clone used to be.
#[derive(Debug, ToDataFrame)]
struct NoCloneTag {
    label: String,
}

impl AsRef<str> for NoCloneTag {
    fn as_ref(&self) -> &str {
        &self.label
    }
}

// A generic struct with the payload under every wrapper shape. No `T: Clone`
// bound is declared here, and the derive must not add one.
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

// The same wrapper shapes with `as_str` on every generic field. The two-layer
// shapes (`Vec<Option<T>>`, `Vec<Vec<T>>`) are the ones that used to clone
// each element. `T: AsRef<str>` is what the attribute itself needs; `Clone`
// is deliberately absent.
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

    // A slice of values.
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

    // The two-layer shapes are the ones that once required `T: Clone`.
    let list_optional_av = df.column("list_optional").unwrap().get(0).unwrap();
    if let AnyValue::List(s) = list_optional_av {
        let vals: Vec<Option<&str>> = s.str().unwrap().iter().collect();
        assert_eq!(vals, vec![Some("d"), None, Some("e")]);
    } else {
        panic!("expected list for list_optional");
    }

    let list_list_av = df.column("list_list").unwrap().get(0).unwrap();
    if let AnyValue::List(outer) = list_list_av {
        let inner_lists: Vec<_> = outer.list().unwrap().series_iter().collect();
        assert_eq!(inner_lists.len(), 3);
        let row0: Vec<Option<&str>> = inner_lists[0]
            .as_ref()
            .unwrap()
            .str()
            .unwrap()
            .iter()
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

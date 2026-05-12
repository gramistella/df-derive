use crate::core::dataframe::{ToDataFrame, ToDataFrameVec};
use df_derive::ToDataFrame;
use polars::prelude::*;

#[derive(ToDataFrame, Clone)]
struct Payload {
    value: i32,
}

mod domain {
    use super::*;

    #[derive(ToDataFrame, Clone)]
    pub struct Vec<T> {
        pub value: T,
    }

    #[derive(ToDataFrame, Clone)]
    pub struct Option<T> {
        pub value: T,
    }

    #[derive(ToDataFrame, Clone)]
    pub struct Box<T> {
        pub value: T,
    }

    #[derive(ToDataFrame, Clone)]
    pub struct Rc<T> {
        pub value: T,
    }

    #[derive(ToDataFrame, Clone)]
    pub struct Arc<T> {
        pub value: T,
    }

    #[derive(ToDataFrame, Clone)]
    pub struct Cow<T> {
        pub value: T,
    }
}

mod custom_as_str {
    pub struct Option<T>(pub T);

    impl<T> AsRef<str> for Option<T> {
        fn as_ref(&self) -> &str {
            "custom-option"
        }
    }
}

#[derive(ToDataFrame, Clone)]
struct UsesDomainWrappers {
    vec_named_type: domain::Vec<Payload>,
    option_named_type: domain::Option<Payload>,
    box_named_type: domain::Box<Payload>,
    rc_named_type: domain::Rc<Payload>,
    arc_named_type: domain::Arc<Payload>,
    cow_named_type: domain::Cow<Payload>,
}

struct NotAsRef;

#[derive(ToDataFrame)]
struct CustomWrapperAsStr {
    #[df_derive(as_str)]
    label: custom_as_str::Option<NotAsRef>,
}

#[test]
fn runtime_semantics() {
    let rows = vec![UsesDomainWrappers {
        vec_named_type: domain::Vec {
            value: Payload { value: 1 },
        },
        option_named_type: domain::Option {
            value: Payload { value: 2 },
        },
        box_named_type: domain::Box {
            value: Payload { value: 3 },
        },
        rc_named_type: domain::Rc {
            value: Payload { value: 4 },
        },
        arc_named_type: domain::Arc {
            value: Payload { value: 5 },
        },
        cow_named_type: domain::Cow {
            value: Payload { value: 6 },
        },
    }];

    let df = rows.as_slice().to_dataframe().unwrap();

    assert_eq!(df.shape(), (1, 6));
    assert_eq!(
        df.get_column_names(),
        [
            "vec_named_type.value.value",
            "option_named_type.value.value",
            "box_named_type.value.value",
            "rc_named_type.value.value",
            "arc_named_type.value.value",
            "cow_named_type.value.value",
        ]
    );

    assert_eq!(
        df.column("vec_named_type.value.value")
            .unwrap()
            .get(0)
            .unwrap(),
        AnyValue::Int32(1)
    );
    assert_eq!(
        df.column("option_named_type.value.value")
            .unwrap()
            .get(0)
            .unwrap(),
        AnyValue::Int32(2)
    );
    assert_eq!(
        df.column("box_named_type.value.value")
            .unwrap()
            .get(0)
            .unwrap(),
        AnyValue::Int32(3)
    );
    assert_eq!(
        df.column("rc_named_type.value.value")
            .unwrap()
            .get(0)
            .unwrap(),
        AnyValue::Int32(4)
    );
    assert_eq!(
        df.column("arc_named_type.value.value")
            .unwrap()
            .get(0)
            .unwrap(),
        AnyValue::Int32(5)
    );
    assert_eq!(
        df.column("cow_named_type.value.value")
            .unwrap()
            .get(0)
            .unwrap(),
        AnyValue::Int32(6)
    );
}

#[test]
fn custom_option_as_str_uses_outer_path_bound() {
    let df = CustomWrapperAsStr {
        label: custom_as_str::Option(NotAsRef),
    }
    .to_dataframe()
    .unwrap();

    assert_eq!(df.shape(), (1, 1));
    assert_eq!(df.column("label").unwrap().dtype(), &DataType::String);
    assert_eq!(
        df.column("label").unwrap().get(0).unwrap(),
        AnyValue::String("custom-option")
    );
}

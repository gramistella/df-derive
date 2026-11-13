// A builder is not tied to the instance of the row type that named it: not
// to its lifetimes, and not to the lifetimes inside its type arguments. So a
// type with lifetimes can be held by any other type, a generic type can be
// given a borrowed argument, and a projection can build rows that have both
// a lifetime and a type parameter.
//
// Every row below borrows from a local `String`, so no lifetime here is
// `'static`.

use std::collections::BTreeMap;

use df_derive::ToDataFrame;
use df_derive::dataframe::{Bitmap, ColumnBuilder, Columnar, Fields};
use df_derive::dataframe::{ToDataFrame, ToDataFrameVec};
use polars::prelude::*;

fn names(df: &DataFrame) -> Vec<String> {
    df.get_column_names()
        .iter()
        .map(|name| name.to_string())
        .collect()
}

fn string(df: &DataFrame, column: &str, row: usize) -> String {
    match df.column(column).unwrap().get(row).unwrap() {
        AnyValue::String(text) => text.to_owned(),
        AnyValue::StringOwned(text) => text.to_string(),
        other => panic!("{column}[{row}] is not a string: {other:?}"),
    }
}

fn strings(df: &DataFrame, column: &str, row: usize) -> Vec<String> {
    match df.column(column).unwrap().get(row).unwrap() {
        AnyValue::List(list) => list
            .iter()
            .map(|value| match value {
                AnyValue::String(text) => text.to_owned(),
                AnyValue::StringOwned(text) => text.to_string(),
                other => panic!("{column}[{row}] holds {other:?}"),
            })
            .collect(),
        other => panic!("{column}[{row}] is not a list: {other:?}"),
    }
}

// --- a lifetime and a type parameter, inside a type with only a lifetime -------

#[derive(ToDataFrame)]
struct Tagged<'a, T> {
    name: &'a str,
    value: T,
}

#[derive(ToDataFrame)]
struct Holder<'a> {
    id: u32,
    tagged: Tagged<'a, u32>,
    many: Vec<Tagged<'a, f64>>,
    maybe: Option<Tagged<'a, bool>>,
}

#[test]
fn a_generic_type_with_a_lifetime_nests_in_a_type_with_only_a_lifetime() {
    let first = String::from("alpha");
    let second = String::from("beta");
    let rows = [
        Holder {
            id: 1,
            tagged: Tagged {
                name: &first,
                value: 7,
            },
            many: vec![
                Tagged {
                    name: &first,
                    value: 0.5,
                },
                Tagged {
                    name: &second,
                    value: 1.5,
                },
            ],
            maybe: None,
        },
        Holder {
            id: 2,
            tagged: Tagged {
                name: &second,
                value: 8,
            },
            many: Vec::new(),
            maybe: Some(Tagged {
                name: &first,
                value: true,
            }),
        },
    ];

    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(
        names(&df),
        [
            "id",
            "tagged.name",
            "tagged.value",
            "many.name",
            "many.value",
            "maybe.name",
            "maybe.value"
        ]
    );
    assert_eq!(string(&df, "tagged.name", 0), "alpha");
    assert_eq!(string(&df, "tagged.name", 1), "beta");
    assert_eq!(
        df.column("tagged.value").unwrap().get(1).unwrap(),
        AnyValue::UInt32(8)
    );
    assert_eq!(strings(&df, "many.name", 0), ["alpha", "beta"]);
    assert_eq!(
        df.column("maybe.name").unwrap().get(0).unwrap(),
        AnyValue::Null
    );
    assert_eq!(string(&df, "maybe.name", 1), "alpha");
    assert_eq!(
        df.column("maybe.value").unwrap().get(1).unwrap(),
        AnyValue::Boolean(true)
    );
}

// --- the same, with every layer generic ----------------------------------------

#[derive(ToDataFrame)]
struct Outer<'a, T> {
    tagged: Tagged<'a, T>,
    borrowed: &'a T,
    #[df_derive(flatten)]
    flat: Tagged<'a, T>,
}

#[derive(ToDataFrame)]
struct Top<'a> {
    outer: Outer<'a, i64>,
}

#[test]
fn generic_types_with_lifetimes_nest_in_each_other() {
    let text = String::from("gamma");
    let number = 40_i64;
    let rows = [Top {
        outer: Outer {
            tagged: Tagged {
                name: &text,
                value: 1,
            },
            borrowed: &number,
            flat: Tagged {
                name: &text,
                value: 2,
            },
        },
    }];

    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(
        names(&df),
        [
            "outer.tagged.name",
            "outer.tagged.value",
            "outer.borrowed",
            "outer.name",
            "outer.value"
        ]
    );
    assert_eq!(
        df.column("outer.borrowed").unwrap().get(0).unwrap(),
        AnyValue::Int64(40)
    );
    assert_eq!(
        df.column("outer.value").unwrap().get(0).unwrap(),
        AnyValue::Int64(2)
    );

    // The generic layer is a row type itself, for any lifetime and payload.
    let direct = [Outer {
        tagged: Tagged {
            name: &text,
            value: text.as_str(),
        },
        borrowed: &text.as_str(),
        flat: Tagged {
            name: &text,
            value: text.as_str(),
        },
    }];
    let df = direct.as_slice().to_dataframe().unwrap();
    assert_eq!(string(&df, "tagged.value", 0), "gamma");
    assert_eq!(string(&df, "borrowed", 0), "gamma");
}

// --- two lifetimes, one bounded by the other, and an encoding on a borrow -------

#[derive(ToDataFrame)]
struct Labeled<'short, 'long: 'short, T: AsRef<str>> {
    #[df_derive(as_str)]
    label: &'short T,
    source: &'long str,
}

#[derive(ToDataFrame)]
struct Labels<'short, 'long: 'short> {
    labeled: Labeled<'short, 'long, String>,
}

#[test]
fn bounded_lifetimes_and_encoded_borrows_nest() {
    let source = String::from("feed");
    let label = String::from("delta");
    let rows = [Labels {
        labeled: Labeled {
            label: &label,
            source: &source,
        },
    }];

    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(names(&df), ["labeled.label", "labeled.source"]);
    assert_eq!(string(&df, "labeled.label", 0), "delta");
    assert_eq!(string(&df, "labeled.source", 0), "feed");
}

// --- a projected type that has a lifetime ----------------------------------------

#[derive(ToDataFrame)]
#[df_derive(via = "QuoteRow<'_>")]
struct Quote<'a> {
    symbol: &'a str,
    cents: i32,
}

#[derive(ToDataFrame)]
struct QuoteRow<'a> {
    symbol: &'a str,
    price: f64,
}

impl<'a, 'b> From<&'b Quote<'a>> for QuoteRow<'b> {
    fn from(quote: &'b Quote<'a>) -> Self {
        Self {
            symbol: quote.symbol,
            price: f64::from(quote.cents) / 100.0,
        }
    }
}

#[derive(ToDataFrame)]
struct Book<'a> {
    best: Quote<'a>,
    all: Vec<Quote<'a>>,
    last: Option<Quote<'a>>,
}

#[test]
fn a_projected_type_with_a_lifetime_nests_in_a_type_with_only_a_lifetime() {
    let symbol = String::from("ABC");
    let rows = [Book {
        best: Quote {
            symbol: &symbol,
            cents: 1_250,
        },
        all: vec![
            Quote {
                symbol: &symbol,
                cents: 100,
            },
            Quote {
                symbol: &symbol,
                cents: 200,
            },
        ],
        last: None,
    }];

    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(
        names(&df),
        [
            "best.symbol",
            "best.price",
            "all.symbol",
            "all.price",
            "last.symbol",
            "last.price"
        ]
    );
    assert_eq!(string(&df, "best.symbol", 0), "ABC");
    assert_eq!(
        df.column("best.price").unwrap().get(0).unwrap(),
        AnyValue::Float64(12.5)
    );
    assert_eq!(strings(&df, "all.symbol", 0), ["ABC", "ABC"]);
    assert_eq!(
        df.column("last.price").unwrap().get(0).unwrap(),
        AnyValue::Null
    );
}

// --- a projection whose row has a lifetime and a type parameter ------------------

// What the row type needs of `T` to be a row type is taken from the row type.
// Only what the conversion needs is said here.
#[derive(ToDataFrame)]
#[df_derive(via = "ReadingRow<'_, T>", bound = "T: Copy")]
struct Reading<T> {
    sensor: String,
    value: T,
}

#[derive(ToDataFrame)]
struct ReadingRow<'a, T> {
    sensor: &'a str,
    value: T,
    doubled: Option<T>,
}

impl<'a, T: Copy> From<&'a Reading<T>> for ReadingRow<'a, T> {
    fn from(reading: &'a Reading<T>) -> Self {
        Self {
            sensor: &reading.sensor,
            value: reading.value,
            doubled: None,
        }
    }
}

#[test]
fn a_projection_row_may_have_a_lifetime_and_a_type_parameter() {
    let rows = [
        Reading {
            sensor: "north".to_owned(),
            value: 1.5_f64,
        },
        Reading {
            sensor: "south".to_owned(),
            value: 2.5,
        },
    ];

    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(names(&df), ["sensor", "value", "doubled"]);
    assert_eq!(string(&df, "sensor", 1), "south");
    assert_eq!(
        df.column("value").unwrap().get(1).unwrap(),
        AnyValue::Float64(2.5)
    );
    assert_eq!(
        Reading::<u8>::schema().unwrap()[1],
        ("value".to_owned(), DataType::UInt8)
    );
}

// --- a hand-written builder for a type with a lifetime ---------------------------

struct Code<'a>(&'a str);

/// Encodes a `Code` of any lifetime: nothing ties the builder to the one in
/// `Code<'a>`.
struct CodeBuilder(<String as Columnar>::Builder);

impl<'a> ColumnBuilder<Code<'a>> for CodeBuilder {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        <<String as Columnar>::Builder as ColumnBuilder<str>>::schema(name, out);
    }

    fn new() -> Self {
        Self(ColumnBuilder::<str>::new())
    }

    fn reserve(&mut self, additional: usize) {
        ColumnBuilder::<str>::reserve(&mut self.0, additional);
    }

    fn push(&mut self, value: Option<&Code<'a>>) -> PolarsResult<()> {
        self.0.push(value.map(|code| code.0))
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        ColumnBuilder::<str>::finish(self.0, name, validity, out)
    }
}

impl Columnar for Code<'_> {
    type Builder = CodeBuilder;
}

#[derive(ToDataFrame)]
struct Coded<'a> {
    code: Code<'a>,
    codes: Vec<Code<'a>>,
}

#[test]
fn a_hand_written_builder_serves_every_lifetime() {
    let text = String::from("X1");
    let rows = [Coded {
        code: Code(&text),
        codes: vec![Code(&text), Code("Y2")],
    }];

    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(string(&df, "code", 0), "X1");
    assert_eq!(strings(&df, "codes", 0), ["X1", "Y2"]);
}

// --- a generic type whose argument is a borrow ------------------------------------

#[derive(ToDataFrame)]
struct Wrapper<T> {
    value: T,
}

#[derive(ToDataFrame)]
struct Wraps<'a> {
    one: Wrapper<&'a str>,
    many: Vec<Wrapper<Option<&'a str>>>,
    tagged: Tagged<'a, &'a str>,
}

#[test]
fn a_generic_type_takes_a_borrowed_argument() {
    let first = String::from("alpha");
    let second = String::from("beta");
    let rows = [Wraps {
        one: Wrapper { value: &first },
        many: vec![
            Wrapper {
                value: Some(&second),
            },
            Wrapper { value: None },
        ],
        tagged: Tagged {
            name: &first,
            value: &second,
        },
    }];

    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(
        names(&df),
        ["one.value", "many.value", "tagged.name", "tagged.value"]
    );
    assert_eq!(string(&df, "one.value", 0), "alpha");
    assert_eq!(string(&df, "tagged.value", 0), "beta");
    match df.column("many.value").unwrap().get(0).unwrap() {
        AnyValue::List(list) => {
            assert_eq!(list.len(), 2);
            assert_eq!(list.null_count(), 1);
        }
        other => panic!("many.value[0] is not a list: {other:?}"),
    }
}

// --- the same through a generic layer ---------------------------------------------

#[derive(ToDataFrame)]
struct Passes<'a, T> {
    wrapped: Wrapper<&'a T>,
    plain: Wrapper<T>,
    count: u32,
}

#[derive(ToDataFrame)]
struct Uses<'a> {
    numbers: Passes<'a, i64>,
    texts: Passes<'a, &'a str>,
}

#[test]
fn a_borrowed_argument_passes_through_a_generic_layer() {
    let text = String::from("gamma");
    let number = 7_i64;
    let borrowed = text.as_str();
    let rows = [Uses {
        numbers: Passes {
            wrapped: Wrapper { value: &number },
            plain: Wrapper { value: 8 },
            count: 1,
        },
        texts: Passes {
            wrapped: Wrapper { value: &borrowed },
            plain: Wrapper { value: &text },
            count: 2,
        },
    }];

    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(
        names(&df),
        [
            "numbers.wrapped.value",
            "numbers.plain.value",
            "numbers.count",
            "texts.wrapped.value",
            "texts.plain.value",
            "texts.count"
        ]
    );
    assert_eq!(
        df.column("numbers.wrapped.value").unwrap().get(0).unwrap(),
        AnyValue::Int64(7)
    );
    assert_eq!(string(&df, "texts.wrapped.value", 0), "gamma");
    assert_eq!(string(&df, "texts.plain.value", 0), "gamma");
}

// --- a projection onto a generic type with a borrowed argument --------------------

#[derive(ToDataFrame)]
#[df_derive(via = "Wrapper<&'_ str>")]
struct Owned {
    text: String,
}

impl<'a> From<&'a Owned> for Wrapper<&'a str> {
    fn from(owned: &'a Owned) -> Self {
        Self { value: &owned.text }
    }
}

#[test]
fn a_projection_row_may_be_a_generic_type_with_a_borrowed_argument() {
    let rows = [
        Owned {
            text: "one".to_owned(),
        },
        Owned {
            text: "two".to_owned(),
        },
    ];

    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(names(&df), ["value"]);
    assert_eq!(string(&df, "value", 1), "two");
}

// --- a projection row with two borrows, and one that keeps a lifetime -------------

#[derive(ToDataFrame)]
struct Pair<'a, 'b> {
    left: &'a str,
    right: &'b str,
}

#[derive(ToDataFrame)]
#[df_derive(via = "Pair<'_, '_>")]
struct Names {
    first: String,
    last: String,
}

impl<'a> From<&'a Names> for Pair<'a, 'a> {
    fn from(names: &'a Names) -> Self {
        Self {
            left: &names.first,
            right: &names.last,
        }
    }
}

#[derive(ToDataFrame)]
#[df_derive(via = "Pair<'a, '_>")]
struct Labelled<'a> {
    label: &'a str,
    text: String,
}

impl<'a, 'b> From<&'b Labelled<'a>> for Pair<'a, 'b> {
    fn from(labelled: &'b Labelled<'a>) -> Self {
        Self {
            left: labelled.label,
            right: &labelled.text,
        }
    }
}

#[derive(ToDataFrame)]
struct Sheet<'a> {
    labelled: Vec<Labelled<'a>>,
    names: Names,
}

#[test]
fn a_projection_row_may_borrow_twice_and_keep_a_lifetime_of_the_type() {
    let label = String::from("id");
    let rows = [Sheet {
        labelled: vec![Labelled {
            label: &label,
            text: "A1".to_owned(),
        }],
        names: Names {
            first: "Ada".to_owned(),
            last: "Lovelace".to_owned(),
        },
    }];

    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(
        names(&df),
        [
            "labelled.left",
            "labelled.right",
            "names.left",
            "names.right"
        ]
    );
    assert_eq!(strings(&df, "labelled.left", 0), ["id"]);
    assert_eq!(strings(&df, "labelled.right", 0), ["A1"]);
    assert_eq!(string(&df, "names.right", 0), "Lovelace");
}

// --- a hand-written builder for a generic wrapper ---------------------------------

struct Boxed<T>(T);

/// Generic over the builder of what the wrapper holds, not over what it
/// holds, so a lifetime inside `T` does not tie the builder either.
struct BoxedBuilder<B>(B);

impl<T, B: ColumnBuilder<T>> ColumnBuilder<Boxed<T>> for BoxedBuilder<B> {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        B::schema(name, out);
    }

    fn new() -> Self {
        Self(B::new())
    }

    fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);
    }

    fn push(&mut self, value: Option<&Boxed<T>>) -> PolarsResult<()> {
        self.0.push(value.map(|boxed| &boxed.0))
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        self.0.finish(name, validity, out)
    }
}

impl<T: Columnar> Columnar for Boxed<T> {
    type Builder = BoxedBuilder<T::Builder>;
}

#[derive(ToDataFrame)]
struct Crate<'a> {
    label: Boxed<&'a str>,
    wrapped: Wrapper<Boxed<&'a str>>,
}

#[test]
fn a_hand_written_generic_builder_takes_a_borrowed_argument() {
    let text = String::from("fragile");
    let rows = [Crate {
        label: Boxed(&text),
        wrapped: Wrapper {
            value: Boxed(&text),
        },
    }];

    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(names(&df), ["label", "wrapped.value"]);
    assert_eq!(string(&df, "wrapped.value", 0), "fragile");
}

// --- a builder that depends on no parameter serves every instance -----------------

#[derive(ToDataFrame)]
struct Sample<T: AsRef<str>> {
    #[df_derive(as_str)]
    label: T,
    weight: f64,
}

#[derive(ToDataFrame)]
struct Samples<'a> {
    borrowed: Sample<&'a str>,
    owned: Sample<String>,
}

#[test]
fn an_encoded_field_of_generic_type_does_not_tie_the_builder() {
    let text = String::from("x");
    let rows = [Samples {
        borrowed: Sample {
            label: &text,
            weight: 0.5,
        },
        owned: Sample {
            label: "y".to_owned(),
            weight: 1.5,
        },
    }];

    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(string(&df, "borrowed.label", 0), "x");
    assert_eq!(string(&df, "owned.label", 0), "y");
}

// --- the key of a map is a field of generic type like any other -------------------

enum Kind {
    Spot,
    Future,
}

impl AsRef<str> for Kind {
    fn as_ref(&self) -> &str {
        match self {
            Self::Spot => "spot",
            Self::Future => "future",
        }
    }
}

#[derive(ToDataFrame)]
struct Tally<'a, K> {
    #[df_derive(as_str)]
    kinds: BTreeMap<K, &'a Kind>,
}

#[derive(ToDataFrame)]
struct Tallies<'a> {
    by_name: Tally<'a, &'a str>,
    by_id: Tally<'a, u32>,
}

#[test]
fn a_map_key_of_generic_type_needs_no_bound_from_the_user() {
    let (spot, future) = (Kind::Spot, Kind::Future);
    let name = String::from("btc");
    let rows = [Tallies {
        by_name: Tally {
            kinds: BTreeMap::from([(name.as_str(), &spot)]),
        },
        by_id: Tally {
            kinds: BTreeMap::from([(1, &spot), (2, &future)]),
        },
    }];

    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(
        names(&df),
        [
            "by_name.kinds.key",
            "by_name.kinds.value",
            "by_id.kinds.key",
            "by_id.kinds.value"
        ]
    );
    assert_eq!(strings(&df, "by_name.kinds.key", 0), ["btc"]);
    assert_eq!(strings(&df, "by_id.kinds.value", 0), ["spot", "future"]);
}

// --- a lifetime that a field's type binds itself is not the row's ----------------

/// A count and something that is not stored.
struct Counted<F>(u32, F);

struct CountedBuilder(<u32 as Columnar>::Builder);

impl<F> ColumnBuilder<Counted<F>> for CountedBuilder {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        <<u32 as Columnar>::Builder as ColumnBuilder<u32>>::schema(name, out);
    }

    fn new() -> Self {
        Self(ColumnBuilder::<u32>::new())
    }

    fn reserve(&mut self, additional: usize) {
        ColumnBuilder::<u32>::reserve(&mut self.0, additional);
    }

    fn push(&mut self, value: Option<&Counted<F>>) -> PolarsResult<()> {
        self.0.push(value.map(|counted| &counted.0))
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        ColumnBuilder::<u32>::finish(self.0, name, validity, out)
    }
}

impl<F> Columnar for Counted<F> {
    type Builder = CountedBuilder;
}

#[derive(ToDataFrame)]
struct Hooks<'r> {
    name: &'r str,
    // `'a` is bound by the function type, so it stays as it is where the
    // row's own lifetime is written out of the builder.
    length: Counted<for<'a> fn(&'a str) -> usize>,
}

#[test]
fn a_lifetime_bound_inside_a_field_type_is_left_alone() {
    let name = String::from("len");
    let rows = [Hooks {
        name: &name,
        length: Counted(3, str::len),
    }];

    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(string(&df, "name", 0), "len");
    assert_eq!(
        df.column("length").unwrap().get(0).unwrap(),
        AnyValue::UInt32(3)
    );
    assert_eq!((rows[0].length.1)("four"), 4);
}

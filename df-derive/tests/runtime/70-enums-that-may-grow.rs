// The type of an `Enum` column is the list of its variants, which a new
// variant changes. An enum that may be given one asks with `categorical` for
// a `Categorical` column of the same names, whose type lists nothing. These
// tests state what the derive gives such an enum, that frames of two releases
// of one enum go together, what a type gets that asks for the column by hand,
// and what happens when the names or the positions are wrong.

use std::fmt;

use df_derive::ToDataFrame;
use df_derive::dataframe::builders::CategoricalVariantsBuilder;
use df_derive::dataframe::{Columnar, ToDataFrame, ToDataFrameVec, Variants};
use polars::prelude::*;

/// An order as a first release has it, and as a later one does, with a
/// status added between the two there were.
macro_rules! release {
    ($release:ident: $($Status:ident),+) => {
        mod $release {
            use df_derive::ToDataFrame;

            #[derive(ToDataFrame, Clone, Copy)]
            #[df_derive(categorical)]
            #[non_exhaustive]
            pub enum Status {
                $($Status,)+
            }

            #[derive(ToDataFrame)]
            pub struct Order {
                pub id: u32,
                pub status: Status,
                pub earlier: Option<Status>,
                pub history: Vec<Status>,
                pub boxed: Box<Status>,
            }

            /// The same enum as an `Enum` column: its list is its type.
            #[derive(ToDataFrame, Clone, Copy)]
            pub enum Fixed {
                $($Status,)+
            }

            #[derive(ToDataFrame)]
            pub struct Pinned {
                pub status: Fixed,
            }
        }
    };
}
release!(first: Open, Filled);
release!(later: Open, Partial, Filled);

fn first_orders() -> Vec<first::Order> {
    use first::{Order, Status};
    vec![
        Order {
            id: 1,
            status: Status::Filled,
            earlier: Some(Status::Open),
            history: vec![Status::Open, Status::Filled],
            boxed: Box::new(Status::Open),
        },
        Order {
            id: 2,
            status: Status::Open,
            earlier: None,
            history: vec![],
            boxed: Box::new(Status::Filled),
        },
    ]
}

fn later_orders() -> Vec<later::Order> {
    use later::{Order, Status};
    vec![
        Order {
            id: 3,
            status: Status::Partial,
            earlier: Some(Status::Open),
            history: vec![Status::Open, Status::Partial],
            boxed: Box::new(Status::Filled),
        },
        Order {
            id: 4,
            status: Status::Filled,
            earlier: Some(Status::Partial),
            history: vec![Status::Filled],
            boxed: Box::new(Status::Partial),
        },
    ]
}

/// The values of a column of names, as text.
fn strings(column: &Column) -> Vec<Option<String>> {
    column
        .cast(&DataType::String)
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .map(|name| name.map(str::to_owned))
        .collect()
}

fn text(values: &[Option<&str>]) -> Vec<Option<String>> {
    values
        .iter()
        .map(|value| value.map(str::to_owned))
        .collect()
}

/// The type of every categorical column df-derive makes.
fn categories() -> DataType {
    DataType::from_categories(Categories::global())
}

/// What a list column holds behind its lists.
fn inner(df: &DataFrame, name: &str) -> Column {
    df.column(name).unwrap().list().unwrap().get_inner().into()
}

#[test]
fn an_enum_that_may_grow_is_a_categorical_column() {
    let df = first_orders().as_slice().to_dataframe().unwrap();

    let status = df.column("status").unwrap();
    assert_eq!(status.dtype(), &categories());
    assert_eq!(strings(status), text(&[Some("Filled"), Some("Open")]));

    // The schema says the same without a row.
    let schema = first::Order::schema().unwrap();
    for (name, dtype) in &schema {
        assert_eq!(dtype, df.column(name).unwrap().dtype(), "column {name}");
    }
    let empty = first::Order::empty_dataframe().unwrap();
    assert_eq!(empty.height(), 0);
    assert_eq!(empty.schema(), df.schema());

    // The same enum without `categorical` keeps its list in its type.
    let pinned = [first::Pinned {
        status: first::Fixed::Filled,
    }]
    .as_slice()
    .to_dataframe()
    .unwrap();
    assert!(matches!(
        pinned.column("status").unwrap().dtype(),
        DataType::Enum(_, _)
    ));
}

#[test]
fn the_wrappers_of_a_value_are_those_of_any_other() {
    let df = first_orders().as_slice().to_dataframe().unwrap();

    assert_eq!(
        strings(df.column("earlier").unwrap()),
        text(&[Some("Open"), None])
    );
    assert_eq!(
        strings(df.column("boxed").unwrap()),
        text(&[Some("Open"), Some("Filled")])
    );
    let history = inner(&df, "history");
    assert_eq!(history.dtype(), &categories());
    assert_eq!(strings(&history), text(&[Some("Open"), Some("Filled")]));
}

#[test]
fn frames_of_two_releases_go_together() {
    let old = first_orders().as_slice().to_dataframe().unwrap();
    let new = later_orders().as_slice().to_dataframe().unwrap();

    // One type, whichever release the rows are of.
    assert_eq!(old.schema(), new.schema());
    let both = old.vstack(&new).unwrap();
    assert_eq!(
        strings(both.column("status").unwrap()),
        text(&[
            Some("Filled"),
            Some("Open"),
            Some("Partial"),
            Some("Filled"),
        ])
    );

    // They compare and join as they are.
    let joined = old
        .join(
            &new,
            ["status"],
            ["status"],
            JoinArgs::new(JoinType::Inner),
            None,
        )
        .unwrap();
    assert_eq!(joined.height(), 1);
    assert_eq!(
        strings(joined.column("status").unwrap()),
        text(&[Some("Filled")])
    );
}

#[test]
fn enum_columns_of_two_releases_do_not() {
    // This is what `categorical` is for. Polars takes the two lists for two
    // types and refuses to put their columns together.
    let old = [first::Fixed::Open, first::Fixed::Filled]
        .map(|status| first::Pinned { status })
        .as_slice()
        .to_dataframe()
        .unwrap();
    let new = [
        later::Fixed::Open,
        later::Fixed::Partial,
        later::Fixed::Filled,
    ]
    .map(|status| later::Pinned { status })
    .as_slice()
    .to_dataframe()
    .unwrap();

    assert_ne!(old.schema(), new.schema());
    assert!(old.vstack(&new).is_err());

    // Cast to the newer type, the older column goes with the newer one.
    let newer = new.column("status").unwrap().dtype();
    let cast = old.column("status").unwrap().cast(newer).unwrap();
    assert_eq!(cast.dtype(), newer);
    assert_eq!(strings(&cast), text(&[Some("Open"), Some("Filled")]));
}

/// An order whose status arrived as text, stored as categories.
#[derive(ToDataFrame)]
struct Report {
    #[df_derive(categorical)]
    status: String,
    venue: u8,
}

#[test]
fn the_names_are_categories_like_any_text() {
    let orders = later_orders().as_slice().to_dataframe().unwrap();
    let reports = [("Filled", 7), ("Cancelled", 8)]
        .map(|(status, venue)| Report {
            status: status.to_owned(),
            venue,
        })
        .as_slice()
        .to_dataframe()
        .unwrap();

    assert_eq!(
        orders.column("status").unwrap().dtype(),
        reports.column("status").unwrap().dtype()
    );
    let joined = orders
        .join(
            &reports,
            ["status"],
            ["status"],
            JoinArgs::new(JoinType::Inner),
            None,
        )
        .unwrap();
    assert_eq!(joined.column("id").unwrap().u32().unwrap().get(0), Some(4));
    assert_eq!(
        joined.column("venue").unwrap().u8().unwrap().get(0),
        Some(7)
    );
}

/// The names come from where they do for an `Enum` column: a variant's
/// identifier or `rename`, or the enum's own text.
#[derive(ToDataFrame, Clone, Copy)]
#[df_derive(categorical)]
enum Venue {
    #[df_derive(rename = "XNAS")]
    Nasdaq,
    Dark,
    #[allow(non_camel_case_types)]
    r#match,
}

#[derive(ToDataFrame, Clone, Copy)]
#[df_derive(as_str, categorical)]
#[non_exhaustive]
enum Side {
    Buy,
    Sell,
}

impl AsRef<str> for Side {
    fn as_ref(&self) -> &str {
        match self {
            Self::Buy => "BUY",
            Self::Sell => "SELL",
        }
    }
}

#[derive(ToDataFrame, Clone, Copy)]
#[non_exhaustive]
#[df_derive(as_string)]
#[df_derive(categorical)]
enum Interval {
    Minutes5,
    Hours1,
}

impl fmt::Display for Interval {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        out.write_str(match self {
            Self::Minutes5 => "5m",
            Self::Hours1 => "1h",
        })
    }
}

#[derive(ToDataFrame)]
struct Bar {
    venue: Venue,
    side: Side,
    interval: Interval,
}

#[test]
fn the_names_come_from_where_an_enum_column_takes_them() {
    let df = [
        Bar {
            venue: Venue::Nasdaq,
            side: Side::Sell,
            interval: Interval::Hours1,
        },
        Bar {
            venue: Venue::r#match,
            side: Side::Buy,
            interval: Interval::Minutes5,
        },
        Bar {
            venue: Venue::Dark,
            side: Side::Sell,
            interval: Interval::Minutes5,
        },
    ]
    .as_slice()
    .to_dataframe()
    .unwrap();

    for name in ["venue", "side", "interval"] {
        assert_eq!(df.column(name).unwrap().dtype(), &categories(), "{name}");
    }
    assert_eq!(
        strings(df.column("venue").unwrap()),
        text(&[Some("XNAS"), Some("match"), Some("Dark")])
    );
    assert_eq!(
        strings(df.column("side").unwrap()),
        text(&[Some("SELL"), Some("BUY"), Some("SELL")])
    );
    assert_eq!(
        strings(df.column("interval").unwrap()),
        text(&[Some("1h"), Some("5m"), Some("5m")])
    );

    // The derive still says what the variants are called and where each is.
    let names: Vec<String> = Venue::names()
        .into_iter()
        .map(|name| name.to_string())
        .collect();
    assert_eq!(names, ["XNAS", "Dark", "match"]);
    assert_eq!(Venue::r#match.position(), 2);
}

/// A column of more rows than its type has names is built another way than a
/// shorter one. Which way it was built does not show.
#[test]
fn a_column_is_the_same_however_many_rows_it_has() {
    let venues = [Venue::Dark, Venue::r#match, Venue::Nasdaq];
    let names = ["Dark", "match", "XNAS"];
    for rows in [0, 1, 2, 3, 4, 5, 64, 1_000] {
        let routed: Vec<Routed> = (0..rows)
            .map(|row| Routed {
                venue: venues[row % 3],
                earlier: (row % 4 != 1).then_some(venues[(row + 1) % 3]),
            })
            .collect();
        let df = routed.as_slice().to_dataframe().unwrap();

        assert_eq!(df.height(), rows);
        for name in ["venue", "earlier"] {
            assert_eq!(
                df.column(name).unwrap().dtype(),
                &categories(),
                "{rows} rows"
            );
        }
        let venue: Vec<Option<String>> = (0..rows)
            .map(|row| Some(names[row % 3].to_owned()))
            .collect();
        let earlier: Vec<Option<String>> = (0..rows)
            .map(|row| (row % 4 != 1).then(|| names[(row + 1) % 3].to_owned()))
            .collect();
        assert_eq!(strings(df.column("venue").unwrap()), venue, "{rows} rows");
        assert_eq!(
            strings(df.column("earlier").unwrap()),
            earlier,
            "{rows} rows"
        );
    }
}

#[derive(ToDataFrame)]
struct Routed {
    venue: Venue,
    earlier: Option<Venue>,
}

/// A type with three hundred values that asks for the column by hand. It is
/// not an enum: all `Variants` asks for is names and positions.
#[derive(Clone, Copy)]
struct Lot(usize);

impl Variants for Lot {
    fn names() -> impl IntoIterator<Item = impl fmt::Display> {
        (0..300).map(|lot| format!("lot-{lot:03}"))
    }

    fn position(&self) -> usize {
        self.0
    }
}

impl Columnar for Lot {
    type Builder = CategoricalVariantsBuilder<Self>;
}

#[derive(ToDataFrame)]
struct Fill {
    lot: Lot,
    spare: Option<Lot>,
}

#[test]
fn a_type_can_ask_for_the_column_by_hand() {
    let fills: Vec<Fill> = (0..1_000)
        .map(|row| Fill {
            lot: Lot((row * 7) % 300),
            spare: (row % 3 != 0).then_some(Lot(299 - row % 300)),
        })
        .collect();
    let df = fills.as_slice().to_dataframe().unwrap();

    let lot = strings(df.column("lot").unwrap());
    let spare = strings(df.column("spare").unwrap());
    for (row, fill) in fills.iter().enumerate() {
        assert_eq!(
            lot[row],
            Some(format!("lot-{:03}", fill.lot.0)),
            "row {row}"
        );
        assert_eq!(
            spare[row],
            fill.spare.map(|lot| format!("lot-{:03}", lot.0)),
            "row {row}"
        );
    }
}

#[test]
fn a_position_past_the_names_is_an_error() {
    let expected = format!(
        "df-derive: `{}::position` returned 300 in column `lot`, but `names` gives 300 names",
        std::any::type_name::<Lot>()
    );
    // As the first value of a column and after others, in a column of fewer
    // rows than the type has names and in one of more.
    for good in [0, 5, 400] {
        let fills: Vec<Fill> = (0..good)
            .map(|lot| Lot(lot % 300))
            .chain([Lot(300)])
            .map(|lot| Fill { lot, spare: None })
            .collect();
        let error = fills.as_slice().to_dataframe().unwrap_err().to_string();
        assert!(error.ends_with(&expected), "{error}");
    }
}

/// A type that gives two of its values one name.
struct Twin(usize);

impl Variants for Twin {
    fn names() -> impl IntoIterator<Item = impl fmt::Display> {
        ["left", "right", "left"]
    }

    fn position(&self) -> usize {
        self.0
    }
}

impl Columnar for Twin {
    type Builder = CategoricalVariantsBuilder<Self>;
}

#[derive(ToDataFrame)]
struct Pair {
    twin: Option<Twin>,
}

#[test]
fn names_that_repeat_are_an_error() {
    let expected = format!(
        "df-derive: two values of `{}` are named `left`: a name stands for one value only",
        std::any::type_name::<Twin>()
    );
    // Whatever the rows hold, a value or none: the names are those of the
    // type.
    for twin in [Some(Twin(1)), None] {
        let error = [Pair { twin }].as_slice().to_dataframe().unwrap_err();
        assert!(error.to_string().ends_with(&expected), "{error}");
    }
    assert!(Pair::empty_dataframe().is_err());
    assert_eq!(Pair::schema().unwrap_err().to_string(), expected);
}

/// A row that holds one value of any type.
#[derive(ToDataFrame)]
struct Holding<T> {
    held: T,
}

#[test]
fn names_that_repeat_are_an_error_inside_a_list_or_a_map() {
    use std::collections::BTreeMap;

    fn refused<T: ToDataFrame>(expected: &str) {
        assert!(T::empty_dataframe().is_err());
        assert_eq!(T::schema().unwrap_err().to_string(), expected);
    }

    let expected = format!(
        "df-derive: two values of `{}` are named `left`: a name stands for one value only",
        std::any::type_name::<Twin>()
    );
    // The column type is then a list of an unknown type. `schema` sees it
    // because Polars's `DataType::is_known` looks inside lists.
    refused::<Holding<Vec<Twin>>>(&expected);
    refused::<Holding<BTreeMap<String, Twin>>>(&expected);
    refused::<Holding<Option<Vec<Vec<Option<Twin>>>>>>(&expected);
}

/// An enum one of whose variants cannot be written as text.
#[derive(ToDataFrame)]
#[df_derive(as_string, categorical)]
enum Mute {
    Spoken,
    Silent,
}

impl fmt::Display for Mute {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spoken => out.write_str("spoken"),
            Self::Silent => Err(fmt::Error),
        }
    }
}

#[derive(ToDataFrame)]
struct Line {
    mute: Mute,
}

#[test]
fn a_name_that_cannot_be_formatted_is_an_error() {
    let expected = format!(
        "df-derive: a name of `{}` could not be formatted: its `Display` returned an error",
        std::any::type_name::<Mute>()
    );
    for mute in [Mute::Spoken, Mute::Silent] {
        let error = [Line { mute }].as_slice().to_dataframe().unwrap_err();
        assert!(error.to_string().ends_with(&expected), "{error}");
    }
    assert_eq!(Line::schema().unwrap_err().to_string(), expected);
}

/// A type that names its variants and is text by default. A field can ask
/// for the column of categories by naming the builder.
#[derive(Clone, Copy)]
enum Flag {
    Up,
    Down,
}

impl Variants for Flag {
    fn names() -> impl IntoIterator<Item = impl fmt::Display> {
        ["up", "down"]
    }

    fn position(&self) -> usize {
        *self as usize
    }
}

#[derive(ToDataFrame)]
struct Signal {
    #[df_derive(with = "CategoricalVariantsBuilder<Flag>")]
    flag: Flag,
    #[df_derive(with = "CategoricalVariantsBuilder<Flag>")]
    flags: Vec<Option<Flag>>,
}

#[test]
fn a_field_can_name_the_builder() {
    let df = [Signal {
        flag: Flag::Down,
        flags: vec![Some(Flag::Up), None],
    }]
    .as_slice()
    .to_dataframe()
    .unwrap();

    let flag = df.column("flag").unwrap();
    assert_eq!(flag.dtype(), &categories());
    assert_eq!(strings(flag), text(&[Some("down")]));
    assert_eq!(strings(&inner(&df, "flags")), text(&[Some("up"), None]));
}

#[test]
fn columns_built_on_many_threads_agree() {
    let frames: Vec<DataFrame> = std::thread::scope(|scope| {
        // Every thread is started before any is waited for.
        let mut threads = Vec::new();
        for thread in 0..8 {
            threads.push(scope.spawn(move || {
                let fills: Vec<Fill> = (0..5_000)
                    .map(|row| Fill {
                        lot: Lot((row * 11 + thread * 37) % 300),
                        spare: None,
                    })
                    .collect();
                fills.as_slice().to_dataframe().unwrap()
            }));
        }
        threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect()
    });

    for (thread, frame) in frames.iter().enumerate() {
        let lot = strings(frame.column("lot").unwrap());
        for row in [0, 1, 2_500, 4_999] {
            assert_eq!(
                lot[row],
                Some(format!("lot-{:03}", (row * 11 + thread * 37) % 300)),
                "thread {thread}, row {row}"
            );
        }
    }
    let mut all = frames[0].clone();
    for frame in &frames[1..] {
        all.vstack_mut(frame).unwrap();
    }
    assert_eq!(all.height(), 40_000);
    assert_eq!(all.column("lot").unwrap().n_unique().unwrap(), 300);
}

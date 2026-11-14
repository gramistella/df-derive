// An enum whose variants hold no data is one `Enum` column. The column's
// type lists the names of the variants, and a row holds the position of its
// variant among them. These tests state what the derive gives such an enum,
// where the names come from, what a type gets that names its variants by
// hand, and what happens when the names or the positions are wrong.

use std::fmt;

use df_derive::ToDataFrame;
use df_derive::dataframe::builders::EnumBuilder;
use df_derive::dataframe::{Columnar, ToDataFrame, ToDataFrameVec, Variants};
use polars::prelude::*;

/// Declared in an order that is not the alphabet's.
#[derive(ToDataFrame, Clone, Copy, Debug, PartialEq)]
enum Size {
    Small,
    Medium,
    Large,
}

#[derive(ToDataFrame, Clone, Copy)]
enum Venue {
    #[df_derive(rename = "XNAS")]
    Nasdaq,
    #[df_derive(rename = "XNYS")]
    Nyse,
    Dark,
    // A variant is the same value however its lack of data is written.
    Auction {},
    Internal(),
    #[allow(non_camel_case_types)]
    r#match,
}

#[derive(ToDataFrame)]
struct Order {
    size: Size,
    venue: Venue,
    maybe: Option<Size>,
    history: Vec<Size>,
    legs: Vec<Option<Venue>>,
    pair: [Size; 2],
    boxed: Box<Size>,
    again: Size,
}

fn orders() -> Vec<Order> {
    vec![
        Order {
            size: Size::Large,
            venue: Venue::Nasdaq,
            maybe: Some(Size::Small),
            history: vec![Size::Small, Size::Medium],
            legs: vec![Some(Venue::Dark), None, Some(Venue::r#match)],
            pair: [Size::Small, Size::Large],
            boxed: Box::new(Size::Medium),
            again: Size::Large,
        },
        Order {
            size: Size::Small,
            venue: Venue::Auction {},
            maybe: None,
            history: vec![],
            legs: vec![],
            pair: [Size::Medium, Size::Medium],
            boxed: Box::new(Size::Small),
            again: Size::Small,
        },
        Order {
            size: Size::Medium,
            venue: Venue::Internal(),
            maybe: Some(Size::Large),
            history: vec![Size::Large],
            legs: vec![Some(Venue::Nyse)],
            pair: [Size::Large, Size::Small],
            boxed: Box::new(Size::Large),
            again: Size::Medium,
        },
    ]
}

/// The names an `Enum` type lists, in its order.
fn names(dtype: &DataType) -> Vec<String> {
    let DataType::Enum(categories, _) = dtype else {
        panic!("{dtype:?} is not an enum");
    };
    categories
        .categories()
        .values_iter()
        .map(str::to_owned)
        .collect()
}

/// The values of an `Enum` column as the names of their variants.
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

/// What a list column holds behind its lists.
fn inner(df: &DataFrame, name: &str) -> Column {
    df.column(name).unwrap().list().unwrap().get_inner().into()
}

#[test]
fn the_type_of_the_column_lists_the_variants() {
    let df = orders().as_slice().to_dataframe().unwrap();

    assert_eq!(
        names(df.column("size").unwrap().dtype()),
        ["Small", "Medium", "Large"]
    );
    // A variant is named by `rename` where it has one, and a raw identifier
    // without its prefix.
    assert_eq!(
        names(df.column("venue").unwrap().dtype()),
        ["XNAS", "XNYS", "Dark", "Auction", "Internal", "match"]
    );

    // The schema says the same without a row.
    let schema = Order::schema().unwrap();
    for (name, dtype) in &schema {
        assert_eq!(dtype, df.column(name).unwrap().dtype(), "column {name}");
    }
    let empty = Order::empty_dataframe().unwrap();
    assert_eq!(empty.height(), 0);
    assert_eq!(empty.schema(), df.schema());
}

#[test]
fn a_row_holds_the_position_of_its_variant() {
    let df = orders().as_slice().to_dataframe().unwrap();

    let size = df.column("size").unwrap();
    assert_eq!(
        strings(size),
        text(&[Some("Large"), Some("Small"), Some("Medium")])
    );
    // A handful of variants are told apart by one byte.
    let codes: Vec<Option<u8>> = size.cat8().unwrap().physical().iter().collect();
    assert_eq!(codes, [Some(2), Some(0), Some(1)]);

    assert_eq!(
        strings(df.column("venue").unwrap()),
        text(&[Some("XNAS"), Some("Auction"), Some("Internal")])
    );
}

#[test]
fn the_column_sorts_in_the_order_of_the_variants() {
    let df = orders().as_slice().to_dataframe().unwrap();
    let sorted = df.sort(["size"], SortMultipleOptions::default()).unwrap();
    assert_eq!(
        strings(sorted.column("size").unwrap()),
        text(&[Some("Small"), Some("Medium"), Some("Large")])
    );
}

#[test]
fn the_wrappers_of_a_value_are_those_of_any_other() {
    let df = orders().as_slice().to_dataframe().unwrap();

    assert_eq!(
        strings(df.column("maybe").unwrap()),
        text(&[Some("Small"), None, Some("Large")])
    );
    assert_eq!(
        strings(df.column("boxed").unwrap()),
        text(&[Some("Medium"), Some("Small"), Some("Large")])
    );

    // Lists hold their values in one `Enum` column behind them.
    let lens = |name: &str| -> Vec<usize> {
        let lists = df.column(name).unwrap().list().unwrap();
        (0..3)
            .map(|row| lists.get_as_series(row).unwrap().len())
            .collect()
    };
    assert_eq!(lens("history"), [2, 0, 1]);
    assert_eq!(
        strings(&inner(&df, "history")),
        text(&[Some("Small"), Some("Medium"), Some("Large")])
    );
    assert_eq!(lens("legs"), [3, 0, 1]);
    assert_eq!(
        strings(&inner(&df, "legs")),
        text(&[Some("Dark"), None, Some("match"), Some("XNYS")])
    );
    assert_eq!(lens("pair"), [2, 2, 2]);
    assert_eq!(
        strings(&inner(&df, "pair")),
        text(&[
            Some("Small"),
            Some("Large"),
            Some("Medium"),
            Some("Medium"),
            Some("Large"),
            Some("Small"),
        ])
    );
}

#[test]
fn every_column_of_a_type_has_one_type() {
    let df = orders().as_slice().to_dataframe().unwrap();

    // Two fields of a row, the values behind a list, and a second frame.
    let size = df.column("size").unwrap().dtype();
    assert_eq!(size, df.column("again").unwrap().dtype());
    assert_eq!(size, df.column("maybe").unwrap().dtype());
    assert_eq!(size, inner(&df, "history").dtype());
    assert_ne!(size, df.column("venue").unwrap().dtype());

    let again = orders().as_slice().to_dataframe().unwrap();
    assert_eq!(df.schema(), again.schema());
    let stacked = df.vstack(&again).unwrap();
    assert_eq!(stacked.height(), 6);
    assert_eq!(
        strings(stacked.column("size").unwrap())[3..],
        text(&[Some("Large"), Some("Small"), Some("Medium")])
    );
}

/// Two types with the same variants. A parameter makes each a type of its
/// own.
#[derive(ToDataFrame, Clone, Copy)]
enum Grade<const SCALE: usize> {
    Low,
    High,
}

#[derive(ToDataFrame)]
struct Review {
    of_five: Grade<5>,
    of_ten: Grade<10>,
}

#[test]
fn types_with_the_same_names_share_a_column_type() {
    let df = [
        Review {
            of_five: Grade::High,
            of_ten: Grade::Low,
        },
        Review {
            of_five: Grade::Low,
            of_ten: Grade::Low,
        },
    ]
    .as_slice()
    .to_dataframe()
    .unwrap();

    // The type of a column is its names, whichever Rust type they are of.
    let of_five = df.column("of_five").unwrap();
    let of_ten = df.column("of_ten").unwrap();
    assert_eq!(of_five.dtype(), of_ten.dtype());
    assert_eq!(strings(of_five), text(&[Some("High"), Some("Low")]));
    assert_eq!(strings(of_ten), text(&[Some("Low"), Some("Low")]));
}

/// Variants named like what generated code might take them for.
#[derive(ToDataFrame, Clone, Copy)]
enum Outcome {
    Some,
    None,
    Ok,
    Err,
    Self_,
}

#[derive(ToDataFrame)]
struct Attempt {
    outcome: Outcome,
    earlier: Option<Outcome>,
}

#[test]
fn a_variant_can_have_any_name() {
    let attempts = [
        Outcome::Some,
        Outcome::None,
        Outcome::Ok,
        Outcome::Err,
        Outcome::Self_,
    ]
    .map(|outcome| Attempt {
        outcome,
        earlier: (outcome.position() % 2 == 0).then_some(outcome),
    });
    let df = attempts.as_slice().to_dataframe().unwrap();

    assert_eq!(
        strings(df.column("outcome").unwrap()),
        text(&[
            Some("Some"),
            Some("None"),
            Some("Ok"),
            Some("Err"),
            Some("Self_"),
        ])
    );
    assert_eq!(
        strings(df.column("earlier").unwrap()),
        text(&[Some("Some"), None, Some("Ok"), None, Some("Self_")])
    );
}

/// The names a type gives its variants, as text.
fn names_of<T: Variants>() -> Vec<String> {
    T::names()
        .into_iter()
        .map(|name| name.to_string())
        .collect()
}

#[test]
fn the_derive_names_the_variants_for_anyone() {
    assert_eq!(names_of::<Size>(), ["Small", "Medium", "Large"]);
    assert_eq!(Size::Large.position(), 2);
    assert_eq!(
        names_of::<Venue>(),
        ["XNAS", "XNYS", "Dark", "Auction", "Internal", "match"]
    );
    assert_eq!(Venue::Auction {}.position(), 3);
    assert_eq!(Venue::r#match.position(), 5);
}

/// An enum that already says what its variants are called takes their names
/// from there: here from `AsRef<str>`.
#[derive(ToDataFrame, Clone, Copy)]
#[df_derive(as_str)]
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

/// And here from `Display`.
#[derive(ToDataFrame, Clone, Copy)]
#[df_derive(as_string)]
enum Interval {
    Minutes5,
    Hours1,
    Days1,
}

impl fmt::Display for Interval {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        out.write_str(match self {
            Self::Minutes5 => "5m",
            Self::Hours1 => "1h",
            Self::Days1 => "1d",
        })
    }
}

#[derive(ToDataFrame)]
struct Bar {
    side: Side,
    interval: Interval,
    last: Option<Interval>,
}

#[test]
fn an_enum_can_name_its_variants_by_its_own_text() {
    let df = [
        Bar {
            side: Side::Sell,
            interval: Interval::Days1,
            last: Some(Interval::Hours1),
        },
        Bar {
            side: Side::Buy,
            interval: Interval::Minutes5,
            last: None,
        },
    ]
    .as_slice()
    .to_dataframe()
    .unwrap();

    let side = df.column("side").unwrap();
    assert_eq!(names(side.dtype()), ["BUY", "SELL"]);
    assert_eq!(strings(side), text(&[Some("SELL"), Some("BUY")]));

    // The order is still that of the variants, not of their text.
    let interval = df.column("interval").unwrap();
    assert_eq!(names(interval.dtype()), ["5m", "1h", "1d"]);
    assert_eq!(strings(interval), text(&[Some("1d"), Some("5m")]));
    assert_eq!(
        strings(df.column("last").unwrap()),
        text(&[Some("1h"), None])
    );

    assert_eq!(names_of::<Side>(), ["BUY", "SELL"]);
    assert_eq!(names_of::<Interval>(), ["5m", "1h", "1d"]);
}

/// A type with three hundred values, which is more than a byte tells apart.
/// It is not an enum: all `Variants` asks for is names and positions, and
/// the names may be computed.
#[derive(Clone, Copy)]
struct Lot(usize);

impl Variants for Lot {
    fn names() -> impl IntoIterator<Item = impl fmt::Display> {
        (0..300).map(|lot| format!("{lot:03}"))
    }

    fn position(&self) -> usize {
        self.0
    }
}

impl Columnar for Lot {
    type Builder = EnumBuilder<Self>;
}

#[derive(ToDataFrame)]
struct Fill {
    lot: Lot,
    spare: Option<Lot>,
}

#[test]
fn many_variants_get_wider_codes() {
    let fills: Vec<Fill> = [0, 255, 256, 299]
        .into_iter()
        .map(|lot| Fill {
            lot: Lot(lot),
            spare: (lot != 255).then_some(Lot(299 - lot)),
        })
        .collect();
    let df = fills.as_slice().to_dataframe().unwrap();

    let lot = df.column("lot").unwrap();
    assert_eq!(names(lot.dtype()).len(), 300);
    let codes: Vec<Option<u16>> = lot.cat16().unwrap().physical().iter().collect();
    assert_eq!(codes, [Some(0), Some(255), Some(256), Some(299)]);
    assert_eq!(
        strings(lot),
        text(&[Some("000"), Some("255"), Some("256"), Some("299")])
    );
    assert_eq!(
        strings(df.column("spare").unwrap()),
        text(&[Some("299"), None, Some("043"), Some("000")])
    );
}

#[test]
fn a_position_past_the_names_is_an_error() {
    // The first position past the names, and one that would be the position
    // of `007` if it were cut to the 32 bits a column keeps of a position.
    let far = usize::try_from((1_u64 << 32) + 7).unwrap_or(usize::MAX);
    for (past, reported) in [(300, "300"), (far, "4294967295 or more")] {
        let fills = |spare: Option<Lot>| -> Vec<Fill> {
            (0..500)
                .map(|row| Fill {
                    lot: Lot(row % 300),
                    spare: if row == 400 { spare } else { Some(Lot(7)) },
                })
                .collect()
        };
        let expected = format!(
            "df-derive: `{}::position` returned {reported}, but `names` gives 300 names",
            std::any::type_name::<Lot>()
        );

        // In a column that has empty slots, and in one that has none.
        let error = fills(Some(Lot(past))).as_slice().to_dataframe();
        assert_eq!(error.unwrap_err().to_string(), expected);
        let rows: Vec<Fill> = fills(Some(Lot(past)))
            .into_iter()
            .map(|fill| Fill {
                lot: fill.spare.unwrap(),
                spare: None,
            })
            .collect();
        let error = rows.as_slice().to_dataframe();
        assert_eq!(error.unwrap_err().to_string(), expected);
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
    type Builder = EnumBuilder<Self>;
}

#[derive(ToDataFrame)]
struct Pair {
    twin: Twin,
}

#[test]
fn names_that_repeat_are_an_error() {
    let error = [Pair { twin: Twin(0) }]
        .as_slice()
        .to_dataframe()
        .unwrap_err()
        .to_string();
    assert!(error.contains("duplicate 'left'"), "{error}");

    // Without a row too: there is no such column type, and asking for it
    // says why.
    assert!(Pair::empty_dataframe().is_err());
    let error = Pair::schema().unwrap_err().to_string();
    assert!(error.contains("duplicate 'left'"), "{error}");
}

/// An enum one of whose variants cannot be written as text.
#[derive(ToDataFrame)]
#[df_derive(as_string)]
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
    // Whichever variant the rows hold: the names are those of the type.
    for mute in [Mute::Spoken, Mute::Silent] {
        let error = [Line { mute }].as_slice().to_dataframe().unwrap_err();
        assert_eq!(error.to_string(), expected);
    }
    assert_eq!(Line::schema().unwrap_err().to_string(), expected);
}

/// A type that names its variants and is text by default. A field can still
/// ask for the `Enum` column by naming the builder.
#[derive(Clone, Copy)]
enum Flag {
    Up,
    Down,
}

impl Variants for Flag {
    fn names() -> impl IntoIterator<Item = impl fmt::Display> {
        [Self::Up, Self::Down].map(|flag| flag.as_ref().to_owned())
    }

    fn position(&self) -> usize {
        *self as usize
    }
}

impl AsRef<str> for Flag {
    fn as_ref(&self) -> &str {
        match self {
            Self::Up => "up",
            Self::Down => "down",
        }
    }
}

#[derive(ToDataFrame)]
struct Signal {
    #[df_derive(as_str)]
    text: Flag,
    #[df_derive(with = "EnumBuilder<Flag>")]
    flag: Flag,
    #[df_derive(with = "EnumBuilder<Flag>")]
    flags: Vec<Option<Flag>>,
}

#[test]
fn a_field_can_name_the_builder() {
    let df = [Signal {
        text: Flag::Down,
        flag: Flag::Down,
        flags: vec![Some(Flag::Up), None],
    }]
    .as_slice()
    .to_dataframe()
    .unwrap();

    assert_eq!(df.column("text").unwrap().dtype(), &DataType::String);
    let flag = df.column("flag").unwrap();
    assert_eq!(names(flag.dtype()), ["up", "down"]);
    assert_eq!(strings(flag), text(&[Some("down")]));
    assert_eq!(strings(&inner(&df, "flags")), text(&[Some("up"), None]));
}

/// An enum with no variants has no values, so only empty slots hold it.
/// Clippy would write a type with no values as `!`. What the derive makes of
/// an enum is what is tested here.
#[derive(ToDataFrame)]
#[allow(clippy::empty_enums)]
enum Never {}

#[derive(ToDataFrame)]
struct Quiet {
    never: Option<Never>,
    none: Vec<Never>,
}

#[test]
fn an_enum_without_variants_is_a_column_of_nulls() {
    let df = [
        Quiet {
            never: None,
            none: vec![],
        },
        Quiet {
            never: None,
            none: vec![],
        },
    ]
    .as_slice()
    .to_dataframe()
    .unwrap();

    let never = df.column("never").unwrap();
    assert_eq!(names(never.dtype()), Vec::<String>::new());
    assert_eq!(never.null_count(), 2);
}

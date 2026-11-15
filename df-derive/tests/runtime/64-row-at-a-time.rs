// Which builders are fed a row at a time, and in what order.
//
// A builder that sets `ROW_AT_A_TIME` is pushed to, one row after the other,
// together with the row's other builders that set it and before the rest.
// Every other builder gets the rows of a block as one run.
//
// The builder of a projected type makes rows and feeds its row type's
// builder: a row at a time while the row has few columns, in runs once it
// has more, and after measuring where the row borrows a list.

use std::cell::RefCell;

use df_derive::ToDataFrame;
use df_derive::dataframe::builders::PrimitiveBuilder;
use df_derive::dataframe::{Bitmap, ColumnBuilder, Columnar, Fields, ToDataFrameVec};
use polars::prelude::*;

/// One call that fed a builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fed {
    /// One slot, through `push`, to the builder with this id.
    Row(u8),
    /// A run of this many slots, through `extend`.
    Run(u8, usize),
    /// This many slots shown ahead, through `measure`.
    Shown(usize),
    /// Room asked for this many slots, through `reserve`.
    Room(usize),
}

thread_local! {
    static CALLS: RefCell<Vec<Fed>> = const { RefCell::new(Vec::new()) };
}

/// Encodes the rows and returns the calls that fed the probes, in order.
fn calls<T: df_derive::dataframe::ToDataFrame>(rows: &[T]) -> (DataFrame, Vec<Fed>) {
    CALLS.with(|calls| calls.borrow_mut().clear());
    let df = rows.to_dataframe().unwrap();
    (df, CALLS.with(|calls| calls.borrow().clone()))
}

/// A value whose builder records how it is fed. `ID` tells the fields of a
/// row apart, and `BY_ROW` is what the builder asks for.
struct Probe<const ID: u8, const BY_ROW: bool>(i64);

struct ProbeBuilder<const ID: u8, const BY_ROW: bool>(PrimitiveBuilder<i64>);

impl<const ID: u8, const BY_ROW: bool> ColumnBuilder<Probe<ID, BY_ROW>>
    for ProbeBuilder<ID, BY_ROW>
{
    const ROW_AT_A_TIME: bool = BY_ROW;

    fn schema(name: PlSmallStr, out: &mut Fields) {
        <PrimitiveBuilder<i64> as ColumnBuilder<i64>>::schema(name, out);
    }

    fn new() -> Self {
        Self(ColumnBuilder::<i64>::new())
    }

    fn reserve(&mut self, additional: usize) {
        ColumnBuilder::<i64>::reserve(&mut self.0, additional);
    }

    fn push(&mut self, value: Option<&Probe<ID, BY_ROW>>) -> PolarsResult<()> {
        CALLS.with(|calls| calls.borrow_mut().push(Fed::Row(ID)));
        self.0.push(value.map(|probe| &probe.0))
    }

    fn extend<'a, I>(&mut self, values: I) -> PolarsResult<()>
    where
        Probe<ID, BY_ROW>: 'a,
        I: Iterator<Item = Option<&'a Probe<ID, BY_ROW>>> + Clone,
    {
        CALLS.with(|calls| {
            calls
                .borrow_mut()
                .push(Fed::Run(ID, values.clone().count()));
        });
        self.0
            .extend(values.map(|value| value.map(|probe| &probe.0)))
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        ColumnBuilder::<i64>::finish(self.0, name, validity, out)
    }
}

impl<const ID: u8, const BY_ROW: bool> Columnar for Probe<ID, BY_ROW> {
    type Builder = ProbeBuilder<ID, BY_ROW>;
}

fn column(df: &DataFrame, name: &str) -> Vec<Option<i64>> {
    let column = df.column(name).unwrap();
    (0..column.len())
        .map(|row| match column.get(row).unwrap() {
            AnyValue::Int64(value) => Some(value),
            AnyValue::Null => None,
            other => panic!("{name}[{row}] is not an integer: {other:?}"),
        })
        .collect()
}

// --- the fields of one row -------------------------------------------------------

#[derive(ToDataFrame)]
struct Mixed {
    a: Probe<1, true>,
    b: Probe<2, false>,
    c: Probe<3, true>,
}

#[test]
fn row_at_a_time_fields_are_fed_together_and_first() {
    let rows: Vec<Mixed> = (0..3)
        .map(|n| Mixed {
            a: Probe(n),
            b: Probe(10 + n),
            c: Probe(20 + n),
        })
        .collect();

    let (df, calls) = calls(&rows);
    assert_eq!(
        calls,
        [
            Fed::Row(1),
            Fed::Row(3),
            Fed::Row(1),
            Fed::Row(3),
            Fed::Row(1),
            Fed::Row(3),
            Fed::Run(2, 3),
        ]
    );
    // The columns are the same whichever way they were filled.
    assert_eq!(column(&df, "a"), [Some(0), Some(1), Some(2)]);
    assert_eq!(column(&df, "b"), [Some(10), Some(11), Some(12)]);
    assert_eq!(column(&df, "c"), [Some(20), Some(21), Some(22)]);
}

// --- options and pointers pass the request on --------------------------------------

#[derive(ToDataFrame)]
struct Wrapped<'a> {
    optional: Option<Probe<1, true>>,
    boxed: Box<Probe<2, true>>,
    borrowed: &'a Probe<3, true>,
    run: Option<Probe<4, false>>,
}

#[test]
fn options_and_pointers_pass_the_request_on() {
    let shared = Probe(7);
    let rows = [
        Wrapped {
            optional: Some(Probe(1)),
            boxed: Box::new(Probe(2)),
            borrowed: &shared,
            run: None,
        },
        Wrapped {
            optional: None,
            boxed: Box::new(Probe(3)),
            borrowed: &shared,
            run: Some(Probe(4)),
        },
    ];

    let (df, calls) = calls(&rows);
    assert_eq!(
        calls,
        [
            Fed::Row(1),
            Fed::Row(2),
            Fed::Row(3),
            Fed::Row(1),
            Fed::Row(2),
            Fed::Row(3),
            Fed::Run(4, 2),
        ]
    );
    assert_eq!(column(&df, "optional"), [Some(1), None]);
    assert_eq!(column(&df, "boxed"), [Some(2), Some(3)]);
    assert_eq!(column(&df, "borrowed"), [Some(7), Some(7)]);
    assert_eq!(column(&df, "run"), [None, Some(4)]);
}

// --- a nested row or tuple asks for it when all of its fields do -------------------

#[derive(ToDataFrame)]
struct AllByRow {
    x: Probe<4, true>,
    y: Probe<5, true>,
}

#[derive(ToDataFrame)]
struct PartlyByRow {
    x: Probe<6, true>,
    y: Probe<7, false>,
}

#[derive(ToDataFrame)]
struct Outer {
    first: Probe<1, true>,
    all: AllByRow,
    pair: (Probe<2, true>, Probe<3, true>),
    partly: PartlyByRow,
    mixed_pair: (Probe<8, true>, Probe<9, false>),
}

#[test]
fn nested_rows_and_tuples_join_in_when_all_their_fields_do() {
    let rows: Vec<Outer> = (0..2)
        .map(|n| Outer {
            first: Probe(n),
            all: AllByRow {
                x: Probe(n),
                y: Probe(n),
            },
            pair: (Probe(n), Probe(n)),
            partly: PartlyByRow {
                x: Probe(n),
                y: Probe(n),
            },
            mixed_pair: (Probe(n), Probe(n)),
        })
        .collect();

    let (df, calls) = calls(&rows);
    assert_eq!(
        calls,
        [
            // One walk over the rows for everything that is fed by row.
            Fed::Row(1),
            Fed::Row(4),
            Fed::Row(5),
            Fed::Row(2),
            Fed::Row(3),
            Fed::Row(1),
            Fed::Row(4),
            Fed::Row(5),
            Fed::Row(2),
            Fed::Row(3),
            // A nested row with a field that takes runs is a run itself, and
            // feeds its own fields the same way.
            Fed::Row(6),
            Fed::Row(6),
            Fed::Run(7, 2),
            // So is a tuple.
            Fed::Row(8),
            Fed::Row(8),
            Fed::Run(9, 2),
        ]
    );
    assert_eq!(column(&df, "all.y"), [Some(0), Some(1)]);
    assert_eq!(column(&df, "partly.y"), [Some(0), Some(1)]);
    assert_eq!(column(&df, "mixed_pair.field_1"), [Some(0), Some(1)]);
}

// --- a projection asks for what its row type asks for ------------------------------

#[derive(ToDataFrame)]
#[df_derive(via = "AllByRow")]
struct Projected(i64);

impl From<&Projected> for AllByRow {
    fn from(projected: &Projected) -> Self {
        Self {
            x: Probe(projected.0),
            y: Probe(-projected.0),
        }
    }
}

#[derive(ToDataFrame)]
struct Holder {
    first: Probe<1, true>,
    projected: Projected,
    last: Probe<2, false>,
}

#[test]
fn a_projection_asks_for_what_its_row_type_asks_for() {
    let rows = [
        Holder {
            first: Probe(1),
            projected: Projected(5),
            last: Probe(3),
        },
        Holder {
            first: Probe(2),
            projected: Projected(6),
            last: Probe(4),
        },
    ];

    let (df, calls) = calls(&rows);
    assert_eq!(
        calls,
        [
            Fed::Row(1),
            Fed::Row(4),
            Fed::Row(5),
            Fed::Row(1),
            Fed::Row(4),
            Fed::Row(5),
            Fed::Run(2, 2),
        ]
    );
    assert_eq!(column(&df, "projected.y"), [Some(-5), Some(-6)]);
}

// --- a projected row of few columns is pushed, one of many gets runs ----------------

#[derive(ToDataFrame)]
struct Narrow {
    a: Probe<1, false>,
    b: Probe<2, false>,
}

#[derive(ToDataFrame)]
#[df_derive(via = "Narrow")]
struct ToNarrow(i64);

impl From<&ToNarrow> for Narrow {
    fn from(value: &ToNarrow) -> Self {
        Self {
            a: Probe(value.0),
            b: Probe(-value.0),
        }
    }
}

#[test]
fn a_projected_row_of_few_columns_is_pushed() {
    let (df, calls) = calls(&[ToNarrow(1), ToNarrow(2), ToNarrow(3)]);
    // Straight from the conversion, with no row kept: its fields take runs,
    // but there are too few of them for a run to be worth a buffer of rows.
    assert_eq!(
        calls,
        [
            Fed::Row(1),
            Fed::Row(2),
            Fed::Row(1),
            Fed::Row(2),
            Fed::Row(1),
            Fed::Row(2),
        ]
    );
    assert_eq!(column(&df, "b"), [Some(-1), Some(-2), Some(-3)]);
}

#[derive(ToDataFrame)]
struct Wide {
    c1: Probe<1, false>,
    c2: Probe<2, false>,
    c3: Probe<3, false>,
    c4: Probe<4, false>,
    c5: Probe<5, false>,
    c6: Probe<6, false>,
    c7: Probe<7, false>,
    c8: Probe<8, false>,
    c9: Probe<9, false>,
}

#[derive(ToDataFrame)]
#[df_derive(via = "Wide")]
struct ToWide(i64);

impl From<&ToWide> for Wide {
    fn from(value: &ToWide) -> Self {
        let n = value.0;
        Self {
            c1: Probe(n),
            c2: Probe(n),
            c3: Probe(n),
            c4: Probe(n),
            c5: Probe(n),
            c6: Probe(n),
            c7: Probe(n),
            c8: Probe(n),
            c9: Probe(n * 10),
        }
    }
}

#[derive(ToDataFrame)]
struct MaybeWide {
    wide: Option<ToWide>,
}

/// One call for each of the nine fields, in order.
fn each(call: impl Fn(u8) -> Fed) -> Vec<Fed> {
    (1..=9).map(call).collect()
}

#[test]
fn a_projected_row_of_many_columns_gets_runs() {
    let (df, fed) = calls(&[ToWide(1), ToWide(2), ToWide(3)]);
    assert_eq!(fed, each(|id| Fed::Run(id, 3)));
    assert_eq!(column(&df, "c9"), [Some(10), Some(20), Some(30)]);

    // An empty slot ends a run, and a run of one row is pushed.
    let rows: Vec<MaybeWide> = [
        Some(1),
        Some(2),
        None,
        Some(4),
        None,
        None,
        Some(7),
        Some(8),
    ]
    .into_iter()
    .map(|wide| MaybeWide {
        wide: wide.map(ToWide),
    })
    .collect();
    let (df, fed) = calls(&rows);
    let expected: Vec<Fed> = [
        each(|id| Fed::Run(id, 2)),
        // The empty slot, the one row between two empty slots, and two more
        // empty slots.
        each(Fed::Row),
        each(Fed::Row),
        each(Fed::Row),
        each(Fed::Row),
        each(|id| Fed::Run(id, 2)),
    ]
    .concat();
    assert_eq!(fed, expected);
    assert_eq!(
        column(&df, "wide.c9"),
        [
            Some(10),
            Some(20),
            None,
            Some(40),
            None,
            None,
            Some(70),
            Some(80)
        ]
    );
}

// --- a list that a projected row borrows is measured --------------------------------

/// A value whose builder records what it is shown ahead of `reserve`, and
/// the room it is then asked for.
struct Counted(i64);

struct CountedBuilder(PrimitiveBuilder<i64>);

impl ColumnBuilder<Counted> for CountedBuilder {
    fn schema(name: PlSmallStr, out: &mut Fields) {
        <PrimitiveBuilder<i64> as ColumnBuilder<i64>>::schema(name, out);
    }

    fn new() -> Self {
        Self(ColumnBuilder::<i64>::new())
    }

    fn measure<'a, I>(&mut self, values: I)
    where
        I: Iterator<Item = Option<&'a Counted>> + Clone,
    {
        CALLS.with(|calls| calls.borrow_mut().push(Fed::Shown(values.count())));
    }

    fn reserve(&mut self, additional: usize) {
        CALLS.with(|calls| calls.borrow_mut().push(Fed::Room(additional)));
        ColumnBuilder::<i64>::reserve(&mut self.0, additional);
    }

    fn push(&mut self, value: Option<&Counted>) -> PolarsResult<()> {
        self.0.push(value.map(|counted| &counted.0))
    }

    fn finish(
        self,
        name: PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()> {
        ColumnBuilder::<i64>::finish(self.0, name, validity, out)
    }
}

impl Columnar for Counted {
    type Builder = CountedBuilder;
}

#[derive(ToDataFrame)]
struct Borrowing<'a> {
    items: &'a [Counted],
}

#[derive(ToDataFrame)]
struct Owning {
    items: Vec<Counted>,
}

#[derive(ToDataFrame)]
#[df_derive(via = "Borrowing<'_>")]
struct Lent(Vec<Counted>);

impl<'a> From<&'a Lent> for Borrowing<'a> {
    fn from(lent: &'a Lent) -> Self {
        Self { items: &lent.0 }
    }
}

#[derive(ToDataFrame)]
#[df_derive(via = "Owning")]
struct Copied(Vec<i64>);

impl From<&Copied> for Owning {
    fn from(copied: &Copied) -> Self {
        Self {
            items: copied.0.iter().copied().map(Counted).collect(),
        }
    }
}

#[test]
fn a_list_that_a_projected_row_borrows_is_measured() {
    let lists = [vec![1, 2], vec![], vec![3, 4, 5, 6, 7]];

    // A row that owns nothing costs nothing to make, so each is made a first
    // time to count the elements of its list, and the room for all seven is
    // made once.
    let lent: Vec<Lent> = lists
        .iter()
        .map(|list| Lent(list.iter().copied().map(Counted).collect()))
        .collect();
    let (_, fed) = calls(&lent);
    assert_eq!(
        fed,
        [Fed::Shown(2), Fed::Shown(0), Fed::Shown(5), Fed::Room(7)]
    );

    // A row that owns its list was allocated for. It is made once, and its
    // list grows as it fills.
    let copied: Vec<Copied> = lists.iter().cloned().map(Copied).collect();
    let (_, fed) = calls(&copied);
    assert_eq!(fed, [Fed::Room(0)]);
}

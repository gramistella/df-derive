// Polars keeps the total length of a string column's values while the column
// is built, and reads it later: to size a concatenation, to estimate memory.
// Setting a validity makes it forget the total, so a column with no empty
// slot is finished without one and the total is still known.

use df_derive::ToDataFrame;
use df_derive::dataframe::ToDataFrameVec;

#[derive(ToDataFrame)]
struct Entry {
    name: String,
    #[df_derive(as_binary)]
    raw: Vec<u8>,
    #[df_derive(as_string)]
    code: u32,
    note: Option<String>,
}

fn entries() -> Vec<Entry> {
    vec![
        Entry {
            name: "a name longer than twelve bytes".to_owned(),
            raw: vec![1, 2, 3],
            code: 7,
            note: None,
        },
        Entry {
            name: "short".to_owned(),
            raw: Vec::new(),
            code: 1234,
            note: Some("kept".to_owned()),
        },
    ]
}

#[test]
fn a_column_with_no_empty_slot_keeps_its_byte_total() {
    let df = entries().to_dataframe().unwrap();

    let name = df.column("name").unwrap().as_materialized_series();
    let name = name.str().unwrap().downcast_iter().next().unwrap();
    assert_eq!(name.try_total_bytes_len(), Some(31 + 5));

    let raw = df.column("raw").unwrap().as_materialized_series();
    let raw = raw.binary().unwrap().downcast_iter().next().unwrap();
    assert_eq!(raw.try_total_bytes_len(), Some(3));

    let code = df.column("code").unwrap().as_materialized_series();
    let code = code.str().unwrap().downcast_iter().next().unwrap();
    assert_eq!(code.try_total_bytes_len(), Some(1 + 4));
}

#[test]
fn a_column_with_an_empty_slot_still_totals_right() {
    let df = entries().to_dataframe().unwrap();

    let note = df.column("note").unwrap().as_materialized_series();
    assert_eq!(note.null_count(), 1);
    let note = note.str().unwrap().downcast_iter().next().unwrap();
    assert_eq!(note.total_bytes_len(), 4);
}

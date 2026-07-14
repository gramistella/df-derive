use df_derive::ToDataFrame;
use df_derive::dataframe::{Columnar, ToDataFrameVec};

#[derive(ToDataFrame)]
struct MisreportedHintRow {
    flag: Option<bool>,
    text: Option<String>,
    #[df_derive(as_binary)]
    bytes: Option<Vec<u8>>,
    #[df_derive(as_str)]
    as_str: Option<String>,
    #[df_derive(as_string)]
    as_string: Option<i32>,
    nested: Vec<Vec<Option<i32>>>,
}

struct MisreportedExact<'a, T> {
    inner: std::slice::Iter<'a, T>,
    claimed_len: usize,
}

impl<'a, T> MisreportedExact<'a, T> {
    fn new(rows: &'a [T], claimed_len: usize) -> Self {
        Self {
            inner: rows.iter(),
            claimed_len,
        }
    }
}

impl<'a, T> Iterator for MisreportedExact<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.claimed_len, Some(self.claimed_len))
    }
}

#[test]
fn scalar_nullable_bitmaps_tolerate_incorrect_exact_size_hints() {
    let rows = [
        MisreportedHintRow {
            flag: Some(true),
            text: Some("alpha".into()),
            bytes: Some(vec![1, 2, 3]),
            as_str: Some("first".into()),
            as_string: Some(10),
            nested: vec![vec![Some(1), None], vec![]],
        },
        MisreportedHintRow {
            flag: None,
            text: None,
            bytes: None,
            as_str: None,
            as_string: None,
            nested: vec![],
        },
        MisreportedHintRow {
            flag: Some(false),
            text: Some("omega".into()),
            bytes: Some(vec![9; 24]),
            as_str: Some("last".into()),
            as_string: Some(-7),
            nested: vec![vec![None], vec![Some(9), Some(10)]],
        },
    ];
    let expected = rows.as_slice().to_dataframe().unwrap();

    for claimed_len in [0, 1, rows.len(), rows.len() + 5] {
        let actual = MisreportedHintRow::encode(MisreportedExact::new(&rows, claimed_len))
            .unwrap_or_else(|error| panic!("claimed_len={claimed_len}: {error}"));
        assert!(
            actual.equals_missing(&expected),
            "claimed_len={claimed_len}\nactual={actual:?}\nexpected={expected:?}",
        );
    }
}

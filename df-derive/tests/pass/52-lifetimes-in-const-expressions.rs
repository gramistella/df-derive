//! A field's type may hold a const expression, and a name after `'` in one
//! need not be a lifetime of the row: it can be a loop label, or the lifetime
//! parameter of an item. The derive writes field types as they are.
use df_derive::ToDataFrame;
use df_derive::dataframe::ToDataFrameVec;

#[derive(ToDataFrame)]
struct Sized<'a> {
    name: &'a str,
    counted: [u8; {
        'count: loop {
            break 'count 2;
        }
    }],
    picked: [u8; {
        const fn pick<'x>(value: &'x usize) -> &'x usize {
            value
        }
        *pick(&3)
    }],
}

#[derive(ToDataFrame)]
struct Generic<'a, T> {
    value: &'a T,
    counted: [T; {
        'count: loop {
            break 'count 2;
        }
    }],
}

fn main() {
    let name = String::from("row");
    let rows = [Sized {
        name: &name,
        counted: [1, 2],
        picked: [1, 2, 3],
    }];
    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(df.shape(), (1, 3));

    let value = 5_i64;
    let rows = [Generic {
        value: &value,
        counted: [6, 7],
    }];
    let df = rows.as_slice().to_dataframe().unwrap();
    assert_eq!(df.shape(), (1, 2));
}

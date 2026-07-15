use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
pub struct SixteenOneElementTupleFields {
    t00: (i64,),
    t01: (i64,),
    t02: (i64,),
    t03: (i64,),
    t04: (i64,),
    t05: (i64,),
    t06: (i64,),
    t07: (i64,),
    t08: (i64,),
    t09: (i64,),
    t10: (i64,),
    t11: (i64,),
    t12: (i64,),
    t13: (i64,),
    t14: (i64,),
    t15: (i64,),
}

#[derive(ToDataFrame)]
#[allow(clippy::type_complexity)]
pub struct SeventeenScalarTupleTerminals {
    values: (
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
    ),
}

#[derive(ToDataFrame)]
pub struct FlatThirtyTwoScalarFields {
    f00: i64,
    f01: i64,
    f02: i64,
    f03: i64,
    f04: i64,
    f05: i64,
    f06: i64,
    f07: i64,
    f08: i64,
    f09: i64,
    f10: i64,
    f11: i64,
    f12: i64,
    f13: i64,
    f14: i64,
    f15: i64,
    f16: i64,
    f17: i64,
    f18: i64,
    f19: i64,
    f20: i64,
    f21: i64,
    f22: i64,
    f23: i64,
    f24: i64,
    f25: i64,
    f26: i64,
    f27: i64,
    f28: i64,
    f29: i64,
    f30: i64,
    f31: i64,
}

fn row_value(row: usize, width: i64, offset: i64) -> i64 {
    i64::try_from(row).unwrap() * width + offset
}

fn flat_row_value(row: usize, offset: i64) -> i64 {
    i64::try_from(row).unwrap() + offset
}

pub fn make_sixteen_one_element_tuple_fields(
    row_count: usize,
) -> Vec<SixteenOneElementTupleFields> {
    (0..row_count)
        .map(|row| SixteenOneElementTupleFields {
            t00: (row_value(row, 16, 0),),
            t01: (row_value(row, 16, 1),),
            t02: (row_value(row, 16, 2),),
            t03: (row_value(row, 16, 3),),
            t04: (row_value(row, 16, 4),),
            t05: (row_value(row, 16, 5),),
            t06: (row_value(row, 16, 6),),
            t07: (row_value(row, 16, 7),),
            t08: (row_value(row, 16, 8),),
            t09: (row_value(row, 16, 9),),
            t10: (row_value(row, 16, 10),),
            t11: (row_value(row, 16, 11),),
            t12: (row_value(row, 16, 12),),
            t13: (row_value(row, 16, 13),),
            t14: (row_value(row, 16, 14),),
            t15: (row_value(row, 16, 15),),
        })
        .collect()
}

pub fn make_seventeen_scalar_tuple_terminals(
    row_count: usize,
) -> Vec<SeventeenScalarTupleTerminals> {
    (0..row_count)
        .map(|row| SeventeenScalarTupleTerminals {
            values: (
                row_value(row, 17, 0),
                row_value(row, 17, 1),
                row_value(row, 17, 2),
                row_value(row, 17, 3),
                row_value(row, 17, 4),
                row_value(row, 17, 5),
                row_value(row, 17, 6),
                row_value(row, 17, 7),
                row_value(row, 17, 8),
                row_value(row, 17, 9),
                row_value(row, 17, 10),
                row_value(row, 17, 11),
                row_value(row, 17, 12),
                row_value(row, 17, 13),
                row_value(row, 17, 14),
                row_value(row, 17, 15),
                row_value(row, 17, 16),
            ),
        })
        .collect()
}

#[allow(clippy::too_many_lines)]
pub fn make_flat_thirty_two_scalar_fields(row_count: usize) -> Vec<FlatThirtyTwoScalarFields> {
    (0..row_count)
        .map(|row| FlatThirtyTwoScalarFields {
            f00: flat_row_value(row, 0),
            f01: flat_row_value(row, 1),
            f02: flat_row_value(row, 2),
            f03: flat_row_value(row, 3),
            f04: flat_row_value(row, 4),
            f05: flat_row_value(row, 5),
            f06: flat_row_value(row, 6),
            f07: flat_row_value(row, 7),
            f08: flat_row_value(row, 8),
            f09: flat_row_value(row, 9),
            f10: flat_row_value(row, 10),
            f11: flat_row_value(row, 11),
            f12: flat_row_value(row, 12),
            f13: flat_row_value(row, 13),
            f14: flat_row_value(row, 14),
            f15: flat_row_value(row, 15),
            f16: flat_row_value(row, 16),
            f17: flat_row_value(row, 17),
            f18: flat_row_value(row, 18),
            f19: flat_row_value(row, 19),
            f20: flat_row_value(row, 20),
            f21: flat_row_value(row, 21),
            f22: flat_row_value(row, 22),
            f23: flat_row_value(row, 23),
            f24: flat_row_value(row, 24),
            f25: flat_row_value(row, 25),
            f26: flat_row_value(row, 26),
            f27: flat_row_value(row, 27),
            f28: flat_row_value(row, 28),
            f29: flat_row_value(row, 29),
            f30: flat_row_value(row, 30),
            f31: flat_row_value(row, 31),
        })
        .collect()
}

// A float column holds exactly the bits it was given: the payload of a NaN,
// the sign of a zero, a number too small to be normal. A float travels as its
// bits while its column is built, and these tests state what that keeps. An
// empty slot is a null over a zero, whatever the slots around it hold.

use df_derive::ToDataFrame;
use df_derive::dataframe::ToDataFrameVec;
use polars::prelude::*;

#[derive(ToDataFrame)]
struct Floats {
    wide: f64,
    narrow: f32,
    maybe: Option<f64>,
    maybe_narrow: Option<f32>,
    list: Vec<Option<f64>>,
}

/// Bit patterns that arithmetic would not keep.
const WIDE: [u64; 7] = [
    0x7FF8_0000_0000_1234, // a quiet NaN with a payload
    0xFFF0_0000_0000_0001, // a signalling NaN, negative
    0x8000_0000_0000_0000, // negative zero
    0x0000_0000_0000_0001, // the smallest subnormal
    0x7FF0_0000_0000_0000, // infinity
    0xFFEF_FFFF_FFFF_FFFF, // the most negative finite number
    0x3FF8_0000_0000_0000, // 1.5
];
const NARROW: [u32; 7] = [
    0x7FC0_1234,
    0xFF80_0001,
    0x8000_0000,
    0x0000_0001,
    0x7F80_0000,
    0xFF7F_FFFF,
    0x3FC0_0000,
];

fn rows() -> Vec<Floats> {
    (0..WIDE.len())
        .map(|index| {
            let wide = f64::from_bits(WIDE[index]);
            let narrow = f32::from_bits(NARROW[index]);
            Floats {
                wide,
                narrow,
                // Every other slot is empty.
                maybe: (index % 2 == 0).then_some(wide),
                maybe_narrow: (index % 2 == 1).then_some(narrow),
                list: vec![Some(wide), None, Some(-wide)],
            }
        })
        .collect()
}

/// The buffer behind a column of `f64`, placeholders included, as bits.
fn wide_buffer(column: &Column) -> Vec<u64> {
    let floats = column.f64().unwrap();
    let chunk = floats.downcast_iter().next().unwrap();
    chunk.values().iter().map(|value| value.to_bits()).collect()
}

fn narrow_buffer(column: &Column) -> Vec<u32> {
    let floats = column.f32().unwrap();
    let chunk = floats.downcast_iter().next().unwrap();
    chunk.values().iter().map(|value| value.to_bits()).collect()
}

#[test]
fn a_float_keeps_its_bits() {
    let df = rows().as_slice().to_dataframe().unwrap();

    assert_eq!(df.column("wide").unwrap().dtype(), &DataType::Float64);
    assert_eq!(df.column("narrow").unwrap().dtype(), &DataType::Float32);
    assert_eq!(wide_buffer(df.column("wide").unwrap()), WIDE);
    assert_eq!(narrow_buffer(df.column("narrow").unwrap()), NARROW);
}

#[test]
fn an_empty_slot_is_a_null_over_a_zero() {
    let df = rows().as_slice().to_dataframe().unwrap();

    let maybe = df.column("maybe").unwrap();
    let expected: Vec<u64> = (0..WIDE.len())
        .map(|index| if index % 2 == 0 { WIDE[index] } else { 0 })
        .collect();
    assert_eq!(wide_buffer(maybe), expected);
    let nulls: Vec<bool> = maybe
        .f64()
        .unwrap()
        .iter()
        .map(|slot| slot.is_none())
        .collect();
    assert_eq!(nulls, [false, true, false, true, false, true, false]);

    let maybe_narrow = df.column("maybe_narrow").unwrap();
    let expected: Vec<u32> = (0..NARROW.len())
        .map(|index| if index % 2 == 1 { NARROW[index] } else { 0 })
        .collect();
    assert_eq!(narrow_buffer(maybe_narrow), expected);
    assert_eq!(maybe_narrow.null_count(), 4);
}

#[test]
fn the_floats_of_a_list_keep_their_bits() {
    let df = rows().as_slice().to_dataframe().unwrap();

    let lists = df.column("list").unwrap().list().unwrap();
    let items = lists.get_inner();
    let expected: Vec<u64> = WIDE
        .iter()
        .flat_map(|&bits| [bits, 0, (-f64::from_bits(bits)).to_bits()])
        .collect();
    assert_eq!(wide_buffer(&items.clone().into()), expected);
    assert_eq!(items.null_count(), WIDE.len());
}

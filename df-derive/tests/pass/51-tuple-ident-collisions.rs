#![allow(non_upper_case_globals)]

use df_derive::ToDataFrame;

#[derive(ToDataFrame)]
#[allow(clippy::type_complexity)]
struct TupleIdentCollisions<
    const __df_derive_t: usize,
    const __df_derive_t_count_0: usize,
    const __df_derive_t_item_0: usize,
    const __df_derive_t_input_1: usize,
    const __df_derive_t_value_1: usize,
    const __df_derive_t_off_0_0: usize,
    const __df_derive_t_off_buf_0_0: usize,
    const __df_derive_t_valmb_0_0: usize,
    const __df_derive_t_valbm_0_0: usize,
    const __df_derive_t_bind_0_0: usize,
    const __df_derive_t_inner_0: usize,
    const __df_derive_t_rech_0: usize,
    const __df_derive_t_chunk_0: usize,
    const __df_derive_t_prefix_arr_0_0: usize,
    const __df_derive_tuple_series: usize,
    const __df_derive_tuple_named: usize,
    const __df_derive_t_logical_dtype: usize,
    const __df_derive_slot_0: usize,
> {
    values: Option<Vec<Option<(Option<(i32, String)>, bool)>>>,
}

#[derive(ToDataFrame)]
struct PrimitiveIdentCollisions<
    const __df_derive_ri_0: usize,
    const __df_derive_leaf_count_1: usize,
    const __df_derive_leaf_segments_1: usize,
    const __df_derive_leaf_segment_1: usize,
    const __df_derive_prepared_len_1: usize,
    const __df_derive_prepared_validity_1: usize,
> {
    flag: Option<bool>,
    values: Vec<Option<bool>>,
    numbers: Vec<i32>,
    decimals: Vec<Option<rust_decimal::Decimal>>,
    nested: Vec<Vec<i32>>,
}

#[derive(ToDataFrame)]
struct NestedPayload {
    value: i32,
}

#[derive(ToDataFrame)]
struct NestedSlotIdentCollision<const __df_derive_slot_0: usize> {
    nested: NestedPayload,
}

fn main() {}

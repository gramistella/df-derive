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
> {
    values: Option<Vec<Option<(Option<(i32, String)>, bool)>>>,
}

fn main() {}

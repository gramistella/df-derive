//! Per-field encoder construction.
//!
//! `Vec` wrappers are already normalized into [`crate::ir::VecLayers`] before
//! this phase. Polars folds consecutive `Option` layers into a single validity
//! bit, so the encoder collapses multi-Option access chains before emitting
//! the option leaf arm.

mod access_chain;
mod ctx;
mod emit;
pub(in crate::codegen) mod idents;
mod leaf;
mod leaf_kind;
mod nested_columns;
mod nested_leaf;
mod option;
mod shape_walk;
mod stringy;
mod tuple;
mod vec;

pub use ctx::{BaseCtx, Encoder, LeafCtx, build_encoder};
pub use nested_leaf::{NestedLeafCtx, build_nested_encoder};
pub use stringy::struct_type_tokens;
pub use tuple::{
    build_field_emit as build_tuple_field_emit, build_field_entries as build_tuple_field_entries,
};

pub(super) use access_chain::{
    access_chain_to_option_ref, access_chain_to_ref, collapse_options_to_ref, idx_size_len_expr,
    list_offset_i64_expr,
};
pub(super) use ctx::build_encoder_with_option_receiver;
pub(super) use stringy::{StringyExprKind, stringy_value_expr};

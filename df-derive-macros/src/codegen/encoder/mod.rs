//! Per-column encoder construction.
//!
//! `Vec` wrappers are already normalized into [`crate::ir::VecLayers`] before
//! this phase. Polars folds consecutive `Option` layers into a single validity
//! bit, so the encoder collapses multi-Option access chains before emitting
//! the option leaf arm.

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
mod tuple_group;
mod vec;
mod wrapper_access;

pub use ctx::{BaseCtx, EncodeLifecycle, Encoder, LeafCardinality, LeafCtx, build_encoder};
pub use nested_leaf::{NestedLeafCtx, build_nested_encoder};
pub use stringy::struct_type_tokens;
pub(in crate::codegen) use tuple_group::{
    REPLAY_STATIC_TUPLE_MIN_TERMINALS, TupleFieldEmitParams, build_tuple_field_emit,
    replayable_tuple_terminal_count,
};

pub(super) use ctx::build_encoder_with_option_receiver;
pub(super) use stringy::{StringyExprKind, stringy_value_expr};
pub(super) use wrapper_access::{
    access_chain_to_option_ref, access_chain_to_ref, collapse_options_to_ref, idx_size_len_expr,
    list_offset_i64_expr,
};

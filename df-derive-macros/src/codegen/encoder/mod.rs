//! Per-column encoder construction.
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
mod projected;
mod shape_walk;
mod stringy;
mod vec;

pub use ctx::{BaseCtx, Encoder, LeafCtx, build_encoder};
pub use nested_leaf::{NestedLeafCtx, build_nested_encoder};
pub(in crate::codegen) use projected::{build_projected_vec_nested, build_projected_vec_primitive};
pub use stringy::struct_type_tokens;

pub(super) use access_chain::{
    access_chain_to_option_ref, access_chain_to_ref, collapse_options_to_ref, idx_size_len_expr,
    list_offset_i64_expr,
};
pub(super) use ctx::build_encoder_with_option_receiver;
pub(super) use stringy::{StringyExprKind, stringy_value_expr};

//! Centralized identifier generation for the encoder IR.
//!
//! Every identifier the macro injects into generated code is built from
//! the `__df_derive_` prefix plus a structured suffix. Keep generated
//! identifiers routed through this module tree so new emitters do not
//! accidentally collide with existing locals.

mod layers;
mod nested;
mod primitive;
mod support;
mod tuple;

pub(in crate::codegen) use layers::*;
pub(in crate::codegen) use nested::*;
pub(in crate::codegen) use primitive::*;
pub(in crate::codegen) use support::*;
pub(in crate::codegen) use tuple::*;

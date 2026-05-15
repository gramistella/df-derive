//! Lowering helpers between type analysis and codegen IR.

mod diagnostics;
mod field;
mod tuple;
mod validation;
mod wrappers;

pub use field::lower_field;

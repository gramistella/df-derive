//! Lowering helpers between type analysis and codegen IR.

mod binary;
mod columns;
mod errors;
mod field;
mod leaf;
mod tuple;
mod validation;
mod wrappers;

pub use columns::project_fields_to_columns;
pub use field::lower_field;

//! Lowering helpers between type analysis and codegen IR.

mod binary;
mod errors;
mod field;
mod leaf;
mod projection;
mod tuple;
mod validation;
mod wrappers;

pub use field::lower_field;
pub use projection::project_fields_to_columns;

//! Lowering helpers between type analysis and codegen IR.

mod binary;
mod errors;
mod field;
mod leaf;
mod planning;
mod tuple;
mod validation;
mod wrappers;

pub use field::lower_field;
pub use planning::plan_fields;

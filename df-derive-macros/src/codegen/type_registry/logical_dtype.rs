use proc_macro2::TokenStream;
use quote::quote;

use crate::ir::{DateTimeUnit, NumericKind, PrimitiveLeaf, WrapperShape};

use crate::codegen::external_paths::ExternalPaths;

use super::numeric::numeric_info_for;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::codegen) enum LogicalPrimitive {
    Numeric(NumericKind),
    String,
    Bool,
    Binary,
    DateTime(DateTimeUnit),
    NaiveDateTime(DateTimeUnit),
    NaiveDate,
    NaiveTime,
    Duration(DateTimeUnit),
    Decimal { precision: u8, scale: u8 },
}

pub(in crate::codegen) fn time_unit_tokens(
    unit: DateTimeUnit,
    paths: &ExternalPaths,
) -> TokenStream {
    let pp = paths.prelude();
    match unit {
        DateTimeUnit::Milliseconds => quote! { #pp::TimeUnit::Milliseconds },
        DateTimeUnit::Microseconds => quote! { #pp::TimeUnit::Microseconds },
        DateTimeUnit::Nanoseconds => quote! { #pp::TimeUnit::Nanoseconds },
    }
}

impl LogicalPrimitive {
    pub(in crate::codegen) fn dtype(self, paths: &ExternalPaths) -> TokenStream {
        let pp = paths.prelude();
        let dt = quote! { #pp::DataType };
        match self {
            Self::Numeric(kind) => numeric_info_for(kind, paths).dtype,
            Self::String => quote! { #dt::String },
            Self::Bool => quote! { #dt::Boolean },
            Self::Binary => quote! { #dt::Binary },
            Self::DateTime(unit) | Self::NaiveDateTime(unit) => {
                let unit = time_unit_tokens(unit, paths);
                quote! { #dt::Datetime(#unit, ::std::option::Option::None) }
            }
            Self::NaiveDate => quote! { #dt::Date },
            Self::NaiveTime => quote! { #dt::Time },
            Self::Duration(unit) => {
                let unit = time_unit_tokens(unit, paths);
                quote! { #dt::Duration(#unit) }
            }
            Self::Decimal { precision, scale } => {
                let p = precision as usize;
                let s = scale as usize;
                quote! { #dt::Decimal(#p, #s) }
            }
        }
    }
}

impl PrimitiveLeaf<'_> {
    pub(in crate::codegen) const fn logical(&self) -> LogicalPrimitive {
        match *self {
            Self::Numeric(kind) => LogicalPrimitive::Numeric(kind),
            Self::String | Self::AsString | Self::AsStr(_) => LogicalPrimitive::String,
            Self::Bool => LogicalPrimitive::Bool,
            Self::Binary => LogicalPrimitive::Binary,
            Self::DateTime(unit) => LogicalPrimitive::DateTime(unit),
            Self::NaiveDateTime(unit) => LogicalPrimitive::NaiveDateTime(unit),
            Self::NaiveDate => LogicalPrimitive::NaiveDate,
            Self::NaiveTime => LogicalPrimitive::NaiveTime,
            Self::Duration { unit, source: _ } => LogicalPrimitive::Duration(unit),
            Self::Decimal { precision, scale } => LogicalPrimitive::Decimal { precision, scale },
        }
    }

    pub(in crate::codegen) fn dtype(&self, paths: &ExternalPaths) -> TokenStream {
        self.logical().dtype(paths)
    }
}

pub(in crate::codegen) fn full_dtype(
    leaf: PrimitiveLeaf<'_>,
    wrapper: &WrapperShape,
    paths: &ExternalPaths,
) -> TokenStream {
    let pp = paths.prelude();
    let elem_dtype = leaf.dtype(paths);
    crate::codegen::external_paths::wrap_list_layers_compile_time(
        pp,
        elem_dtype,
        wrapper.vec_depth(),
    )
}

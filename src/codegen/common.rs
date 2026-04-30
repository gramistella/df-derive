use super::strategy::{
    self, ColumnarBuilderFinisher as _, ColumnarBulkEmitter as _, VecAnyvaluesBulkEmitter as _,
    VecAnyvaluesFinisher as _,
};
use crate::ir::{BaseType, PrimitiveTransform, StructIR};
use proc_macro2::TokenStream;
use quote::quote;

/// Generate the access/transformation expression for a primitive value based on an abstract transform.
pub fn generate_primitive_access_expr(
    var: &TokenStream,
    transform: Option<&PrimitiveTransform>,
) -> TokenStream {
    crate::codegen::type_registry::map_primitive_expr(var, transform)
}

// generate_fast_path_primitives removed; logic is now handled in strategies

// Unified collection iteration preparation helpers

pub fn prepare_vec_anyvalues_parts(
    ir: &StructIR,
    it_ident: &syn::Ident,
) -> (Vec<TokenStream>, Vec<TokenStream>, Vec<TokenStream>) {
    let strategies = strategy::build_strategies(ir);
    let mut decls: Vec<TokenStream> = Vec::new();
    let mut pushes: Vec<TokenStream> = Vec::new();
    let mut finishers: Vec<TokenStream> = Vec::new();
    for (idx, s) in strategies.iter().enumerate() {
        if let Some(bulk) = s.gen_bulk_vec_anyvalues_emit(idx) {
            finishers.extend(bulk);
        } else {
            decls.extend(super::strategy::ColumnPopulator::gen_populator_inits(
                s, idx,
            ));
            pushes.push(super::strategy::ColumnPopulator::gen_populator_push(
                s, it_ident, idx,
            ));
            finishers.push(s.gen_vec_values_finishers(idx));
        }
    }
    (decls, pushes, finishers)
}

pub fn prepare_columnar_parts(
    ir: &StructIR,
    it_ident: &syn::Ident,
) -> (Vec<TokenStream>, Vec<TokenStream>, Vec<TokenStream>) {
    let strategies = strategy::build_strategies(ir);
    let mut decls: Vec<TokenStream> = Vec::new();
    let mut pushes: Vec<TokenStream> = Vec::new();
    let mut builders: Vec<TokenStream> = Vec::new();
    for (idx, s) in strategies.iter().enumerate() {
        if let Some(bulk) = s.gen_bulk_columnar_emit(idx) {
            builders.extend(bulk);
        } else {
            decls.extend(super::strategy::ColumnPopulator::gen_populator_inits(
                s, idx,
            ));
            pushes.push(super::strategy::ColumnPopulator::gen_populator_push(
                s, it_ident, idx,
            ));
            builders.extend(s.gen_columnar_builders(idx));
        }
    }
    (decls, pushes, builders)
}

// --- Shared helpers and wrapper-aware builders ---

/// Build an efficient inner Series from a Vec or Vec<Option<T>> for primitive types.
/// Applies mapping only when transforms require it (stringify/casts), otherwise zero-copy.
pub fn generate_inner_series_from_vec(
    vec_access: &TokenStream,
    base_type: &BaseType,
    transform: Option<&PrimitiveTransform>,
) -> TokenStream {
    // Borrowing path for `as_str` on `Vec<T>` shapes: build a `Vec<&str>`
    // through `AsRef<str>` via UFCS so the polars Series sees the same
    // `&[&str]` slice the bare-`String` borrowing path uses. No per-element
    // allocation.
    if matches!(transform, Some(PrimitiveTransform::AsStr)) {
        let ty_path = match base_type {
            BaseType::Struct(ident, args) => super::strategy::build_type_path(ident, args.as_ref()),
            BaseType::Generic(ident) => quote! { #ident },
            // `BaseType::String` falls through here too: `String: AsRef<str>`
            // makes UFCS valid. Non-string primitive bases are caught by the
            // per-field `AsRef<str>` assert in helpers.rs; this fallback
            // keeps codegen syntactically valid up to that assert firing.
            _ => quote! { ::std::string::String },
        };
        // `#vec_access` must be a place expression (e.g. `self.field` or a
        // named binding) — never a temporary that drops at the next `;`.
        // The on_vec callers all pass non-temp expressions; the deep-nesting
        // path in `gen_primitive_vec_inner_series` binds the inner clone to
        // a named local before recursing, so that recursion enters this
        // function with a name too.
        return quote! {{
            let __df_derive_conv: ::std::vec::Vec<&str> = (#vec_access)
                .iter()
                .map(<#ty_path as ::core::convert::AsRef<str>>::as_ref)
                .collect();
            let inner_series = polars::prelude::Series::new("".into(), &__df_derive_conv);
            inner_series
        }};
    }
    let mapping = crate::codegen::type_registry::compute_mapping(base_type, transform, &[]);
    let dtype = mapping.element_dtype;
    let do_cast = crate::codegen::type_registry::needs_cast(transform);
    let needs_conv =
        transform.is_some_and(|t| matches!(*t, PrimitiveTransform::ToString)) || do_cast;
    if needs_conv {
        let elem_ident = syn::Ident::new("__df_derive_e", proc_macro2::Span::call_site());
        let var_ts = quote! { #elem_ident };
        let mapped = generate_primitive_access_expr(&var_ts, transform);
        quote! {{
            let __df_derive_conv: ::std::vec::Vec<_> = (#vec_access).iter().map(|#elem_ident| { #mapped }).collect();
            let mut inner_series = polars::prelude::Series::new("".into(), &__df_derive_conv);
            if #do_cast { inner_series = inner_series.cast(&#dtype)?; }
            inner_series
        }}
    } else {
        quote! {{
            let mut inner_series = polars::prelude::Series::new("".into(), &(#vec_access));
            if #do_cast { inner_series = inner_series.cast(&#dtype)?; }
            inner_series
        }}
    }
}

// removed unused generate_anyvalue_from_scalar

// moved to wrapping.rs and used via strategy

// moved to wrapping.rs and used via strategy

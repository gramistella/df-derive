use super::strategy;
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
    config: &super::MacroConfig,
    it_ident: &syn::Ident,
) -> (Vec<TokenStream>, Vec<TokenStream>, Vec<TokenStream>) {
    let strategies = strategy::build_strategies(ir, config);
    let mut decls: Vec<TokenStream> = Vec::new();
    let mut pushes: Vec<TokenStream> = Vec::new();
    let mut finishers: Vec<TokenStream> = Vec::new();
    for (idx, s) in strategies.iter().enumerate() {
        if let Some(bulk) = s.gen_bulk_vec_anyvalues_emit(idx) {
            finishers.extend(bulk);
        } else {
            decls.extend(s.gen_populator_inits(idx));
            pushes.push(s.gen_populator_push(it_ident, idx));
            finishers.push(s.gen_vec_values_finishers(idx));
        }
    }
    (decls, pushes, finishers)
}

pub fn prepare_columnar_parts(
    ir: &StructIR,
    config: &super::MacroConfig,
    it_ident: &syn::Ident,
) -> (Vec<TokenStream>, Vec<TokenStream>, Vec<TokenStream>) {
    let strategies = strategy::build_strategies(ir, config);
    let mut decls: Vec<TokenStream> = Vec::new();
    let mut pushes: Vec<TokenStream> = Vec::new();
    let mut builders: Vec<TokenStream> = Vec::new();
    for (idx, s) in strategies.iter().enumerate() {
        if let Some(bulk) = s.gen_bulk_columnar_emit(idx) {
            builders.extend(bulk);
        } else {
            decls.extend(s.gen_populator_inits(idx));
            pushes.push(s.gen_populator_push(it_ident, idx));
            builders.extend(s.gen_columnar_builders(idx));
        }
    }
    (decls, pushes, builders)
}

// --- Shared helpers and wrapper-aware builders ---

/// Build an efficient inner Series from a `Vec<T>` (`option_wrap == false`) or
/// `Vec<Option<T>>` (`option_wrap == true`). Applies the per-element transform
/// when one is required (stringify/casts), otherwise hands the slice straight
/// to `Series::new` for a zero-copy build. The Option-wrapped form mirrors the
/// non-Option form in three branches:
///
/// - `as_str` builds `Vec<Option<&str>>` via UFCS so the Series borrows from
///   the original storage instead of allocating a per-row `String`.
/// - `to_string` / decimal-stringify / datetime-to-i64 builds
///   `Vec<Option<U>>` where `U` is the mapping's `rust_element_type`, then
///   optionally `cast`s to the target dtype.
/// - No-transform passes `&Vec<Option<T>>` straight to `Series::new`, relying
///   on polars' `NamedFrom<&[Option<T>], _>` impl for the typed primitive.
pub fn generate_inner_series_from_vec(
    vec_access: &TokenStream,
    base_type: &BaseType,
    transform: Option<&PrimitiveTransform>,
    option_wrap: bool,
) -> TokenStream {
    let pp = super::polars_paths::prelude();

    // Borrowing path for `as_str`: build a `Vec<&str>` (or `Vec<Option<&str>>`)
    // through `AsRef<str>` via UFCS so the polars Series sees the same
    // `&[&str]` / `&[Option<&str>]` slice the bare-`String` borrowing path
    // uses. No per-element allocation.
    if matches!(transform, Some(PrimitiveTransform::AsStr)) {
        // The parser rejects `as_str` on any base outside this trio (see
        // `reject_as_str_on_incompatible_base`); reaching another arm here
        // would mean the IR was hand-constructed past that gate.
        let ty_path = match base_type {
            BaseType::Struct(ident, args) => super::strategy::build_type_path(ident, args.as_ref()),
            BaseType::Generic(ident) => quote! { #ident },
            BaseType::String => quote! { ::std::string::String },
            BaseType::F64
            | BaseType::F32
            | BaseType::I64
            | BaseType::U64
            | BaseType::I32
            | BaseType::U32
            | BaseType::I16
            | BaseType::U16
            | BaseType::I8
            | BaseType::U8
            | BaseType::Bool
            | BaseType::ISize
            | BaseType::USize
            | BaseType::DateTimeUtc
            | BaseType::Decimal => unreachable!(
                "df-derive: as_str on incompatible base type leaked past parser validation"
            ),
        };
        // `#vec_access` must be a place expression (e.g. `self.field` or a
        // named binding) — never a temporary that drops at the next `;`.
        // The on_vec callers all pass non-temp expressions; the deep-nesting
        // path in `gen_primitive_vec_inner_series` binds the inner clone to
        // a named local before recursing, so that recursion enters this
        // function with a name too.
        if option_wrap {
            return quote! {{
                let __df_derive_conv: ::std::vec::Vec<::std::option::Option<&str>> = (#vec_access)
                    .iter()
                    .map(|__df_derive_opt| __df_derive_opt.as_ref().map(<#ty_path as ::core::convert::AsRef<str>>::as_ref))
                    .collect();
                let inner_series = <#pp::Series as #pp::NamedFrom<_, _>>::new("".into(), &__df_derive_conv);
                inner_series
            }};
        }
        return quote! {{
            let __df_derive_conv: ::std::vec::Vec<&str> = (#vec_access)
                .iter()
                .map(<#ty_path as ::core::convert::AsRef<str>>::as_ref)
                .collect();
            let inner_series = <#pp::Series as #pp::NamedFrom<_, _>>::new("".into(), &__df_derive_conv);
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
        // Fallible conversions (currently only `DateTime<Utc>` →
        // `timestamp_nanos_opt`) embed a `?` in the mapped expression. A bare
        // `.map(|e| { #mapped }).collect::<Vec<_>>()` would treat the closure
        // body as the iter item, but `?` short-circuits the *closure* (because
        // its return type is inferred from the body) — so the closure ends up
        // returning `Result<i64, _>`, the collect target needs to match, and a
        // final `?` propagates to the outer fn. Pin the closure return type
        // with an explicit annotation so type inference doesn't surprise us
        // when callers add new fallible transforms.
        let fallible = crate::codegen::type_registry::is_fallible_conversion(transform);
        let collect_ts = match (option_wrap, fallible) {
            (false, false) => quote! {
                let __df_derive_conv: ::std::vec::Vec<_> = (#vec_access)
                    .iter()
                    .map(|#elem_ident| { #mapped })
                    .collect();
            },
            (false, true) => quote! {
                let __df_derive_conv: ::std::vec::Vec<_> = (#vec_access)
                    .iter()
                    .map(|#elem_ident| -> #pp::PolarsResult<_> { ::std::result::Result::Ok({ #mapped }) })
                    .collect::<#pp::PolarsResult<::std::vec::Vec<_>>>()?;
            },
            (true, false) => quote! {
                let __df_derive_conv: ::std::vec::Vec<::std::option::Option<_>> = (#vec_access)
                    .iter()
                    .map(|__df_derive_opt| __df_derive_opt.as_ref().map(|#elem_ident| { #mapped }))
                    .collect();
            },
            (true, true) => quote! {
                // Per-element fallible conversion threaded through the Option
                // layer: a `None` stays `None`, a `Some(v)` becomes either
                // `Some(mapped)` or short-circuits the outer try via `?`.
                let __df_derive_conv: ::std::vec::Vec<::std::option::Option<_>> = (#vec_access)
                    .iter()
                    .map(|__df_derive_opt| -> #pp::PolarsResult<::std::option::Option<_>> {
                        ::std::result::Result::Ok(match __df_derive_opt {
                            ::std::option::Option::Some(#elem_ident) => ::std::option::Option::Some({ #mapped }),
                            ::std::option::Option::None => ::std::option::Option::None,
                        })
                    })
                    .collect::<#pp::PolarsResult<::std::vec::Vec<::std::option::Option<_>>>>()?;
            },
        };
        quote! {{
            #collect_ts
            let mut inner_series = <#pp::Series as #pp::NamedFrom<_, _>>::new("".into(), &__df_derive_conv);
            if #do_cast { inner_series = inner_series.cast(&#dtype)?; }
            inner_series
        }}
    } else {
        // Transform-less branch covers both `Vec<T>` and `Vec<Option<T>>` — in
        // both cases polars' `NamedFrom<&[T_or_Option_T], _>` impls produce
        // the typed Series directly without any per-element allocation.
        quote! {{
            let mut inner_series = <#pp::Series as #pp::NamedFrom<_, _>>::new("".into(), &(#vec_access));
            if #do_cast { inner_series = inner_series.cast(&#dtype)?; }
            inner_series
        }}
    }
}

// removed unused generate_anyvalue_from_scalar

// moved to wrapping.rs and used via strategy

// moved to wrapping.rs and used via strategy

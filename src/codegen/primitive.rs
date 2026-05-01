// Codegen for fields whose base type is a primitive scalar (numeric, bool,
// String, DateTime, Decimal) or a struct/generic routed through a
// `to_string`/`as_str` transform. The three context-specific generators
// (`for_series`, `for_columnar_push`, `for_anyvalue`) share the wrapper
// traversal in `super::wrapper_processor::process_wrappers` and the
// borrow-classification logic in `classify_borrow`.

use crate::ir::{BaseType, PrimitiveTransform, Wrapper, has_option, has_vec, vec_count};
use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use super::populator_idents::PopulatorIdents;

/// Borrow strategy for a leaf that can populate the `Vec<&str>` /
/// `Vec<Option<&str>>` columnar buffer instead of an owning `Vec<String>`.
/// Only the bare leaf and bare `Option<…>` shapes flatten into one of these
/// two buffer layouts; `Vec<_>` wrappers (and deeper nestings) build their
/// inner Series via `common::generate_inner_series_from_vec`, which has its
/// own borrowing path for `as_str`.
enum BorrowKind {
    /// Bare `String` / `Option<String>` with no transform (or with the
    /// redundant `as_str` attribute, which behaves identically because
    /// `&String` deref-coerces to `&str`). Emit `&(#access)` and
    /// `(#access).as_deref()`.
    StringLeaf,
    /// `as_str` on a non-`String` base type. Emit
    /// `<T as ::core::convert::AsRef<str>>::as_ref(&(#access))` (UFCS) and
    /// the analogous `Option`-mapped form. The carried token is the
    /// type-path expression suitable for splicing into UFCS (`Foo`,
    /// `Foo::<M>`, or `T`).
    AsStr(TokenStream),
}

/// Classify whether a primitive leaf can use the borrowing fast path. Returns
/// `None` for `Vec<_>`-wrapped or deeper-nested fields, and for any base/
/// transform combination that requires an owned buffer.
fn classify_borrow(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> Option<BorrowKind> {
    if !matches!(wrappers, [] | [Wrapper::Option]) {
        return None;
    }
    match (base, transform) {
        (BaseType::String, None | Some(PrimitiveTransform::AsStr)) => Some(BorrowKind::StringLeaf),
        (BaseType::Struct(ident, args), Some(PrimitiveTransform::AsStr)) => Some(
            BorrowKind::AsStr(super::strategy::build_type_path(ident, args.as_ref())),
        ),
        (BaseType::Generic(ident), Some(PrimitiveTransform::AsStr)) => {
            Some(BorrowKind::AsStr(quote! { #ident }))
        }
        // `AsStr` on a non-string primitive base would be meaningless; the
        // per-field `AsRef<str>` const-fn assert in helpers.rs catches it
        // with a clean error span. Falling through here lets the existing
        // owning-buffer path handle compilation up to that assert.
        _ => None,
    }
}

/// Build tokens that evaluate to a `polars::prelude::Series` representing
/// the inner list for a primitive vector (including nested wrappers in `tail`).
fn gen_primitive_vec_inner_series(
    acc: &TokenStream,
    base_type: &BaseType,
    transform: Option<&PrimitiveTransform>,
    tail: &[Wrapper],
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    let elem_ident = syn::Ident::new("__df_derive_vec_elem", proc_macro2::Span::call_site());
    let list_vals_ident = syn::Ident::new("__df_derive_list_vals", proc_macro2::Span::call_site());

    let base_is_struct = matches!(base_type, BaseType::Struct(..));
    // `as_str` routes Vec<T> and Vec<Option<T>> shapes (including with struct
    // bases) through the borrowing fast path: `generate_inner_series_from_vec`
    // already builds `Vec<&str>` / `Vec<Option<&str>>` via UFCS. Without this
    // carve-out, `Vec<Struct>+as_str` would fall to the recursive per-element
    // loop and clone every element.
    let as_str_fast_ok = matches!(transform, Some(PrimitiveTransform::AsStr));

    if (!base_is_struct || as_str_fast_ok) && tail.is_empty() {
        let fast_inner_ts =
            super::common::generate_inner_series_from_vec(acc, base_type, transform, false);
        return quote! {{ { #fast_inner_ts } }};
    }
    // `Vec<Option<T>>` shapes for every (base, transform) combination that
    // reaches this codegen path: the unified `generate_inner_series_from_vec`
    // builds a typed `Vec<Option<U>>` (or `Vec<Option<&str>>` for `as_str`)
    // and hands it to `Series::new`, with an optional `cast` for transforms
    // that go through a stand-in dtype (i64 → Datetime, String → Decimal).
    // Struct/Generic bases only land here paired with an `AsStr`/`ToString`
    // transform per `build_strategies`, so all reachable combinations are
    // representable as one of the unified branches.
    if (!base_is_struct || as_str_fast_ok)
        && let [Wrapper::Option] = tail
    {
        let fast_inner_ts =
            super::common::generate_inner_series_from_vec(acc, base_type, transform, true);
        return quote! {{ { #fast_inner_ts } }};
    }

    // Fallback recursive per-element path (rare wrapper depths and any
    // non-`as_str` `Vec<Struct>`-with-tail). Eagerly emitting the fast-path
    // tokens above for shapes that don't use them risks unrelated codegen
    // errors leaking into the user's output, so build them lazily.
    let mapping = crate::codegen::type_registry::compute_mapping(base_type, transform, tail);
    let elem_dtype = mapping.element_dtype;
    let do_cast = crate::codegen::type_registry::needs_cast(transform);
    // For empty input we have to construct a typed empty Series — `Series::new`
    // on an empty `Vec<AnyValue>` produces dtype Null, which strict-typed list
    // builders (e.g. `ListPrimitiveChunkedBuilder`, `ListStringChunkedBuilder`)
    // reject when the parent feeds the empty inner via `append_series`. The
    // typed-empty needs the same per-tail-Vec wrapping the non-empty Series's
    // inferred dtype gets (each `AnyValue::List(...)` adds one `List<>` layer
    // in inference), so wrap `elem_dtype` once per remaining `Vec` in `tail`.
    // Resolved at codegen time, so emit the wrap layers as nested tokens
    // rather than a runtime `for _ in 0..tail_vec_count` (which would fire
    // `clippy::reversed_empty_ranges` for `tail_vec_count == 0`).
    let tail_vec_count = vec_count(tail);
    let mut empty_dtype = elem_dtype.clone();
    for _ in 0..tail_vec_count {
        empty_dtype = quote! { #pp::DataType::List(::std::boxed::Box::new(#empty_dtype)) };
    }
    // The `for #elem_ident in (#acc).iter()` binding is itself a long-lived
    // place expression (a `&T_full` referencing storage owned by `#acc`), so
    // the inner recursion can borrow from it directly. Cloning into an owned
    // local would have forced `T: Clone` on every generic param of the
    // enclosing struct — overly restrictive when only this fallback path
    // (deeper-than-`Vec<T>` shapes with a transform) ever needs the borrow.
    //
    // `generate_primitive_for_anyvalue` always emits exactly one `AnyValue`
    // push per call (one per `Vec` element), so route those pushes straight
    // into the outer list buffer instead of paying for a per-element scratch
    // `Vec` + `pop()` round-trip. That also drops a runtime `polars_err!`
    // branch from the generated code that statically cannot fire.
    let elem_access = quote! { #elem_ident };
    let recur_elem_tokens_ts =
        generate_primitive_for_anyvalue(&list_vals_ident, &elem_access, base_type, transform, tail);
    quote! {{
        let mut #list_vals_ident: ::std::vec::Vec<#pp::AnyValue> = ::std::vec::Vec::with_capacity((#acc).len());
        for #elem_ident in (#acc).iter() {
            #recur_elem_tokens_ts
        }
        let __df_derive_inner = if #list_vals_ident.is_empty() {
            #pp::Series::new_empty("".into(), &#empty_dtype)
        } else {
            let mut __df_derive_s = <#pp::Series as #pp::NamedFrom<_, _>>::new("".into(), &#list_vals_ident);
            if #do_cast { __df_derive_s = __df_derive_s.cast(&#elem_dtype)?; }
            __df_derive_s
        };
        __df_derive_inner
    }}
}

// --- Context-specific generators ---

pub fn generate_primitive_for_columnar_push(
    access: &TokenStream,
    base_type: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
    idx: usize,
) -> TokenStream {
    // Borrowing fast path: the buffer is declared as `Vec<&str>` /
    // `Vec<Option<&str>>` by `primitive_decls`, so we push borrows of the
    // field instead of cloning each row's `String` into an owned buffer.
    // The borrows live as long as `items`, which outlives the buffer.
    if let Some(kind) = classify_borrow(base_type, transform, wrappers) {
        let vec_ident = PopulatorIdents::primitive_buf(idx);
        let opt = has_option(wrappers);
        return match kind {
            BorrowKind::StringLeaf => {
                if opt {
                    quote! { #vec_ident.push((#access).as_deref()); }
                } else {
                    quote! { #vec_ident.push(&(#access)); }
                }
            }
            BorrowKind::AsStr(ty_path) => {
                if opt {
                    quote! {
                        #vec_ident.push(
                            (#access).as_ref().map(<#ty_path as ::core::convert::AsRef<str>>::as_ref)
                        );
                    }
                } else {
                    quote! {
                        #vec_ident.push(<#ty_path as ::core::convert::AsRef<str>>::as_ref(&(#access)));
                    }
                }
            }
        };
    }

    let opt_scalar = has_option(wrappers) && !has_vec(wrappers);

    let on_leaf = |acc: &TokenStream| {
        let vec_ident = PopulatorIdents::primitive_buf(idx);
        let mapped = super::common::generate_primitive_access_expr(acc, transform);
        if opt_scalar {
            quote! { #vec_ident.push(::std::option::Option::Some({ #mapped })); }
        } else {
            quote! { #vec_ident.push({ #mapped }); }
        }
    };

    let on_option_none = |tail: &[Wrapper]| {
        let tail_has_vec = has_vec(tail);
        if tail_has_vec {
            let lb_ident = PopulatorIdents::primitive_list_builder(idx);
            let pp = super::polars_paths::prelude();
            quote! { #pp::ListBuilderTrait::append_null(&mut *#lb_ident); }
        } else {
            let vec_ident = PopulatorIdents::primitive_buf(idx);
            quote! { #vec_ident.push(::std::option::Option::None); }
        }
    };

    let on_vec = |acc: &TokenStream, tail: &[Wrapper]| {
        let inner_series_ts = gen_primitive_vec_inner_series(acc, base_type, transform, tail);
        let lb_ident = PopulatorIdents::primitive_list_builder(idx);
        let pp = super::polars_paths::prelude();
        quote! {{
            let inner_series = { #inner_series_ts };
            #pp::ListBuilderTrait::append_series(&mut *#lb_ident, &inner_series)?;
        }}
    };

    super::wrapper_processor::process_wrappers(access, wrappers, &on_leaf, &on_option_none, &on_vec)
}

pub fn generate_primitive_for_anyvalue(
    values_vec_ident: &Ident,
    access: &TokenStream,
    base_type: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> TokenStream {
    let pp = super::polars_paths::prelude();

    let on_leaf = |acc: &TokenStream| {
        // Direct AnyValue construction — one push per leaf, no detour through
        // a 1-element `Series` + `get(0)?.into_static()`. At ~10M leaves for a
        // 10-field × 1M-row aggregation that's a measurable saving.
        let av = match classify_borrow(base_type, transform, wrappers) {
            Some(BorrowKind::StringLeaf) => {
                quote! { #pp::AnyValue::StringOwned((#acc).clone().into()) }
            }
            Some(BorrowKind::AsStr(ty_path)) => quote! {
                #pp::AnyValue::StringOwned(
                    <#ty_path as ::core::convert::AsRef<str>>::as_ref(&(#acc)).to_string().into()
                )
            },
            None => {
                let mapped = super::common::generate_primitive_access_expr(acc, transform);
                crate::codegen::type_registry::anyvalue_static_expr(base_type, transform, &mapped)
            }
        };
        quote! { #values_vec_ident.push({ #av }); }
    };

    let on_option_none = |_tail: &[Wrapper]| {
        quote! { #values_vec_ident.push(#pp::AnyValue::Null); }
    };

    let on_vec = |acc: &TokenStream, tail: &[Wrapper]| {
        let inner_series_ts = gen_primitive_vec_inner_series(acc, base_type, transform, tail);
        quote! {{
            let inner_series = { #inner_series_ts };
            #values_vec_ident.push(#pp::AnyValue::List(inner_series));
        }}
    };

    super::wrapper_processor::process_wrappers(access, wrappers, &on_leaf, &on_option_none, &on_vec)
}

// --- Columnar populator decls and finishers ---

pub fn primitive_decls(
    wrappers: &[Wrapper],
    base_type: &BaseType,
    transform: Option<&PrimitiveTransform>,
    idx: usize,
) -> Vec<TokenStream> {
    let mut decls: Vec<TokenStream> = Vec::new();
    let opt = has_option(wrappers);
    let vec = has_vec(wrappers);

    // Borrowing fast path for `String` / `Option<String>` and any base type
    // with `as_str` (`AsRef<str>` impl): a `Vec<&str>` (or
    // `Vec<Option<&str>>`) buffer borrows from `items` instead of cloning each
    // row's `String`. `Series::new(name, &Vec<&str>)` dispatches to
    // `StringChunked::from_slice` and produces the same `Utf8ViewArray`-backed
    // column the owning path produces.
    if classify_borrow(base_type, transform, wrappers).is_some() {
        let vec_ident = PopulatorIdents::primitive_buf(idx);
        if opt {
            decls.push(quote! {
                let mut #vec_ident: ::std::vec::Vec<::std::option::Option<&str>> =
                    ::std::vec::Vec::with_capacity(items.len());
            });
        } else {
            decls.push(quote! {
                let mut #vec_ident: ::std::vec::Vec<&str> =
                    ::std::vec::Vec::with_capacity(items.len());
            });
        }
        return decls;
    }

    let mapping = crate::codegen::type_registry::compute_mapping(base_type, transform, wrappers);
    let elem_rust_ty = mapping.rust_element_type;
    if vec {
        // The outer column is `List<inner_dtype>`. Push per-row inner Series
        // through `ListBuilderTrait::append_series` rather than collecting
        // `Vec<AnyValue::List>` and rebuilding via `Series::new` — that
        // intermediate paid for an inferring scan over the AnyValue vec plus a
        // `cast(inner_type)` per row inside Polars' `any_values_to_list`.
        // `get_list_builder` returns a typed builder
        // (`ListPrimitiveChunkedBuilder` for numeric, `ListStringChunkedBuilder`
        // for strings, etc.) so the inner buffer stays typed end-to-end.
        //
        // For nested-Vec shapes (`Vec<Vec<T>>`, `Vec<Vec<Vec<T>>>`, …) the
        // per-row inner Series is itself `List<…>`-shaped, so the builder's
        // expected inner dtype must include those extra list layers — see
        // `outer_list_inner_dtype`. Using `element_dtype` here would
        // wrong-foot the strict-typed builder (`ListPrimitiveChunkedBuilder`
        // unpacks via the inner Native type and rejects a `list[…]` slice).
        let lb_ident = PopulatorIdents::primitive_list_builder(idx);
        let inner_dtype =
            crate::codegen::type_registry::outer_list_inner_dtype(base_type, transform, wrappers);
        let pp = super::polars_paths::prelude();
        let cab = super::polars_paths::chunked_array_builder();
        decls.push(quote! {
            let mut #lb_ident: ::std::boxed::Box<dyn #pp::ListBuilderTrait> =
                #cab::get_list_builder(
                    &#inner_dtype,
                    items.len() * 4,
                    items.len(),
                    "".into(),
                );
        });
    } else {
        let vec_ident = PopulatorIdents::primitive_buf(idx);
        if opt {
            decls.push(quote! { let mut #vec_ident: ::std::vec::Vec<::std::option::Option<#elem_rust_ty>> = ::std::vec::Vec::with_capacity(items.len()); });
        } else {
            decls.push(quote! { let mut #vec_ident: ::std::vec::Vec<#elem_rust_ty> = ::std::vec::Vec::with_capacity(items.len()); });
        }
    }
    decls
}

pub fn primitive_finishers_for_vec_anyvalues(
    wrappers: &[Wrapper],
    base_type: &BaseType,
    transform: Option<&PrimitiveTransform>,
    idx: usize,
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    let vec = has_vec(wrappers);
    let mapping = crate::codegen::type_registry::compute_mapping(base_type, transform, wrappers);
    let needs_cast = crate::codegen::type_registry::needs_cast(transform);
    if vec {
        let lb_ident = PopulatorIdents::primitive_list_builder(idx);
        quote! {
            let inner = #pp::IntoSeries::into_series(
                #pp::ListBuilderTrait::finish(&mut *#lb_ident),
            );
            out_values.push(#pp::AnyValue::List(inner));
        }
    } else {
        let dtype = mapping.full_dtype;
        let vec_ident = PopulatorIdents::primitive_buf(idx);
        quote! {
            let mut inner = <#pp::Series as #pp::NamedFrom<_, _>>::new("".into(), &#vec_ident);
            if #needs_cast { inner = inner.cast(&#dtype)?; }
            out_values.push(#pp::AnyValue::List(inner));
        }
    }
}

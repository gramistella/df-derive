// Codegen for fields whose base type is a primitive scalar (numeric, bool,
// String, DateTime, Decimal) or a struct/generic routed through a
// `to_string`/`as_str` transform. The three context-specific generators
// (`for_series`, `for_columnar_push`, `for_anyvalue`) share the wrapper
// traversal in `super::wrapper_processor::process_wrappers` and the
// borrow-classification logic in `classify_borrow`.

use crate::ir::{
    BaseType, DateTimeUnit, PrimitiveTransform, Wrapper, has_option, has_vec, vec_count,
};
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
    decimal128_encode_trait: &TokenStream,
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
        let fast_inner_ts = super::common::generate_inner_series_from_vec(
            acc,
            base_type,
            transform,
            false,
            decimal128_encode_trait,
        );
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
        let fast_inner_ts = super::common::generate_inner_series_from_vec(
            acc,
            base_type,
            transform,
            true,
            decimal128_encode_trait,
        );
        return quote! {{ { #fast_inner_ts } }};
    }

    // `Vec<Vec<…>>` (and deeper) — replace the outer `Vec<AnyValue::List(...)>`
    // aggregation + inferring `Series::new` with a typed `ListBuilder` whose
    // inner dtype matches the recursive call's output. Per outer element we
    // recursively build a typed inner Series via the fast paths above (or a
    // deeper Step-3 layer for `Vec<Vec<Vec<…>>>`), then `append_series` — no
    // per-row `AnyValue::List` allocation, no inferring scan over a
    // `Vec<AnyValue>` at the end.
    if let [Wrapper::Vec, rest @ ..] = tail {
        let cab = super::polars_paths::chunked_array_builder();
        let mut inner_series_dtype =
            crate::codegen::type_registry::compute_mapping(base_type, transform, &[]).element_dtype;
        // Each remaining `Vec` in `rest` adds one `List<>` layer to the inner
        // Series's runtime dtype (Option layers don't add a layer — Polars
        // carries nullability in values). The strict-typed builder below
        // rejects an `append_series` whose dtype doesn't match `inner_dtype`,
        // so this wrap has to track the recursive call's output exactly.
        for _ in 0..vec_count(rest) {
            inner_series_dtype =
                quote! { #pp::DataType::List(::std::boxed::Box::new(#inner_series_dtype)) };
        }
        let inner_series_ts = gen_primitive_vec_inner_series(
            &quote! { #elem_ident },
            base_type,
            transform,
            rest,
            decimal128_encode_trait,
        );
        return quote! {{
            let mut __df_derive_lb: ::std::boxed::Box<dyn #pp::ListBuilderTrait> =
                #cab::get_list_builder(
                    &#inner_series_dtype,
                    (#acc).len() * 4,
                    (#acc).len(),
                    "".into(),
                );
            for #elem_ident in (#acc).iter() {
                let __df_derive_inner_series = { #inner_series_ts };
                #pp::ListBuilderTrait::append_series(
                    &mut *__df_derive_lb,
                    &__df_derive_inner_series,
                )?;
            }
            #pp::IntoSeries::into_series(#pp::ListBuilderTrait::finish(&mut *__df_derive_lb))
        }};
    }

    // Fallback recursive per-element path: any tail starting with `Option` and
    // having a non-empty rest (e.g. `Vec<Option<Vec<T>>>` or
    // `Vec<Option<Option<T>>>`), and `Vec<Struct>`-with-tail shapes that pair
    // a struct base with a non-`AsStr` transform. Eagerly emitting the
    // fast-path tokens above for shapes that don't use them risks unrelated
    // codegen errors leaking into the user's output, so build them lazily.
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
    let recur_elem_tokens_ts = generate_primitive_for_anyvalue(
        &list_vals_ident,
        &elem_access,
        base_type,
        transform,
        tail,
        decimal128_encode_trait,
    );
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
    decimal128_encode_trait: &TokenStream,
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
        let mapped =
            super::common::generate_primitive_access_expr(acc, transform, decimal128_encode_trait);
        if opt_scalar {
            quote! { #vec_ident.push(::std::option::Option::Some({ #mapped })); }
        } else {
            quote! { #vec_ident.push({ #mapped }); }
        }
    };

    // Typed-builder shapes: `Vec<Decimal>` / `Vec<Option<Decimal>>` and
    // `Vec<DateTime>` / `Vec<Option<DateTime>>`. The decl emitted in
    // `primitive_decls` is a concrete `ListPrimitiveChunkedBuilder<Native>`,
    // not a `Box<dyn ListBuilderTrait>`, so the push site references it by
    // value (no `&mut *` deref) and reaches `append_iter` / `append_null`
    // inherently / via the trait.
    let typed_info = typed_primitive_list_info(base_type, transform, wrappers);

    let on_option_none = |tail: &[Wrapper]| {
        let tail_has_vec = has_vec(tail);
        if tail_has_vec {
            let lb_ident = PopulatorIdents::primitive_list_builder(idx);
            let pp = super::polars_paths::prelude();
            if typed_info.is_some() {
                quote! { #pp::ListBuilderTrait::append_null(&mut #lb_ident); }
            } else {
                quote! { #pp::ListBuilderTrait::append_null(&mut *#lb_ident); }
            }
        } else {
            let vec_ident = PopulatorIdents::primitive_buf(idx);
            quote! { #vec_ident.push(::std::option::Option::None); }
        }
    };

    let on_vec = |acc: &TokenStream, tail: &[Wrapper]| {
        let lb_ident = PopulatorIdents::primitive_list_builder(idx);
        if let Some(info) = typed_info.as_ref() {
            return gen_typed_list_append(
                acc,
                tail,
                base_type,
                transform,
                info,
                &lb_ident,
                decimal128_encode_trait,
            );
        }
        let inner_series_ts = gen_primitive_vec_inner_series(
            acc,
            base_type,
            transform,
            tail,
            decimal128_encode_trait,
        );
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
    decimal128_encode_trait: &TokenStream,
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
                let mapped = super::common::generate_primitive_access_expr(
                    acc,
                    transform,
                    decimal128_encode_trait,
                );
                crate::codegen::type_registry::anyvalue_static_expr(base_type, transform, &mapped)
            }
        };
        quote! { #values_vec_ident.push({ #av }); }
    };

    let on_option_none = |_tail: &[Wrapper]| {
        quote! { #values_vec_ident.push(#pp::AnyValue::Null); }
    };

    let on_vec = |acc: &TokenStream, tail: &[Wrapper]| {
        let inner_series_ts = gen_primitive_vec_inner_series(
            acc,
            base_type,
            transform,
            tail,
            decimal128_encode_trait,
        );
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
        let pp = super::polars_paths::prelude();
        // Single-`Vec` shapes covered by `typed_primitive_list_info` use a
        // concrete typed builder so the per-parent-row push site can call
        // `append_iter` directly — bypassing `Series::new` + cast +
        // `append_series`'s `to_physical_repr` round-trip. Other primitive
        // shapes (e.g. nested `Vec<Vec<…>>`, `Vec<String>`, `Vec<bool>`,
        // struct-based leaves) keep using the `Box<dyn>` path because their
        // Native types don't match what `ListPrimitiveChunkedBuilder` accepts
        // or because the typed route was measured slower for them.
        if let Some(info) = typed_primitive_list_info(base_type, transform, wrappers) {
            match &info.kind {
                BuilderKind::Primitive(p) => {
                    let native = &p.native_type;
                    let mapping_inner =
                        crate::codegen::type_registry::compute_mapping(base_type, transform, &[]);
                    let inner_logical = mapping_inner.element_dtype;
                    let constructor = if p.needs_values_type {
                        // Decimal: physical `Int128` differs from logical `Decimal(p, s)`.
                        // `new_with_values_type` gives the chunked array its physical
                        // storage type while the field carries the logical dtype, so
                        // the finished `ListChunked`'s schema matches `T::schema()`
                        // exactly without a post-finish cast.
                        quote! {
                            #pp::ListPrimitiveChunkedBuilder::<#native>::new_with_values_type(
                                "".into(),
                                items.len(),
                                items.len() * 4,
                                #pp::DataType::Int128,
                                #inner_logical,
                            )
                        }
                    } else {
                        // DateTime / bare numerics: physical matches what the buffer
                        // stores; the field dtype is the schema's logical dtype
                        // directly.
                        quote! {
                            #pp::ListPrimitiveChunkedBuilder::<#native>::new(
                                "".into(),
                                items.len(),
                                items.len() * 4,
                                #inner_logical,
                            )
                        }
                    };
                    decls.push(quote! {
                        let mut #lb_ident: #pp::ListPrimitiveChunkedBuilder<#native> = #constructor;
                    });
                }
                BuilderKind::String => {
                    // `value_capacity` is the inner-builder's `MutableBinaryViewArray`
                    // pre-allocation. `items.len() * 4` matches the heuristic the
                    // `Box<dyn>` path's `get_list_builder` uses for strings (5x
                    // value_capacity at the call site, divided through), and the
                    // builder reallocates as needed regardless.
                    decls.push(quote! {
                        let mut #lb_ident: #pp::ListStringChunkedBuilder = #pp::ListStringChunkedBuilder::new(
                            "".into(),
                            items.len(),
                            items.len() * 4,
                        );
                    });
                }
            }
        } else {
            let inner_dtype = crate::codegen::type_registry::outer_list_inner_dtype(
                base_type, transform, wrappers,
            );
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
        }
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
        // Typed builder: call inherent `finish`-via-trait without `&mut *` deref.
        // Boxed-dyn builder: dereference the Box first.
        let builder_ref = if typed_primitive_list_info(base_type, transform, wrappers).is_some() {
            quote! { &mut #lb_ident }
        } else {
            quote! { &mut *#lb_ident }
        };
        quote! {
            let inner = #pp::IntoSeries::into_series(
                #pp::ListBuilderTrait::finish(#builder_ref),
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

/// Concrete typed-builder declaration parameters for the single-`Vec`
/// primitive shapes that benefit from bypassing the
/// `Box<dyn ListBuilderTrait>` path. Currently covers:
/// - `Vec<Decimal>` / `Vec<Option<Decimal>>` and `Vec<DateTime>` /
///   `Vec<Option<DateTime>>`.
/// - Bare numerics (`i8/i16/i32/i64/u8/u16/u32/u64/f32/f64`) without a
///   transform — `Vec<T>` / `Vec<Option<T>>` shapes feed an
///   `Iterator<Item = Option<Native>> + TrustedLen` straight into
///   `append_iter`, avoiding `Series::new`, the inferring scan, and the
///   `to_physical_repr`/`unpack` round-trip inside `append_series`.
/// - `Vec<String>` / `Vec<Option<String>>` — feed an
///   `Iterator<Item = Option<&str>> + TrustedLen` straight into the typed
///   `ListStringChunkedBuilder::append_trusted_len_iter`, avoiding the
///   per-row `Vec<&str>` + `Series::new` + `append_series` round-trip.
///
/// Returns `None` for any shape that doesn't have exactly one `Vec` layer or
/// for base/transform combinations not in the list above — those keep using
/// the existing `get_list_builder` -> `Box<dyn ListBuilderTrait>` path.
pub(super) struct TypedListInfo {
    pub kind: BuilderKind,
}

/// Which concrete typed list builder a given primitive shape resolves to.
/// `Primitive` covers everything that fits `ListPrimitiveChunkedBuilder<T>`;
/// `String` carries no extra data because `ListStringChunkedBuilder` has a
/// single concrete type with no native-type parameter.
pub(super) enum BuilderKind {
    Primitive(PrimitiveBuilderInfo),
    String,
}

pub(super) struct PrimitiveBuilderInfo {
    /// Polars chunked-array native type token (`Int128Type`, `Int64Type`,
    /// `Int32Type`, …). Splices into `ListPrimitiveChunkedBuilder<#native_type>`.
    pub native_type: TokenStream,
    /// Native rust element type (`i128`, `i64`, …). Used by the fallible
    /// branch in `gen_typed_list_append` to type the per-row scratch
    /// `Vec<Option<#native_rust>>` whose `into_iter()` feeds `append_iter`.
    /// The non-fallible branch doesn't need it (it iterates the source Vec
    /// directly).
    pub native_rust: TokenStream,
    /// Whether the per-element conversion is fallible. Decimal rescale always
    /// is (overflow on scale-up; overflow via the trait is also fallible);
    /// `DateTime` is fallible only at nanosecond precision
    /// (`timestamp_nanos_opt` returns `None` outside ~[1677, 2262]). Bare
    /// numerics are infallible.
    pub fallible: bool,
    /// Whether the schema dtype needs `new_with_values_type` (logical dtype
    /// differs from physical). Decimal does — its physical is `Int128` but
    /// the schema dtype is `Decimal(p, s)`. `DateTime` and bare numerics
    /// don't — physical matches what `::new` stores in the field, and the
    /// cast to the field dtype is metadata-only when the buffer's already
    /// the right Native type.
    pub needs_values_type: bool,
}

pub(super) fn typed_primitive_list_info(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> Option<TypedListInfo> {
    if vec_count(wrappers) != 1 {
        return None;
    }
    let pp = super::polars_paths::prelude();
    let primitive = |native_type: TokenStream,
                     native_rust: TokenStream,
                     fallible: bool,
                     needs_values_type: bool| {
        Some(TypedListInfo {
            kind: BuilderKind::Primitive(PrimitiveBuilderInfo {
                native_type,
                native_rust,
                fallible,
                needs_values_type,
            }),
        })
    };
    let numeric = |native_type: TokenStream, native_rust: TokenStream| {
        primitive(native_type, native_rust, false, false)
    };
    match (base, transform) {
        (BaseType::Decimal, Some(PrimitiveTransform::DecimalToInt128 { .. })) => {
            primitive(quote! { #pp::Int128Type }, quote! { i128 }, true, true)
        }
        (BaseType::DateTimeUtc, Some(PrimitiveTransform::DateTimeToInt(unit))) => primitive(
            quote! { #pp::Int64Type },
            quote! { i64 },
            matches!(unit, DateTimeUnit::Nanoseconds),
            false,
        ),
        (BaseType::I8, None) => numeric(quote! { #pp::Int8Type }, quote! { i8 }),
        (BaseType::I16, None) => numeric(quote! { #pp::Int16Type }, quote! { i16 }),
        (BaseType::I32, None) => numeric(quote! { #pp::Int32Type }, quote! { i32 }),
        (BaseType::I64, None) => numeric(quote! { #pp::Int64Type }, quote! { i64 }),
        (BaseType::U8, None) => numeric(quote! { #pp::UInt8Type }, quote! { u8 }),
        (BaseType::U16, None) => numeric(quote! { #pp::UInt16Type }, quote! { u16 }),
        (BaseType::U32, None) => numeric(quote! { #pp::UInt32Type }, quote! { u32 }),
        (BaseType::U64, None) => numeric(quote! { #pp::UInt64Type }, quote! { u64 }),
        (BaseType::F32, None) => numeric(quote! { #pp::Float32Type }, quote! { f32 }),
        (BaseType::F64, None) => numeric(quote! { #pp::Float64Type }, quote! { f64 }),
        // `Vec<String>` / `Vec<Option<String>>` go through
        // `ListStringChunkedBuilder::append_trusted_len_iter`. The `as_str`
        // transform path is handled separately via `classify_borrow` and the
        // `Vec<&str>` borrowing buffer; we only catch the no-transform
        // shape here so we don't double-route.
        (BaseType::String, None) => Some(TypedListInfo {
            kind: BuilderKind::String,
        }),
        // `BaseType::Bool` is intentionally NOT routed through the typed
        // `ListBooleanChunkedBuilder` fast path. Its `append_iter` goes
        // through `extend_trusted_len_unchecked`, which always pushes a
        // validity bit per element (regardless of nullability), whereas the
        // slow path's `Series::new(&Vec<bool>) + append_series` goes through
        // `BooleanType::from_slice` (bulk, no validity for non-null input)
        // and is faster overall. Bench 12 (`Vec<bool>`) measured a ~45%
        // regression with the typed route — so bool keeps the existing
        // boxed-dyn dispatch.
        //
        // `BaseType::ISize` / `BaseType::USize` are also omitted: their
        // materialized buffer type (`i64` / `u64`) doesn't match the field's
        // native `isize` / `usize`, so the per-element `(__df_derive_e).clone()`
        // produces the wrong Native type for the typed builder. The slow
        // path keeps working through the inferring `Series::new` -> cast.
        _ => None,
    }
}

/// Emit typed-builder inner-list materialization for the typed-builder fast
/// path. `inner_tail` is what remains after the outer `Vec<…>` is consumed —
/// must be `[]` (`Vec<T>` shape) or `[Wrapper::Option]` (`Vec<Option<T>>`
/// shape); other shapes don't reach this helper because
/// `typed_primitive_list_info` requires `vec_count == 1` and the only
/// single-Vec inner shapes are bare and Option.
fn gen_typed_list_append(
    vec_access: &TokenStream,
    inner_tail: &[Wrapper],
    base_type: &BaseType,
    transform: Option<&PrimitiveTransform>,
    info: &TypedListInfo,
    builder_ident: &Ident,
    decimal128_encode_trait: &TokenStream,
) -> TokenStream {
    let inner_opt = matches!(inner_tail, [Wrapper::Option]);
    match &info.kind {
        BuilderKind::Primitive(p) => gen_typed_primitive_list_append(
            vec_access,
            inner_opt,
            base_type,
            transform,
            p,
            builder_ident,
            decimal128_encode_trait,
        ),
        BuilderKind::String => gen_typed_string_list_append(vec_access, inner_opt, builder_ident),
    }
}

fn gen_typed_primitive_list_append(
    vec_access: &TokenStream,
    inner_opt: bool,
    base_type: &BaseType,
    transform: Option<&PrimitiveTransform>,
    info: &PrimitiveBuilderInfo,
    builder_ident: &Ident,
    decimal128_encode_trait: &TokenStream,
) -> TokenStream {
    let elem_ident = syn::Ident::new("__df_derive_e", proc_macro2::Span::call_site());
    let mapped = super::common::generate_primitive_access_expr(
        &quote! { #elem_ident },
        transform,
        decimal128_encode_trait,
    );
    let native_rust = &info.native_rust;

    // The trait-import idiom matches `map_primitive_expr`: anonymously bring
    // `Decimal128Encode` into scope so the trait method on the receiver
    // resolves via dot-syntax method resolution. Idempotent for non-Decimal
    // shapes (the import is unused but doesn't introduce a name).
    let import_trait = match (base_type, transform) {
        (BaseType::Decimal, Some(PrimitiveTransform::DecimalToInt128 { .. })) => {
            quote! { use #decimal128_encode_trait as _; }
        }
        _ => quote! {},
    };

    if info.fallible {
        // Fallible per-element conversion (`?` inside `mapped`, plus the
        // `return Err(polars_err!(...))` path baked into the Decimal rescale
        // expression). Collect into a `Vec<Option<Native>>` first so the
        // closure short-circuits cleanly through the outer `?`, then drive
        // `append_iter` from the owned Vec (`std::vec::IntoIter` is
        // `TrustedLen`).
        //
        // The closure return type pins both type parameters explicitly:
        // `Option<Native>` for the Ok variant, `PolarsError` for the Err
        // variant. The `_` placeholder inferred from the `return Err(
        // polars_err!(...))` path doesn't propagate outward through the
        // proc-macro context (rustc reports E0282 on the derive site), so
        // the explicit `PolarsError` bypasses that.
        let pp = super::polars_paths::prelude();
        let collect_ts = if inner_opt {
            quote! {
                let __df_derive_conv: ::std::vec::Vec<::std::option::Option<#native_rust>> = (#vec_access)
                    .iter()
                    .map(|__df_derive_opt| -> ::std::result::Result<::std::option::Option<#native_rust>, #pp::PolarsError> {
                        ::std::result::Result::Ok(match __df_derive_opt {
                            ::std::option::Option::Some(#elem_ident) => {
                                ::std::option::Option::Some({ #mapped })
                            }
                            ::std::option::Option::None => ::std::option::Option::None,
                        })
                    })
                    .collect::<::std::result::Result<::std::vec::Vec<::std::option::Option<#native_rust>>, #pp::PolarsError>>()?;
            }
        } else {
            quote! {
                let __df_derive_conv: ::std::vec::Vec<::std::option::Option<#native_rust>> = (#vec_access)
                    .iter()
                    .map(|#elem_ident| -> ::std::result::Result<::std::option::Option<#native_rust>, #pp::PolarsError> {
                        ::std::result::Result::Ok(::std::option::Option::Some({ #mapped }))
                    })
                    .collect::<::std::result::Result<::std::vec::Vec<::std::option::Option<#native_rust>>, #pp::PolarsError>>()?;
            }
        };
        quote! {{
            #import_trait
            #collect_ts
            #builder_ident.append_iter(__df_derive_conv.into_iter());
        }}
    } else {
        // Non-fallible: feed `Iter<Option<Native>>` straight to `append_iter`.
        // `std::slice::Iter` is `TrustedLen`; `Map<TrustedLen, _>` preserves it.
        // This avoids the per-row scratch `Vec` allocation entirely.
        let map_expr = if inner_opt {
            quote! {
                .map(|__df_derive_opt| __df_derive_opt
                    .as_ref()
                    .map(|#elem_ident| { #mapped }))
            }
        } else {
            quote! {
                .map(|#elem_ident| ::std::option::Option::Some({ #mapped }))
            }
        };
        quote! {{
            #builder_ident.append_iter((#vec_access).iter() #map_expr);
        }}
    }
}

/// Emit `append_trusted_len_iter`-based inner-list materialization for the
/// typed `ListStringChunkedBuilder` path. The iterator yields `Option<&str>`
/// borrowing from the parent's Vec; `slice::Iter::map(...)` is `TrustedLen`
/// per polars-arrow's impls, so we can call the trusted-len overload and
/// skip the per-row `Vec<&str>` + `Series::new` + `append_series` round-trip.
fn gen_typed_string_list_append(
    vec_access: &TokenStream,
    inner_opt: bool,
    builder_ident: &Ident,
) -> TokenStream {
    let map_expr = if inner_opt {
        quote! { .map(::std::option::Option::<::std::string::String>::as_deref) }
    } else {
        quote! { .map(|__df_derive_e| ::std::option::Option::Some(__df_derive_e.as_str())) }
    };
    quote! {{
        #builder_ident.append_trusted_len_iter((#vec_access).iter() #map_expr);
    }}
}

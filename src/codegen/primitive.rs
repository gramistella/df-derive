// Codegen for fields whose base type is a primitive scalar (numeric, bool,
// String, DateTime, Decimal) or a struct/generic routed through a
// `to_string`/`as_str` transform. The `[]` and `[Option]` shapes are
// served by the encoder IR in `super::encoder`; `Vec<...>`-bearing shapes
// flow through `generate_primitive_for_columnar_push` /
// `primitive_finishers_for_vec_anyvalues` here, sharing the wrapper
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

/// Per-row push tokens for primitive fields whose shape contains a `Vec<...>`
/// wrapper (or otherwise falls outside the encoder IR). The `[]` and `[Option]`
/// shapes are intercepted by the encoder IR in `strategy.rs`, so this function
/// is reached only for `Vec<...>`-bearing wrappers (and the few non-encoder
/// leaf carve-outs the encoder IR doesn't yet cover, currently bare ISize/USize).
pub fn generate_primitive_for_columnar_push(
    access: &TokenStream,
    base_type: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
    idx: usize,
    decimal128_encode_trait: &TokenStream,
) -> TokenStream {
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

/// Per-field decls for primitive shapes the encoder IR doesn't intercept:
/// `Vec<...>`-bearing wrappers (and the few non-encoder leaf carve-outs
/// the encoder IR doesn't yet cover, currently bare ISize/USize). The `[]`
/// and `[Option]` shapes are intercepted by the encoder IR in `strategy.rs`.
pub fn primitive_decls(
    wrappers: &[Wrapper],
    base_type: &BaseType,
    transform: Option<&PrimitiveTransform>,
    idx: usize,
) -> Vec<TokenStream> {
    let mut decls: Vec<TokenStream> = Vec::new();
    let opt = has_option(wrappers);
    let vec = has_vec(wrappers);

    let mapping = crate::codegen::type_registry::compute_mapping(base_type, transform, wrappers);
    let elem_rust_ty = mapping.rust_element_type;
    if vec {
        decls.push(gen_vec_list_builder_decl(
            base_type, transform, wrappers, idx,
        ));
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

/// List-builder declaration for the `Vec<…>` branch of `primitive_decls`.
/// Either:
/// - A concrete typed `ListPrimitiveChunkedBuilder<Native>` /
///   `ListStringChunkedBuilder` (when `typed_primitive_list_info` matches),
///   so the per-parent-row push site reaches `append_iter` /
///   `append_trusted_len_iter` inherently — bypassing `Series::new` + cast +
///   `append_series`'s `to_physical_repr` round-trip.
/// - A `Box<dyn ListBuilderTrait>` from `get_list_builder` for shapes the
///   typed route doesn't cover (nested `Vec<Vec<…>>`, struct-based leaves,
///   bool, etc.) or where the typed route was measured slower.
///
/// For nested-Vec shapes (`Vec<Vec<T>>`, …) `outer_list_inner_dtype` wraps
/// the leaf dtype in extra `List<>` layers so the builder's expected inner
/// dtype matches the per-row inner Series's runtime dtype.
fn gen_vec_list_builder_decl(
    base_type: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
    idx: usize,
) -> TokenStream {
    let lb_ident = PopulatorIdents::primitive_list_builder(idx);
    let pp = super::polars_paths::prelude();
    if let Some(info) = typed_primitive_list_info(base_type, transform, wrappers) {
        match &info.kind {
            BuilderKind::Primitive(p) => {
                let native = &p.native_type;
                let mapping_inner =
                    crate::codegen::type_registry::compute_mapping(base_type, transform, &[]);
                let inner_logical = mapping_inner.element_dtype;
                let constructor = if p.needs_values_type {
                    // Decimal: physical `Int128` differs from logical `Decimal(p, s)`;
                    // `new_with_values_type` carries both so the finished `ListChunked`
                    // matches `T::schema()` exactly without a post-finish cast.
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
                    // DateTime / bare numerics: physical matches the buffer's storage;
                    // the field dtype is the schema's logical dtype directly.
                    quote! {
                        #pp::ListPrimitiveChunkedBuilder::<#native>::new(
                            "".into(),
                            items.len(),
                            items.len() * 4,
                            #inner_logical,
                        )
                    }
                };
                quote! {
                    let mut #lb_ident: #pp::ListPrimitiveChunkedBuilder<#native> = #constructor;
                }
            }
            BuilderKind::String => {
                // `items.len() * 4` matches the `Box<dyn>` path's `get_list_builder`
                // string heuristic; the builder reallocates as needed regardless.
                quote! {
                    let mut #lb_ident: #pp::ListStringChunkedBuilder = #pp::ListStringChunkedBuilder::new(
                        "".into(),
                        items.len(),
                        items.len() * 4,
                    );
                }
            }
        }
    } else {
        let inner_dtype =
            crate::codegen::type_registry::outer_list_inner_dtype(base_type, transform, wrappers);
        let cab = super::polars_paths::chunked_array_builder();
        quote! {
            let mut #lb_ident: ::std::boxed::Box<dyn #pp::ListBuilderTrait> =
                #cab::get_list_builder(
                    &#inner_dtype,
                    items.len() * 4,
                    items.len(),
                    "".into(),
                );
        }
    }
}

/// Vec-anyvalues finisher for primitive shapes that aren't covered by the
/// encoder IR — `Vec<...>`-bearing wrappers (and the few non-encoder leaf
/// carve-outs the encoder IR doesn't yet cover, currently bare ISize/USize).
/// The `[]` and `[Option]` shapes are intercepted by the encoder IR in
/// `strategy.rs::gen_vec_values_finishers`.
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
    if transform.is_none()
        && let Some(n) = super::type_registry::numeric_info(base)
    {
        return primitive(n.builder_type, n.native, false, false);
    }
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

/// Eligible-shape probe for the bulk `Vec<Vec<T>>` numeric-primitive emit.
/// Returns `Some(numeric_info)` only when:
/// - Wrappers exactly `[Vec, Vec]` (no Option layers, no transform).
/// - Base is a bare numeric primitive
///   (`i8/i16/i32/i64/u8/u16/u32/u64/f32/f64`).
///
/// Other shapes — `Vec<Vec<Option<T>>>`, `Option`-wrapped variants, strings,
/// datetimes, decimals, bool, isize/usize, anything with a transform — keep
/// the existing typed-`ListBuilder` per-row push. Bool is excluded because
/// its validity bit semantics differ from numeric leaves and the all-non-null
/// case would still need a separate path; `numeric_info` enforces that.
fn nested_numeric_primitive(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> Option<super::type_registry::NumericInfo> {
    if !matches!(wrappers, [Wrapper::Vec, Wrapper::Vec]) || transform.is_some() {
        return None;
    }
    super::type_registry::numeric_info(base)
}

/// Returns `Some(emit)` for the columnar / vec-anyvalues bulk fast path on
/// `Vec<Vec<#native>>` over a bare numeric primitive base. `None` for any
/// other shape — caller falls back to the per-row decls/push/builders triple.
///
/// The block scans `items` once, flattening every inner element into a
/// single `Vec<Native>` while recording per-inner-vec offsets and per-outer-
/// vec offsets. It then constructs a `PrimitiveArray<Native>::from_vec(flat)`,
/// wraps it in a `LargeListArray` partitioned by the inner offsets, and
/// wraps that in a second `LargeListArray` partitioned by the outer offsets.
/// Finally it consumes the outer array via the in-scope free helper
/// `__df_derive_assemble_list_series_unchecked` (defined at the top of the
/// per-derive `const _: () = { ... };` scope) — same plumbing the nested-
/// struct bulk emitter uses to keep `unsafe` outside any `Self` impl method
/// and silence `clippy::unsafe_derive_deserialize`.
///
/// `parent_name`:
/// - `Some(name)` for the columnar context: the resulting Series is renamed
///   and pushed onto `columns`.
/// - `None` for the vec-anyvalues context: the resulting Series is wrapped
///   in `AnyValue::List(...)` and pushed onto `out_values`.
pub(super) fn try_gen_nested_primitive_vec_emit(
    pa_root: &TokenStream,
    access: &TokenStream,
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
    parent_name: Option<&str>,
) -> Option<TokenStream> {
    let info = nested_numeric_primitive(base, transform, wrappers)?;
    let pp = super::polars_paths::prelude();
    let native = &info.native;
    let leaf_dtype = &info.dtype;
    // The outer Series's logical inner dtype is `List<leaf>`; the
    // `__df_derive_assemble_list_series_unchecked` helper wraps it in another
    // `List<>` so the runtime dtype is `List<List<leaf>>` — same as the
    // typed-inner / boxed-outer path produces.
    let inner_logical_dtype = quote! {
        #pp::DataType::List(::std::boxed::Box::new(#leaf_dtype))
    };
    let series_block = quote! {{
        let mut __df_derive_total_leaves: usize = 0;
        let mut __df_derive_total_inners: usize = 0;
        for __df_derive_it in items {
            for __df_derive_inner in (&(#access)).iter() {
                __df_derive_total_leaves += __df_derive_inner.len();
                __df_derive_total_inners += 1;
            }
        }
        let mut __df_derive_flat: ::std::vec::Vec<#native> =
            ::std::vec::Vec::with_capacity(__df_derive_total_leaves);
        let mut __df_derive_inner_offsets: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(__df_derive_total_inners + 1);
        __df_derive_inner_offsets.push(0);
        let mut __df_derive_outer_offsets: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(items.len() + 1);
        __df_derive_outer_offsets.push(0);
        for __df_derive_it in items {
            for __df_derive_inner in (&(#access)).iter() {
                for __df_derive_e in __df_derive_inner.iter() {
                    __df_derive_flat.push(::std::clone::Clone::clone(__df_derive_e));
                }
                __df_derive_inner_offsets.push(__df_derive_flat.len() as i64);
            }
            __df_derive_outer_offsets.push((__df_derive_inner_offsets.len() - 1) as i64);
        }
        let __df_derive_inner_arr: #pa_root::array::PrimitiveArray<#native> =
            #pa_root::array::PrimitiveArray::<#native>::from_vec(__df_derive_flat);
        let __df_derive_inner_offsets_buf: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(__df_derive_inner_offsets)?;
        let __df_derive_inner_list_arr: #pp::LargeListArray = #pp::LargeListArray::new(
            #pp::LargeListArray::default_datatype(
                #pa_root::array::Array::dtype(&__df_derive_inner_arr).clone(),
            ),
            __df_derive_inner_offsets_buf,
            ::std::boxed::Box::new(__df_derive_inner_arr) as #pp::ArrayRef,
            ::std::option::Option::None,
        );
        let __df_derive_outer_offsets_buf: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(__df_derive_outer_offsets)?;
        let __df_derive_outer_list_arr: #pp::LargeListArray = #pp::LargeListArray::new(
            #pp::LargeListArray::default_datatype(
                #pa_root::array::Array::dtype(&__df_derive_inner_list_arr).clone(),
            ),
            __df_derive_outer_offsets_buf,
            ::std::boxed::Box::new(__df_derive_inner_list_arr) as #pp::ArrayRef,
            ::std::option::Option::None,
        );
        __df_derive_assemble_list_series_unchecked(
            __df_derive_outer_list_arr,
            #inner_logical_dtype,
        )
    }};
    let emit = parent_name.map_or_else(
        || {
            quote! {{
                let __df_derive_series: #pp::Series = #series_block;
                out_values.push(#pp::AnyValue::List(__df_derive_series));
            }}
        },
        |name| {
            quote! {{
                let __df_derive_series: #pp::Series = #series_block;
                let __df_derive_named = __df_derive_series.with_name(#name.into());
                columns.push(__df_derive_named.into());
            }}
        },
    );
    Some(emit)
}

/// Per-element emission info for the bulk `Vec<Option<T>>` direct
/// `LargeListArray` fast path: native rust storage type, leaf logical
/// `DataType` token, whether the per-element transform is fallible (embeds
/// `?` and may early-return a `PolarsError`), and whether the
/// `Decimal128Encode` trait must be imported anonymously into the emitted
/// block so dot-syntax method resolution finds `try_to_i128_mantissa` on the
/// `&Decimal` receiver.
///
/// Returns `Some(...)` only for shapes the fast path supports:
/// - Bare numeric primitive (`i8/i16/i32/i64/u8/u16/u32/u64/f32/f64`), no
///   transform → native = base, leaf = matching `DataType::Int*/UInt*/
///   Float*`, non-fallible, no trait import. The hot per-element store is
///   a plain copy (the value is already the storage type).
/// - `DateTime<Utc>` + `DateTimeToInt(unit)` → native `i64`, leaf
///   `DataType::Datetime(unit, None)`, fallible only for nanosecond unit
///   (`timestamp_nanos_opt` returns `None` outside chrono's representable
///   range), no trait import. The mapped expression dispatches per unit
///   (`timestamp_millis` / `timestamp_micros` / `timestamp_nanos_opt?`).
/// - `Decimal` + `DecimalToInt128 { precision, scale }` → native `i128`,
///   leaf `DataType::Decimal(precision, scale)`, always fallible (rescale
///   may overflow), trait import required. The mapped expression calls
///   `try_to_i128_mantissa(scale)` via the user-pluggable
///   `Decimal128Encode` trait.
///
/// Other shapes — `Vec<Option<bool>>`, `Vec<Option<String>>`,
/// `Option<Vec<Option<T>>>`, `Vec<Vec<Option<T>>>`, ISize/USize bases —
/// return `None` and keep the typed-`ListPrimitiveChunkedBuilder` /
/// `ListStringChunkedBuilder` / boxed-dyn per-row path.
pub(super) struct VecOptionLeafEmitInfo {
    pub native: TokenStream,
    pub leaf_dtype: TokenStream,
    pub needs_decimal_import: bool,
}

pub(super) fn vec_option_leaf_emit_info(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> Option<VecOptionLeafEmitInfo> {
    if !matches!(wrappers, [Wrapper::Vec, Wrapper::Option]) {
        return None;
    }
    let pp = super::polars_paths::prelude();
    if transform.is_none()
        && let Some(n) = super::type_registry::numeric_info(base)
    {
        return Some(VecOptionLeafEmitInfo {
            native: n.native,
            leaf_dtype: n.dtype,
            needs_decimal_import: false,
        });
    }
    match (base, transform) {
        (BaseType::DateTimeUtc, Some(PrimitiveTransform::DateTimeToInt(unit))) => {
            let unit_tokens = match unit {
                DateTimeUnit::Milliseconds => quote! { #pp::TimeUnit::Milliseconds },
                DateTimeUnit::Microseconds => quote! { #pp::TimeUnit::Microseconds },
                DateTimeUnit::Nanoseconds => quote! { #pp::TimeUnit::Nanoseconds },
            };
            Some(VecOptionLeafEmitInfo {
                native: quote! { i64 },
                leaf_dtype: quote! {
                    #pp::DataType::Datetime(#unit_tokens, ::std::option::Option::None)
                },
                needs_decimal_import: false,
            })
        }
        (BaseType::Decimal, Some(PrimitiveTransform::DecimalToInt128 { precision, scale })) => {
            let p = *precision as usize;
            let s = *scale as usize;
            Some(VecOptionLeafEmitInfo {
                native: quote! { i128 },
                leaf_dtype: quote! { #pp::DataType::Decimal(#p, #s) },
                needs_decimal_import: true,
            })
        }
        _ => None,
    }
}

/// Returns `Some(emit)` for the columnar / vec-anyvalues bulk fast path on
/// `Vec<Option<T>>` over the shapes enumerated by `vec_option_leaf_emit_info`
/// — bare numerics with no transform, `DateTime<Utc>` with `DateTimeToInt`,
/// and `Decimal` with `DecimalToInt128`. `None` for any other shape — caller
/// falls through to the typed `ListPrimitiveChunkedBuilder<Native>::append_iter`
/// per-row path.
///
/// The block scans `items` once, computing the total leaf count, then
/// allocates a flat `Vec<Native>` and a parallel `MutableBitmap` pre-filled
/// with `true` at that capacity. Each `Some(v)` row materializes the
/// transformed value (a copy for bare numerics; `timestamp_*` for `DateTime`;
/// `try_to_i128_mantissa` for `Decimal`) and pushes it (the validity bit is
/// already set); each `None` row pushes `<Native>::default()` as a placeholder
/// and flips the corresponding bit via the safe `MutableBitmap::set` (a
/// bounds-checked single-byte write — cheaper than the typed-builder's
/// per-element `MutablePrimitiveArray::push(Option<T>)` branching on the
/// validity-active flag and the discriminant).
///
/// For the fallible transforms (`DateTime<Utc>` at nanosecond precision, all
/// `Decimal` rescales) the per-element conversion embeds `?` directly inside
/// the loop body; failure short-circuits out of the surrounding emit block,
/// then out of the `Columnar::columnar_from_refs` /
/// `__df_derive_vec_to_inner_list_values` method via the same `?` propagation
/// the typed-builder fallible path uses (see
/// `gen_typed_primitive_list_append`'s `info.fallible` branch). Errors
/// preserve the same `polars_err!(ComputeError: ...)` text the typed path
/// produces because the per-element expression is built by
/// `generate_primitive_access_expr`, which routes through `map_primitive_expr`.
///
/// The finisher builds a `PrimitiveArray::<Native>::new(dtype, flat.into(),
/// Some(validity.into()))` — both conversions are zero-copy: `Vec<T> ->
/// Buffer<T>` and `MutableBitmap -> Option<Bitmap>` (the latter collapses
/// to `None` when no bits were unset, preserving the no-null fast path) —
/// wraps it in a `LargeListArray` partitioned by the per-row offsets, and
/// consumes the array via the in-scope free helper
/// `__df_derive_assemble_list_series_unchecked` so the resulting Series's
/// dtype is `List<leaf>` exactly (no post-finish cast). `leaf` is the
/// schema's logical dtype (`Datetime(unit, None)` / `Decimal(p, s)` / bare
/// numeric), which is mismatched physically against the inner i64/i128
/// `PrimitiveArray` for the transform-bearing arms — `unchecked` accepts
/// this exactly the way the typed `ListPrimitiveChunkedBuilder<Int128Type>`
/// + `Decimal(p, s)` logical dtype path does.
///
/// `parent_name`:
/// - `Some(name)` for the columnar context: the resulting Series is renamed
///   and pushed onto `columns`.
/// - `None` for the vec-anyvalues context: the resulting Series is wrapped
///   in `AnyValue::List(...)` and pushed onto `out_values`.
pub(super) fn try_gen_vec_option_numeric_emit(
    pa_root: &TokenStream,
    access: &TokenStream,
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
    parent_name: Option<&str>,
    decimal128_encode_trait: &TokenStream,
) -> Option<TokenStream> {
    let info = vec_option_leaf_emit_info(base, transform, wrappers)?;
    let pp = super::polars_paths::prelude();
    let VecOptionLeafEmitInfo {
        native,
        leaf_dtype,
        needs_decimal_import,
    } = &info;
    let elem_ident = quote! { __df_derive_v };
    // Per-element value materialization. For the no-transform numeric case we
    // keep the original `*v` copy: it's identical machine code to
    // `<T as Clone>::clone(&v)` for Copy primitives but compiles slightly
    // faster and matches the established baseline byte-for-byte. Transform-
    // bearing arms route through `generate_primitive_access_expr`, which
    // dispatches per-transform — `timestamp_millis()` / `timestamp_micros()`
    // / `timestamp_nanos_opt()?` for DateTime, and the `Decimal128Encode`-
    // backed `try_to_i128_mantissa(scale)?` for Decimal.
    let value_expr = if transform.is_some() {
        super::common::generate_primitive_access_expr(
            &elem_ident,
            transform,
            decimal128_encode_trait,
        )
    } else {
        quote! { *#elem_ident }
    };
    let import_trait = if *needs_decimal_import {
        quote! { use #decimal128_encode_trait as _; }
    } else {
        quote! {}
    };
    let series_block = quote! {{
        #import_trait
        let mut __df_derive_total: usize = 0;
        for __df_derive_it in items {
            __df_derive_total += (&(#access)).len();
        }
        let mut __df_derive_flat: ::std::vec::Vec<#native> =
            ::std::vec::Vec::with_capacity(__df_derive_total);
        let mut __df_derive_validity: #pa_root::bitmap::MutableBitmap = {
            let mut __df_derive_b =
                #pa_root::bitmap::MutableBitmap::with_capacity(__df_derive_total);
            __df_derive_b.extend_constant(__df_derive_total, true);
            __df_derive_b
        };
        let mut __df_derive_offsets: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(items.len() + 1);
        __df_derive_offsets.push(0);
        for __df_derive_it in items {
            for __df_derive_opt in (&(#access)).iter() {
                match __df_derive_opt {
                    ::std::option::Option::Some(__df_derive_v) => {
                        __df_derive_flat.push({ #value_expr });
                    }
                    ::std::option::Option::None => {
                        __df_derive_flat.push(<#native as ::std::default::Default>::default());
                        __df_derive_validity.set(__df_derive_flat.len() - 1, false);
                    }
                }
            }
            __df_derive_offsets.push(__df_derive_flat.len() as i64);
        }
        let __df_derive_arr: #pa_root::array::PrimitiveArray<#native> =
            #pa_root::array::PrimitiveArray::<#native>::new(
                <#native as #pa_root::types::NativeType>::PRIMITIVE.into(),
                __df_derive_flat.into(),
                ::std::convert::Into::<::std::option::Option<#pa_root::bitmap::Bitmap>>::into(
                    __df_derive_validity,
                ),
            );
        let __df_derive_offsets_buf: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(__df_derive_offsets)?;
        let __df_derive_list_arr: #pp::LargeListArray = #pp::LargeListArray::new(
            #pp::LargeListArray::default_datatype(
                #pa_root::array::Array::dtype(&__df_derive_arr).clone(),
            ),
            __df_derive_offsets_buf,
            ::std::boxed::Box::new(__df_derive_arr) as #pp::ArrayRef,
            ::std::option::Option::None,
        );
        __df_derive_assemble_list_series_unchecked(
            __df_derive_list_arr,
            #leaf_dtype,
        )
    }};
    let emit = parent_name.map_or_else(
        || {
            quote! {{
                let __df_derive_series: #pp::Series = #series_block;
                out_values.push(#pp::AnyValue::List(__df_derive_series));
            }}
        },
        |name| {
            quote! {{
                let __df_derive_series: #pp::Series = #series_block;
                let __df_derive_named = __df_derive_series.with_name(#name.into());
                columns.push(__df_derive_named.into());
            }}
        },
    );
    Some(emit)
}

/// Eligible-shape probe for the bulk `Vec<Option<String>>` emit. Returns
/// `true` only when wrappers are exactly `[Vec, Option]`, the base is
/// `String`, and there is no transform.
///
/// Excludes the transform-bearing siblings (`Vec<Option<DateTime>>` and
/// `Vec<Option<Decimal>>`), the no-Option sibling (`Vec<String>`), the
/// different-base siblings (`Vec<Option<bool>>` and `Vec<Option<numeric>>`),
/// and deeper nestings such as `Option<Vec<Option<String>>>` whose
/// flattening invariants this emitter doesn't model.
pub(super) const fn is_direct_vec_option_string_leaf(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> bool {
    transform.is_none()
        && matches!(base, BaseType::String)
        && matches!(wrappers, [Wrapper::Vec, Wrapper::Option])
}

/// Returns `Some(emit)` for the columnar / vec-anyvalues bulk fast path on
/// `Vec<Option<String>>` (wrappers exactly `[Vec, Option]`, base `String`,
/// no transform). `None` for any other shape — caller falls back to the
/// typed `ListStringChunkedBuilder::append_trusted_len_iter` per-row path.
///
/// The block scans `items` once, computing the total leaf count, then
/// allocates a `MutableBinaryViewArray<str>` for values and a parallel
/// `MutableBitmap` pre-filled with `true` at that capacity. Each `Some(s)`
/// row pushes the borrowed `&str` via `push_value_ignore_validity` — the
/// view array is built without an inner validity bitmap of its own, so the
/// `_ignore_validity` variant skips the per-element `if validity.is_some()`
/// branch `push_value` would otherwise do; each `None` row pushes an empty
/// placeholder string the same way and flips the corresponding bit on our
/// external validity bitmap via the safe `MutableBitmap::set` (a bounds-
/// checked single-byte write — cheaper than the typed builder's per-element
/// `MutableBinaryViewArray::push` branching on the validity-active flag).
/// The split-buffer layout also skips the per-parent-row `append_iter`
/// setup the typed `ListStringChunkedBuilder` does. The finisher freezes
/// the view buffer into a `Utf8ViewArray`, attaches the bitmap via
/// `with_validity` (`Option<Bitmap>::from(MutableBitmap)` collapses to
/// `None` when no bits were unset, preserving the no-null fast path),
/// wraps in a `LargeListArray` partitioned by the per-row offsets, and
/// consumes the array via the in-scope free helper
/// `__df_derive_assemble_list_series_unchecked` so the resulting Series's
/// dtype is `List<String>` exactly (no post-finish cast).
///
/// `parent_name`:
/// - `Some(name)` for the columnar context: the resulting Series is renamed
///   and pushed onto `columns`.
/// - `None` for the vec-anyvalues context: the resulting Series is wrapped
///   in `AnyValue::List(...)` and pushed onto `out_values`.
pub(super) fn try_gen_vec_option_string_emit(
    pa_root: &TokenStream,
    access: &TokenStream,
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
    parent_name: Option<&str>,
) -> Option<TokenStream> {
    if !is_direct_vec_option_string_leaf(base, transform, wrappers) {
        return None;
    }
    let pp = super::polars_paths::prelude();
    let leaf_dtype = quote! { #pp::DataType::String };
    let series_block = quote! {{
        let mut __df_derive_total: usize = 0;
        for __df_derive_it in items {
            __df_derive_total += (&(#access)).len();
        }
        let mut __df_derive_view_buf: #pa_root::array::MutableBinaryViewArray<str> =
            #pa_root::array::MutableBinaryViewArray::<str>::with_capacity(__df_derive_total);
        let mut __df_derive_validity: #pa_root::bitmap::MutableBitmap = {
            let mut __df_derive_b =
                #pa_root::bitmap::MutableBitmap::with_capacity(__df_derive_total);
            __df_derive_b.extend_constant(__df_derive_total, true);
            __df_derive_b
        };
        let mut __df_derive_offsets: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(items.len() + 1);
        __df_derive_offsets.push(0);
        let mut __df_derive_row_idx: usize = 0;
        for __df_derive_it in items {
            for __df_derive_opt in (&(#access)).iter() {
                match __df_derive_opt {
                    ::std::option::Option::Some(__df_derive_v) => {
                        __df_derive_view_buf
                            .push_value_ignore_validity(__df_derive_v.as_str());
                    }
                    ::std::option::Option::None => {
                        __df_derive_view_buf.push_value_ignore_validity("");
                        __df_derive_validity.set(__df_derive_row_idx, false);
                    }
                }
                __df_derive_row_idx += 1;
            }
            __df_derive_offsets.push(__df_derive_row_idx as i64);
        }
        let __df_derive_arr: #pa_root::array::Utf8ViewArray = __df_derive_view_buf
            .freeze()
            .with_validity(
                ::std::convert::Into::<::std::option::Option<#pa_root::bitmap::Bitmap>>::into(
                    __df_derive_validity,
                ),
            );
        let __df_derive_offsets_buf: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(__df_derive_offsets)?;
        let __df_derive_list_arr: #pp::LargeListArray = #pp::LargeListArray::new(
            #pp::LargeListArray::default_datatype(
                #pa_root::array::Array::dtype(&__df_derive_arr).clone(),
            ),
            __df_derive_offsets_buf,
            ::std::boxed::Box::new(__df_derive_arr) as #pp::ArrayRef,
            ::std::option::Option::None,
        );
        __df_derive_assemble_list_series_unchecked(
            __df_derive_list_arr,
            #leaf_dtype,
        )
    }};
    let emit = parent_name.map_or_else(
        || {
            quote! {{
                let __df_derive_series: #pp::Series = #series_block;
                out_values.push(#pp::AnyValue::List(__df_derive_series));
            }}
        },
        |name| {
            quote! {{
                let __df_derive_series: #pp::Series = #series_block;
                let __df_derive_named = __df_derive_series.with_name(#name.into());
                columns.push(__df_derive_named.into());
            }}
        },
    );
    Some(emit)
}

/// Eligible-shape probe for the bulk `Vec<Option<bool>>` emit. Returns
/// `true` only when wrappers are exactly `[Vec, Option]`, the base is
/// `Bool`, and there is no transform.
///
/// Excludes the numeric/string siblings (`Vec<Option<numeric>>`,
/// `Vec<Option<String>>`) which have their own emitters with different
/// leaf storage, the no-Option sibling (`Vec<bool>`), and any deeper
/// nesting (`Option<Vec<Option<bool>>>`, `Vec<Vec<Option<bool>>>`)
/// whose flattening invariants this emitter doesn't model.
pub(super) const fn is_direct_vec_option_bool_leaf(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> bool {
    transform.is_none()
        && matches!(base, BaseType::Bool)
        && matches!(wrappers, [Wrapper::Vec, Wrapper::Option])
}

/// Returns `Some(emit)` for the columnar / vec-anyvalues bulk fast path on
/// `Vec<Option<bool>>` (wrappers exactly `[Vec, Option]`, base `Bool`, no
/// transform). `None` for any other shape — caller falls back to the
/// boxed-dyn `ListBuilderTrait::append_series` per-row path.
///
/// Combines the flat-and-offsets pattern from the `Vec<bool>` emitter with
/// the inner-validity split-buffer from the `Option<bool>` direct path:
/// values live in a `MutableBitmap` (bool is bit-packed in arrow) pre-
/// filled to all-`false`, validity in a parallel `MutableBitmap` pre-
/// filled to all-`true`. Per-row push-equivalent only flips a single bit
/// for the rare arms — `Some(true)` flips a value bit, `None` flips a
/// validity bit, `Some(false)` is zero-work. Uses the safe
/// `MutableBitmap::set` (a bounds-checked single-byte write) so no
/// `unsafe` lands inside the user's `Columnar` impl method.
///
/// The finisher constructs a `BooleanArray::new(Boolean, values.into(),
/// validity.into())` (validity collapses to `None` if no bits are unset),
/// wraps it in a `LargeListArray` partitioned by the per-row offsets, and
/// consumes the array via the in-scope free helper
/// `__df_derive_assemble_list_series_unchecked` so the resulting Series's
/// dtype is `List<Boolean>` exactly (no post-finish cast).
///
/// `parent_name`:
/// - `Some(name)` for the columnar context: the resulting Series is renamed
///   and pushed onto `columns`.
/// - `None` for the vec-anyvalues context: the resulting Series is wrapped
///   in `AnyValue::List(...)` and pushed onto `out_values`.
pub(super) fn try_gen_vec_option_bool_emit(
    pa_root: &TokenStream,
    access: &TokenStream,
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
    parent_name: Option<&str>,
) -> Option<TokenStream> {
    if !is_direct_vec_option_bool_leaf(base, transform, wrappers) {
        return None;
    }
    let pp = super::polars_paths::prelude();
    let leaf_dtype = quote! { #pp::DataType::Boolean };
    let series_block = quote! {{
        let mut __df_derive_total: usize = 0;
        for __df_derive_it in items {
            __df_derive_total += (&(#access)).len();
        }
        let mut __df_derive_values: #pa_root::bitmap::MutableBitmap = {
            let mut __df_derive_b =
                #pa_root::bitmap::MutableBitmap::with_capacity(__df_derive_total);
            __df_derive_b.extend_constant(__df_derive_total, false);
            __df_derive_b
        };
        let mut __df_derive_validity: #pa_root::bitmap::MutableBitmap = {
            let mut __df_derive_b =
                #pa_root::bitmap::MutableBitmap::with_capacity(__df_derive_total);
            __df_derive_b.extend_constant(__df_derive_total, true);
            __df_derive_b
        };
        let mut __df_derive_offsets: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(items.len() + 1);
        __df_derive_offsets.push(0);
        let mut __df_derive_row_idx: usize = 0;
        for __df_derive_it in items {
            for __df_derive_opt in (&(#access)).iter() {
                match __df_derive_opt {
                    ::std::option::Option::Some(true) => {
                        __df_derive_values.set(__df_derive_row_idx, true);
                    }
                    ::std::option::Option::Some(false) => {}
                    ::std::option::Option::None => {
                        __df_derive_validity.set(__df_derive_row_idx, false);
                    }
                }
                __df_derive_row_idx += 1;
            }
            __df_derive_offsets.push(__df_derive_row_idx as i64);
        }
        let __df_derive_arr: #pa_root::array::BooleanArray =
            #pa_root::array::BooleanArray::new(
                #pa_root::datatypes::ArrowDataType::Boolean,
                ::std::convert::Into::<#pa_root::bitmap::Bitmap>::into(__df_derive_values),
                ::std::convert::Into::<::std::option::Option<#pa_root::bitmap::Bitmap>>::into(
                    __df_derive_validity,
                ),
            );
        let __df_derive_offsets_buf: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(__df_derive_offsets)?;
        let __df_derive_list_arr: #pp::LargeListArray = #pp::LargeListArray::new(
            #pp::LargeListArray::default_datatype(
                #pa_root::array::Array::dtype(&__df_derive_arr).clone(),
            ),
            __df_derive_offsets_buf,
            ::std::boxed::Box::new(__df_derive_arr) as #pp::ArrayRef,
            ::std::option::Option::None,
        );
        __df_derive_assemble_list_series_unchecked(
            __df_derive_list_arr,
            #leaf_dtype,
        )
    }};
    let emit = parent_name.map_or_else(
        || {
            quote! {{
                let __df_derive_series: #pp::Series = #series_block;
                out_values.push(#pp::AnyValue::List(__df_derive_series));
            }}
        },
        |name| {
            quote! {{
                let __df_derive_series: #pp::Series = #series_block;
                let __df_derive_named = __df_derive_series.with_name(#name.into());
                columns.push(__df_derive_named.into());
            }}
        },
    );
    Some(emit)
}

/// Returns `Some(emit)` for the columnar / vec-anyvalues bulk fast path on
/// `Vec<bool>` (wrappers exactly `[Vec]`, base `Bool`, no transform). `None`
/// for any other shape — caller falls back to the boxed-dyn
/// `ListBuilderTrait::append_series` per-row path.
///
/// Sibling shapes (`Option<Vec<bool>>`, `Vec<Vec<bool>>`) keep the
/// existing path: the `[Option, Vec]` case would need a validity bitmap
/// that this emitter intentionally omits, and the numeric `Vec<Vec<T>>`
/// fast path explicitly excludes Bool because of the validity-bit cost
/// it would re-introduce there. `Vec<Option<bool>>` is handled separately
/// by `try_gen_vec_option_bool_emit` via a split-buffer pattern (values
/// in a bit-packed `MutableBitmap`, validity in a parallel `MutableBitmap`).
///
/// The block scans `items` once, extending a flat `Vec<bool>` from each
/// row's inner Vec while recording per-row offsets. It then constructs a
/// `BooleanArray::from_slice(&flat)` (no validity — bool leaves are
/// non-null), wraps it in a `LargeListArray` partitioned by the offsets,
/// and consumes the array via the in-scope free helper
/// `__df_derive_assemble_list_series_unchecked` — same plumbing the
/// `Vec<Vec<numeric>>` and nested-struct bulk emitters use to keep
/// `unsafe` outside any `Self` impl method.
///
/// `parent_name`:
/// - `Some(name)` for the columnar context: the resulting Series is renamed
///   and pushed onto `columns`.
/// - `None` for the vec-anyvalues context: the resulting Series is wrapped
///   in `AnyValue::List(...)` and pushed onto `out_values`.
pub(super) fn try_gen_vec_bool_emit(
    pa_root: &TokenStream,
    access: &TokenStream,
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
    parent_name: Option<&str>,
) -> Option<TokenStream> {
    if !matches!(wrappers, [Wrapper::Vec]) || transform.is_some() || !matches!(base, BaseType::Bool)
    {
        return None;
    }
    let pp = super::polars_paths::prelude();
    let leaf_dtype = quote! { #pp::DataType::Boolean };
    let series_block = quote! {{
        let mut __df_derive_total: usize = 0;
        for __df_derive_it in items {
            __df_derive_total += (&(#access)).len();
        }
        let mut __df_derive_flat: ::std::vec::Vec<bool> =
            ::std::vec::Vec::with_capacity(__df_derive_total);
        let mut __df_derive_offsets: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(items.len() + 1);
        __df_derive_offsets.push(0);
        for __df_derive_it in items {
            __df_derive_flat.extend((&(#access)).iter().copied());
            __df_derive_offsets.push(__df_derive_flat.len() as i64);
        }
        let __df_derive_bool_arr: #pa_root::array::BooleanArray =
            #pa_root::array::BooleanArray::from_slice(&__df_derive_flat);
        let __df_derive_offsets_buf: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(__df_derive_offsets)?;
        let __df_derive_list_arr: #pp::LargeListArray = #pp::LargeListArray::new(
            #pp::LargeListArray::default_datatype(
                #pa_root::array::Array::dtype(&__df_derive_bool_arr).clone(),
            ),
            __df_derive_offsets_buf,
            ::std::boxed::Box::new(__df_derive_bool_arr) as #pp::ArrayRef,
            ::std::option::Option::None,
        );
        __df_derive_assemble_list_series_unchecked(
            __df_derive_list_arr,
            #leaf_dtype,
        )
    }};
    let emit = parent_name.map_or_else(
        || {
            quote! {{
                let __df_derive_series: #pp::Series = #series_block;
                out_values.push(#pp::AnyValue::List(__df_derive_series));
            }}
        },
        |name| {
            quote! {{
                let __df_derive_series: #pp::Series = #series_block;
                let __df_derive_named = __df_derive_series.with_name(#name.into());
                columns.push(__df_derive_named.into());
            }}
        },
    );
    Some(emit)
}

/// Eligible-shape probe for the bulk `Vec<Vec<Option<T>>>` numeric-primitive
/// emit. Returns `true` only when wrappers are exactly `[Vec, Vec, Option]`,
/// the base is a bare numeric primitive
/// (`i8/i16/i32/i64/u8/u16/u32/u64/f32/f64`), and there is no transform.
///
/// Excludes:
/// - `Vec<Vec<Option<bool>>>` / `Vec<Vec<Option<String>>>` — different leaf
///   storage; bool would also need a per-element validity bit on the leaf.
/// - `Vec<Vec<T>>` (no Option) — handled by `try_gen_nested_primitive_vec_emit`.
/// - `Vec<Option<T>>` — handled by `try_gen_vec_option_numeric_emit`.
/// - Any extra `Option` layer (e.g. `Option<Vec<Vec<Option<T>>>>`,
///   `Vec<Option<Vec<Option<T>>>>`) — flattening invariants this emitter
///   relies on don't model an outer/middle Option layer.
/// - Transforms (`DateTime`, `Decimal`, `as_str`, `as_string`).
pub(super) const fn is_direct_vec_vec_option_numeric_leaf(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> bool {
    transform.is_none()
        && matches!(wrappers, [Wrapper::Vec, Wrapper::Vec, Wrapper::Option])
        && super::type_registry::is_numeric_base(base)
}

/// Returns `Some(emit)` for the columnar / vec-anyvalues bulk fast path on
/// `Vec<Vec<Option<T>>>` over a bare numeric primitive base (no transform).
/// `None` for any other shape — caller falls through to the slower
/// per-row typed-`ListBuilder` path.
///
/// Combines the two-level offset stacking from the `Vec<Vec<numeric>>`
/// emitter (`try_gen_nested_primitive_vec_emit`) with the inner-Option
/// split-buffer from the `Vec<Option<numeric>>` emitter
/// (`try_gen_vec_option_numeric_emit`). The block scans `items` once to
/// compute total inner-vec count and total leaf count, then allocates a
/// flat `Vec<Native>` and a parallel `MutableBitmap` pre-filled with `true`
/// at that capacity. Each `Some(v)` leaf pushes the value and advances the
/// bitmap index; each `None` leaf pushes `<Native>::default()` as a
/// placeholder and flips the corresponding bit via the safe
/// `MutableBitmap::set`. Per-inner-vec offsets and per-outer-vec offsets
/// are accumulated alongside.
///
/// The finisher builds a `PrimitiveArray::<Native>::new(dtype, flat.into(),
/// Some(validity.into()))`, wraps it in a `LargeListArray` partitioned by
/// the inner offsets, wraps that in a second `LargeListArray` partitioned
/// by the outer offsets, and consumes the result via the in-scope free
/// helper `__df_derive_assemble_list_series_unchecked` so the resulting
/// Series's dtype is `List<List<leaf>>` exactly (no post-finish cast).
///
/// `parent_name`:
/// - `Some(name)` for the columnar context: the resulting Series is renamed
///   and pushed onto `columns`.
/// - `None` for the vec-anyvalues context: the resulting Series is wrapped
///   in `AnyValue::List(...)` and pushed onto `out_values`.
pub(super) fn try_gen_vec_vec_option_numeric_emit(
    pa_root: &TokenStream,
    access: &TokenStream,
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
    parent_name: Option<&str>,
) -> Option<TokenStream> {
    if !is_direct_vec_vec_option_numeric_leaf(base, transform, wrappers) {
        return None;
    }
    let pp = super::polars_paths::prelude();
    let info = super::type_registry::numeric_info(base)
        .expect("is_direct_vec_vec_option_numeric_leaf gates on is_numeric_base");
    let native = info.native;
    let leaf_dtype = info.dtype;
    let inner_logical_dtype = quote! {
        #pp::DataType::List(::std::boxed::Box::new(#leaf_dtype))
    };
    let series_block = quote! {{
        let mut __df_derive_total_leaves: usize = 0;
        let mut __df_derive_total_inners: usize = 0;
        for __df_derive_it in items {
            for __df_derive_inner in (&(#access)).iter() {
                __df_derive_total_leaves += __df_derive_inner.len();
                __df_derive_total_inners += 1;
            }
        }
        let mut __df_derive_flat: ::std::vec::Vec<#native> =
            ::std::vec::Vec::with_capacity(__df_derive_total_leaves);
        let mut __df_derive_validity: #pa_root::bitmap::MutableBitmap = {
            let mut __df_derive_b =
                #pa_root::bitmap::MutableBitmap::with_capacity(__df_derive_total_leaves);
            __df_derive_b.extend_constant(__df_derive_total_leaves, true);
            __df_derive_b
        };
        let mut __df_derive_inner_offsets: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(__df_derive_total_inners + 1);
        __df_derive_inner_offsets.push(0);
        let mut __df_derive_outer_offsets: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(items.len() + 1);
        __df_derive_outer_offsets.push(0);
        for __df_derive_it in items {
            for __df_derive_inner in (&(#access)).iter() {
                for __df_derive_opt in __df_derive_inner.iter() {
                    match __df_derive_opt {
                        ::std::option::Option::Some(__df_derive_v) => {
                            __df_derive_flat.push(*__df_derive_v);
                        }
                        ::std::option::Option::None => {
                            __df_derive_flat.push(<#native as ::std::default::Default>::default());
                            __df_derive_validity.set(__df_derive_flat.len() - 1, false);
                        }
                    }
                }
                __df_derive_inner_offsets.push(__df_derive_flat.len() as i64);
            }
            __df_derive_outer_offsets.push((__df_derive_inner_offsets.len() - 1) as i64);
        }
        let __df_derive_leaf_arr: #pa_root::array::PrimitiveArray<#native> =
            #pa_root::array::PrimitiveArray::<#native>::new(
                <#native as #pa_root::types::NativeType>::PRIMITIVE.into(),
                __df_derive_flat.into(),
                ::std::convert::Into::<::std::option::Option<#pa_root::bitmap::Bitmap>>::into(
                    __df_derive_validity,
                ),
            );
        let __df_derive_inner_offsets_buf: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(__df_derive_inner_offsets)?;
        let __df_derive_inner_list_arr: #pp::LargeListArray = #pp::LargeListArray::new(
            #pp::LargeListArray::default_datatype(
                #pa_root::array::Array::dtype(&__df_derive_leaf_arr).clone(),
            ),
            __df_derive_inner_offsets_buf,
            ::std::boxed::Box::new(__df_derive_leaf_arr) as #pp::ArrayRef,
            ::std::option::Option::None,
        );
        let __df_derive_outer_offsets_buf: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(__df_derive_outer_offsets)?;
        let __df_derive_outer_list_arr: #pp::LargeListArray = #pp::LargeListArray::new(
            #pp::LargeListArray::default_datatype(
                #pa_root::array::Array::dtype(&__df_derive_inner_list_arr).clone(),
            ),
            __df_derive_outer_offsets_buf,
            ::std::boxed::Box::new(__df_derive_inner_list_arr) as #pp::ArrayRef,
            ::std::option::Option::None,
        );
        __df_derive_assemble_list_series_unchecked(
            __df_derive_outer_list_arr,
            #inner_logical_dtype,
        )
    }};
    let emit = parent_name.map_or_else(
        || {
            quote! {{
                let __df_derive_series: #pp::Series = #series_block;
                out_values.push(#pp::AnyValue::List(__df_derive_series));
            }}
        },
        |name| {
            quote! {{
                let __df_derive_series: #pp::Series = #series_block;
                let __df_derive_named = __df_derive_series.with_name(#name.into());
                columns.push(__df_derive_named.into());
            }}
        },
    );
    Some(emit)
}

/// Eligible-shape probe for the bulk `Vec<Vec<String>>` emit. Returns `true`
/// only when wrappers are exactly `[Vec, Vec]`, the base is `String`, and
/// there is no transform.
///
/// Excludes:
/// - `Vec<Vec<Option<String>>>` — needs a leaf-level validity bitmap.
/// - `Vec<Vec<numeric>>` — handled by `try_gen_nested_primitive_vec_emit`.
/// - `Vec<String>` — handled by `is_direct_view_string_leaf` finishers.
/// - Any extra `Option` layer (e.g. `Option<Vec<Vec<String>>>`) — flattening
///   invariants this emitter relies on don't model an outer Option layer.
/// - Transforms (`as_str`, `as_string`).
pub(super) const fn is_direct_vec_vec_string_leaf(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> bool {
    transform.is_none()
        && matches!(base, BaseType::String)
        && matches!(wrappers, [Wrapper::Vec, Wrapper::Vec])
}

/// Returns `Some(emit)` for the columnar / vec-anyvalues bulk fast path on
/// `Vec<Vec<String>>` (wrappers exactly `[Vec, Vec]`, base `String`, no
/// transform). `None` for any other shape — caller falls through to the
/// slower per-row typed `ListBuilder` path.
///
/// Combines the two-level offset stacking from `Vec<Vec<numeric>>`
/// (`try_gen_nested_primitive_vec_emit`) with the no-validity
/// `MutableBinaryViewArray<str>` accumulation from the bare-`String` direct
/// path (commit `2d9eeab`). The block scans `items` once to compute total
/// inner-vec count and total leaf count, then allocates a
/// `MutableBinaryViewArray<str>` at that capacity. Each leaf string is
/// pushed via `push_value_ignore_validity` — the view array is built
/// without an inner validity bitmap of its own, so the `_ignore_validity`
/// variant skips the per-element `if validity.is_some()` branch
/// `push_value` would otherwise do. Per-inner-vec offsets and per-outer-vec
/// offsets are accumulated alongside; no validity bitmap is needed because
/// `Vec<Vec<String>>` has no Option layer.
///
/// The finisher freezes the view buffer into a `Utf8ViewArray`, wraps it in
/// a `LargeListArray` partitioned by the inner offsets, wraps that in a
/// second `LargeListArray` partitioned by the outer offsets, and consumes
/// the result via the in-scope free helper
/// `__df_derive_assemble_list_series_unchecked` so the resulting Series's
/// dtype is `List<List<String>>` exactly (no post-finish cast).
///
/// `parent_name`:
/// - `Some(name)` for the columnar context: the resulting Series is renamed
///   and pushed onto `columns`.
/// - `None` for the vec-anyvalues context: the resulting Series is wrapped
///   in `AnyValue::List(...)` and pushed onto `out_values`.
pub(super) fn try_gen_vec_vec_string_emit(
    pa_root: &TokenStream,
    access: &TokenStream,
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
    parent_name: Option<&str>,
) -> Option<TokenStream> {
    if !is_direct_vec_vec_string_leaf(base, transform, wrappers) {
        return None;
    }
    let pp = super::polars_paths::prelude();
    let leaf_dtype = quote! { #pp::DataType::String };
    let inner_logical_dtype = quote! {
        #pp::DataType::List(::std::boxed::Box::new(#leaf_dtype))
    };
    let series_block = quote! {{
        let mut __df_derive_total_leaves: usize = 0;
        let mut __df_derive_total_inners: usize = 0;
        for __df_derive_it in items {
            for __df_derive_inner in (&(#access)).iter() {
                __df_derive_total_leaves += __df_derive_inner.len();
                __df_derive_total_inners += 1;
            }
        }
        let mut __df_derive_view_buf: #pa_root::array::MutableBinaryViewArray<str> =
            #pa_root::array::MutableBinaryViewArray::<str>::with_capacity(__df_derive_total_leaves);
        let mut __df_derive_inner_offsets: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(__df_derive_total_inners + 1);
        __df_derive_inner_offsets.push(0);
        let mut __df_derive_outer_offsets: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(items.len() + 1);
        __df_derive_outer_offsets.push(0);
        for __df_derive_it in items {
            for __df_derive_inner in (&(#access)).iter() {
                for __df_derive_s in __df_derive_inner.iter() {
                    __df_derive_view_buf
                        .push_value_ignore_validity(__df_derive_s.as_str());
                }
                __df_derive_inner_offsets.push(__df_derive_view_buf.len() as i64);
            }
            __df_derive_outer_offsets.push((__df_derive_inner_offsets.len() - 1) as i64);
        }
        let __df_derive_leaf_arr: #pa_root::array::Utf8ViewArray =
            __df_derive_view_buf.freeze();
        let __df_derive_inner_offsets_buf: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(__df_derive_inner_offsets)?;
        let __df_derive_inner_list_arr: #pp::LargeListArray = #pp::LargeListArray::new(
            #pp::LargeListArray::default_datatype(
                #pa_root::array::Array::dtype(&__df_derive_leaf_arr).clone(),
            ),
            __df_derive_inner_offsets_buf,
            ::std::boxed::Box::new(__df_derive_leaf_arr) as #pp::ArrayRef,
            ::std::option::Option::None,
        );
        let __df_derive_outer_offsets_buf: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(__df_derive_outer_offsets)?;
        let __df_derive_outer_list_arr: #pp::LargeListArray = #pp::LargeListArray::new(
            #pp::LargeListArray::default_datatype(
                #pa_root::array::Array::dtype(&__df_derive_inner_list_arr).clone(),
            ),
            __df_derive_outer_offsets_buf,
            ::std::boxed::Box::new(__df_derive_inner_list_arr) as #pp::ArrayRef,
            ::std::option::Option::None,
        );
        __df_derive_assemble_list_series_unchecked(
            __df_derive_outer_list_arr,
            #inner_logical_dtype,
        )
    }};
    let emit = parent_name.map_or_else(
        || {
            quote! {{
                let __df_derive_series: #pp::Series = #series_block;
                out_values.push(#pp::AnyValue::List(__df_derive_series));
            }}
        },
        |name| {
            quote! {{
                let __df_derive_series: #pp::Series = #series_block;
                let __df_derive_named = __df_derive_series.with_name(#name.into());
                columns.push(__df_derive_named.into());
            }}
        },
    );
    Some(emit)
}

/// Eligible-shape probe for the bulk `Vec<Vec<bool>>` emit. Returns `true`
/// only when wrappers are exactly `[Vec, Vec]`, the base is `Bool`, and there
/// is no transform.
///
/// Excludes:
/// - `Vec<Vec<Option<bool>>>` — would need a leaf-level validity bitmap.
/// - `Vec<bool>` — handled by `try_gen_vec_bool_emit`.
/// - `Vec<Option<bool>>` — handled by `try_gen_vec_option_bool_emit`.
/// - `Vec<Vec<numeric>>` — handled by `try_gen_nested_primitive_vec_emit`.
/// - `Vec<Vec<String>>` — handled by `try_gen_vec_vec_string_emit`.
/// - Any extra `Option` layer (e.g. `Option<Vec<Vec<bool>>>`) — flattening
///   invariants this emitter relies on don't model an outer Option layer.
pub(super) const fn is_direct_vec_vec_bool_leaf(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> bool {
    transform.is_none()
        && matches!(base, BaseType::Bool)
        && matches!(wrappers, [Wrapper::Vec, Wrapper::Vec])
}

/// Returns `Some(emit)` for the columnar / vec-anyvalues bulk fast path on
/// `Vec<Vec<bool>>` (wrappers exactly `[Vec, Vec]`, base `Bool`, no
/// transform). `None` for any other shape — caller falls through to the
/// boxed-dyn `ListBuilderTrait::append_series` per-row path.
///
/// Combines the two-level offset stacking from `Vec<Vec<numeric>>`
/// (`try_gen_nested_primitive_vec_emit`) with the bit-packed `MutableBitmap`
/// values buffer from the `Vec<bool>` direct path. The block scans `items`
/// once to compute total inner-vec count and total leaf count, then allocates
/// a single `MutableBitmap` of values pre-filled to all-`false`. Each `true`
/// leaf flips a single bit via the safe `MutableBitmap::set` (a bounds-
/// checked single-byte write), each `false` leaf is zero-work. Per-inner-vec
/// offsets and per-outer-vec offsets are accumulated alongside; no validity
/// bitmap is needed because `Vec<Vec<bool>>` has no Option layer.
///
/// The finisher constructs a `BooleanArray::new(Boolean, values.into(),
/// None)`, wraps it in a `LargeListArray` partitioned by the inner offsets,
/// wraps that in a second `LargeListArray` partitioned by the outer offsets,
/// and consumes the result via the in-scope free helper
/// `__df_derive_assemble_list_series_unchecked` so the resulting Series's
/// dtype is `List<List<Boolean>>` exactly (no post-finish cast).
///
/// `parent_name`:
/// - `Some(name)` for the columnar context: the resulting Series is renamed
///   and pushed onto `columns`.
/// - `None` for the vec-anyvalues context: the resulting Series is wrapped
///   in `AnyValue::List(...)` and pushed onto `out_values`.
pub(super) fn try_gen_vec_vec_bool_emit(
    pa_root: &TokenStream,
    access: &TokenStream,
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
    parent_name: Option<&str>,
) -> Option<TokenStream> {
    if !is_direct_vec_vec_bool_leaf(base, transform, wrappers) {
        return None;
    }
    let pp = super::polars_paths::prelude();
    let leaf_dtype = quote! { #pp::DataType::Boolean };
    let inner_logical_dtype = quote! {
        #pp::DataType::List(::std::boxed::Box::new(#leaf_dtype))
    };
    let series_block = quote! {{
        let mut __df_derive_total_leaves: usize = 0;
        let mut __df_derive_total_inners: usize = 0;
        for __df_derive_it in items {
            for __df_derive_inner in (&(#access)).iter() {
                __df_derive_total_leaves += __df_derive_inner.len();
                __df_derive_total_inners += 1;
            }
        }
        let mut __df_derive_values: #pa_root::bitmap::MutableBitmap = {
            let mut __df_derive_b =
                #pa_root::bitmap::MutableBitmap::with_capacity(__df_derive_total_leaves);
            __df_derive_b.extend_constant(__df_derive_total_leaves, false);
            __df_derive_b
        };
        let mut __df_derive_inner_offsets: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(__df_derive_total_inners + 1);
        __df_derive_inner_offsets.push(0);
        let mut __df_derive_outer_offsets: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(items.len() + 1);
        __df_derive_outer_offsets.push(0);
        let mut __df_derive_leaf_idx: usize = 0;
        for __df_derive_it in items {
            for __df_derive_inner in (&(#access)).iter() {
                for __df_derive_e in __df_derive_inner.iter() {
                    if *__df_derive_e {
                        __df_derive_values.set(__df_derive_leaf_idx, true);
                    }
                    __df_derive_leaf_idx += 1;
                }
                __df_derive_inner_offsets.push(__df_derive_leaf_idx as i64);
            }
            __df_derive_outer_offsets.push((__df_derive_inner_offsets.len() - 1) as i64);
        }
        let __df_derive_leaf_arr: #pa_root::array::BooleanArray =
            #pa_root::array::BooleanArray::new(
                #pa_root::datatypes::ArrowDataType::Boolean,
                ::std::convert::Into::<#pa_root::bitmap::Bitmap>::into(__df_derive_values),
                ::std::option::Option::None,
            );
        let __df_derive_inner_offsets_buf: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(__df_derive_inner_offsets)?;
        let __df_derive_inner_list_arr: #pp::LargeListArray = #pp::LargeListArray::new(
            #pp::LargeListArray::default_datatype(
                #pa_root::array::Array::dtype(&__df_derive_leaf_arr).clone(),
            ),
            __df_derive_inner_offsets_buf,
            ::std::boxed::Box::new(__df_derive_leaf_arr) as #pp::ArrayRef,
            ::std::option::Option::None,
        );
        let __df_derive_outer_offsets_buf: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(__df_derive_outer_offsets)?;
        let __df_derive_outer_list_arr: #pp::LargeListArray = #pp::LargeListArray::new(
            #pp::LargeListArray::default_datatype(
                #pa_root::array::Array::dtype(&__df_derive_inner_list_arr).clone(),
            ),
            __df_derive_outer_offsets_buf,
            ::std::boxed::Box::new(__df_derive_inner_list_arr) as #pp::ArrayRef,
            ::std::option::Option::None,
        );
        __df_derive_assemble_list_series_unchecked(
            __df_derive_outer_list_arr,
            #inner_logical_dtype,
        )
    }};
    let emit = parent_name.map_or_else(
        || {
            quote! {{
                let __df_derive_series: #pp::Series = #series_block;
                out_values.push(#pp::AnyValue::List(__df_derive_series));
            }}
        },
        |name| {
            quote! {{
                let __df_derive_series: #pp::Series = #series_block;
                let __df_derive_named = __df_derive_series.with_name(#name.into());
                columns.push(__df_derive_named.into());
            }}
        },
    );
    Some(emit)
}

/// Eligible-shape probe for the bulk `Vec<Vec<Option<String>>>` emit.
/// Returns `true` only when wrappers are exactly `[Vec, Vec, Option]`, the
/// base is `String`, and there is no transform.
///
/// Excludes:
/// - `Vec<Vec<String>>` — handled by `try_gen_vec_vec_string_emit`.
/// - `Vec<Option<String>>` — handled by `try_gen_vec_option_string_emit`.
/// - `Vec<Vec<Option<numeric>>>` — handled by
///   `try_gen_vec_vec_option_numeric_emit`.
/// - Any extra outer `Option` layer (e.g. `Option<Vec<Vec<Option<String>>>>`)
///   — flattening invariants this emitter relies on don't model an outer
///   Option layer.
/// - Transforms (`as_str`, `as_string`).
pub(super) const fn is_direct_vec_vec_option_string_leaf(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> bool {
    transform.is_none()
        && matches!(base, BaseType::String)
        && matches!(wrappers, [Wrapper::Vec, Wrapper::Vec, Wrapper::Option])
}

/// Returns `Some(emit)` for the columnar / vec-anyvalues bulk fast path on
/// `Vec<Vec<Option<String>>>` (wrappers exactly `[Vec, Vec, Option]`, base
/// `String`, no transform). `None` for any other shape — caller falls
/// through to the slower per-row typed `ListBuilder` path.
///
/// Combines the two-level offset stacking from `Vec<Vec<String>>`
/// (`try_gen_vec_vec_string_emit`) with the inner-validity split-buffer
/// pattern from `Vec<Option<String>>` (`try_gen_vec_option_string_emit`).
/// The block scans `items` once to compute total inner-vec count and total
/// leaf count, then allocates a `MutableBinaryViewArray<str>` for values
/// and a parallel `MutableBitmap` pre-filled with `true` at that capacity.
/// Each `Some(s)` leaf pushes the borrowed `&str` via
/// `push_value_ignore_validity`; each `None` leaf pushes an empty
/// placeholder string the same way and flips the corresponding bit on the
/// external validity bitmap via the safe `MutableBitmap::set`. Per-inner-vec
/// offsets and per-outer-vec offsets are accumulated alongside.
///
/// The finisher freezes the view buffer into a `Utf8ViewArray`, attaches the
/// bitmap via `with_validity` (`Option<Bitmap>::from(MutableBitmap)`
/// collapses to `None` when no bits were unset, preserving the no-null fast
/// path), wraps it in a `LargeListArray` partitioned by the inner offsets,
/// wraps that in a second `LargeListArray` partitioned by the outer offsets,
/// and consumes the result via the in-scope free helper
/// `__df_derive_assemble_list_series_unchecked` so the resulting Series's
/// dtype is `List<List<String>>` exactly (no post-finish cast).
///
/// `parent_name`:
/// - `Some(name)` for the columnar context: the resulting Series is renamed
///   and pushed onto `columns`.
/// - `None` for the vec-anyvalues context: the resulting Series is wrapped
///   in `AnyValue::List(...)` and pushed onto `out_values`.
pub(super) fn try_gen_vec_vec_option_string_emit(
    pa_root: &TokenStream,
    access: &TokenStream,
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
    parent_name: Option<&str>,
) -> Option<TokenStream> {
    if !is_direct_vec_vec_option_string_leaf(base, transform, wrappers) {
        return None;
    }
    let pp = super::polars_paths::prelude();
    let leaf_dtype = quote! { #pp::DataType::String };
    let inner_logical_dtype = quote! {
        #pp::DataType::List(::std::boxed::Box::new(#leaf_dtype))
    };
    let series_block = quote! {{
        let mut __df_derive_total_leaves: usize = 0;
        let mut __df_derive_total_inners: usize = 0;
        for __df_derive_it in items {
            for __df_derive_inner in (&(#access)).iter() {
                __df_derive_total_leaves += __df_derive_inner.len();
                __df_derive_total_inners += 1;
            }
        }
        let mut __df_derive_view_buf: #pa_root::array::MutableBinaryViewArray<str> =
            #pa_root::array::MutableBinaryViewArray::<str>::with_capacity(__df_derive_total_leaves);
        let mut __df_derive_validity: #pa_root::bitmap::MutableBitmap = {
            let mut __df_derive_b =
                #pa_root::bitmap::MutableBitmap::with_capacity(__df_derive_total_leaves);
            __df_derive_b.extend_constant(__df_derive_total_leaves, true);
            __df_derive_b
        };
        let mut __df_derive_inner_offsets: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(__df_derive_total_inners + 1);
        __df_derive_inner_offsets.push(0);
        let mut __df_derive_outer_offsets: ::std::vec::Vec<i64> =
            ::std::vec::Vec::with_capacity(items.len() + 1);
        __df_derive_outer_offsets.push(0);
        let mut __df_derive_leaf_idx: usize = 0;
        for __df_derive_it in items {
            for __df_derive_inner in (&(#access)).iter() {
                for __df_derive_opt in __df_derive_inner.iter() {
                    match __df_derive_opt {
                        ::std::option::Option::Some(__df_derive_v) => {
                            __df_derive_view_buf
                                .push_value_ignore_validity(__df_derive_v.as_str());
                        }
                        ::std::option::Option::None => {
                            __df_derive_view_buf.push_value_ignore_validity("");
                            __df_derive_validity.set(__df_derive_leaf_idx, false);
                        }
                    }
                    __df_derive_leaf_idx += 1;
                }
                __df_derive_inner_offsets.push(__df_derive_leaf_idx as i64);
            }
            __df_derive_outer_offsets.push((__df_derive_inner_offsets.len() - 1) as i64);
        }
        let __df_derive_leaf_arr: #pa_root::array::Utf8ViewArray = __df_derive_view_buf
            .freeze()
            .with_validity(
                ::std::convert::Into::<::std::option::Option<#pa_root::bitmap::Bitmap>>::into(
                    __df_derive_validity,
                ),
            );
        let __df_derive_inner_offsets_buf: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(__df_derive_inner_offsets)?;
        let __df_derive_inner_list_arr: #pp::LargeListArray = #pp::LargeListArray::new(
            #pp::LargeListArray::default_datatype(
                #pa_root::array::Array::dtype(&__df_derive_leaf_arr).clone(),
            ),
            __df_derive_inner_offsets_buf,
            ::std::boxed::Box::new(__df_derive_leaf_arr) as #pp::ArrayRef,
            ::std::option::Option::None,
        );
        let __df_derive_outer_offsets_buf: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(__df_derive_outer_offsets)?;
        let __df_derive_outer_list_arr: #pp::LargeListArray = #pp::LargeListArray::new(
            #pp::LargeListArray::default_datatype(
                #pa_root::array::Array::dtype(&__df_derive_inner_list_arr).clone(),
            ),
            __df_derive_outer_offsets_buf,
            ::std::boxed::Box::new(__df_derive_inner_list_arr) as #pp::ArrayRef,
            ::std::option::Option::None,
        );
        __df_derive_assemble_list_series_unchecked(
            __df_derive_outer_list_arr,
            #inner_logical_dtype,
        )
    }};
    let emit = parent_name.map_or_else(
        || {
            quote! {{
                let __df_derive_series: #pp::Series = #series_block;
                out_values.push(#pp::AnyValue::List(__df_derive_series));
            }}
        },
        |name| {
            quote! {{
                let __df_derive_series: #pp::Series = #series_block;
                let __df_derive_named = __df_derive_series.with_name(#name.into());
                columns.push(__df_derive_named.into());
            }}
        },
    );
    Some(emit)
}

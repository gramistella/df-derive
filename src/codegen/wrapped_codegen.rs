use crate::ir::{BaseType, PrimitiveTransform, Wrapper};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

fn is_vec(wrappers: &[Wrapper]) -> bool {
    wrappers.iter().any(|w| matches!(w, Wrapper::Vec))
}

fn is_option(wrappers: &[Wrapper]) -> bool {
    wrappers.iter().any(|w| matches!(w, Wrapper::Option))
}

/// Per-strategy identifier convention for the columnar / vec-anyvalues
/// populator pipeline.
///
/// Names declared here in `primitive_decls` / `nested_decls` are referenced
/// by name in the per-row push helpers (`generate_primitive_for_columnar_push`,
/// `generate_nested_for_columnar_push`) and in the finishers
/// (`primitive_finishers_for_vec_anyvalues`, `nested_finishers_for_vec_anyvalues`,
/// `nested_columnar_builders`, plus `gen_columnar_builders` over in `strategy`).
/// Splitting the convention across `format_ident!` calls in each helper
/// silently breaks generated code on rename — the compiler can't see the
/// link, only a downstream "use of undeclared name" surfaces it. Funneling
/// every site through this struct turns rename mistakes into a compile error
/// at the helper itself.
pub(super) struct PopulatorIdents;

impl PopulatorIdents {
    /// Owning `Vec<T>` / `Vec<Option<T>>` buffer for a primitive scalar
    /// field. Holds `Vec<&str>` / `Vec<Option<&str>>` on the borrowing fast
    /// path (see `classify_borrow`).
    pub(super) fn primitive_buf(idx: usize) -> Ident {
        format_ident!("__df_derive_buf_{}", idx)
    }

    /// `Box<dyn ListBuilderTrait>` for `Vec<primitive>` shapes — the typed
    /// list builder that keeps the inner buffer typed end-to-end.
    pub(super) fn primitive_list_builder(idx: usize) -> Ident {
        format_ident!("__df_derive_pv_lb_{}", idx)
    }

    /// `Vec<Box<dyn ListBuilderTrait>>` — one inner-column builder per inner
    /// schema entry — for `Vec<Struct>` shapes that didn't take the
    /// bulk-concrete fast path.
    pub(super) fn nested_list_builders(idx: usize) -> Ident {
        format_ident!("__df_derive_nv_lbs_{}", idx)
    }

    /// `Vec<Vec<AnyValue>>` — one inner-column accumulator per inner schema
    /// entry — for non-vec nested-struct shapes.
    pub(super) fn nested_struct_cols(idx: usize) -> Ident {
        format_ident!("__df_derive_ns_cols_{}", idx)
    }

    /// Cached `<Inner>::schema()?` for `Vec<Struct>` shapes; paired with
    /// `nested_list_builders`.
    pub(super) fn nested_vec_schema(idx: usize) -> Ident {
        format_ident!("__df_derive_nv_schema_{}", idx)
    }

    /// Cached `<Inner>::schema()?` for non-vec nested-struct shapes; paired
    /// with `nested_struct_cols`.
    pub(super) fn nested_struct_schema(idx: usize) -> Ident {
        format_ident!("__df_derive_ns_schema_{}", idx)
    }
}

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

// --- Context-specific generation APIs ---

// --- Helpers to unify Vec processing across targets ---

/// Build tokens that evaluate to a `polars::prelude::Series` representing
/// the inner list for a primitive vector (including nested wrappers in `tail`).
fn gen_primitive_vec_inner_series(
    acc: &TokenStream,
    base_type: &BaseType,
    transform: Option<&PrimitiveTransform>,
    tail: &[Wrapper],
) -> TokenStream {
    let elem_ident = syn::Ident::new("__df_derive_vec_elem", proc_macro2::Span::call_site());
    let per_item_vals_ident =
        syn::Ident::new("__df_derive_elem_values", proc_macro2::Span::call_site());
    let list_vals_ident = syn::Ident::new("__df_derive_list_vals", proc_macro2::Span::call_site());

    let base_is_struct = matches!(base_type, BaseType::Struct(..));
    // `as_str` routes Vec<T> shapes (including `Vec<Struct>`) through the
    // borrowing fast path: `generate_inner_series_from_vec` already builds
    // `Vec<&str>` via UFCS. Without this carve-out, `Vec<Struct>+as_str`
    // would fall to the recursive per-element loop and clone every element.
    let as_str_fast_ok = matches!(transform, Some(PrimitiveTransform::AsStr));

    if (!base_is_struct || as_str_fast_ok) && tail.is_empty() {
        let fast_inner_ts =
            super::common::generate_inner_series_from_vec(acc, base_type, transform);
        return quote! {{ { #fast_inner_ts } }};
    }
    if !base_is_struct
        && tail.len() == 1
        && matches!(tail[0], Wrapper::Option)
        && transform.is_none()
    {
        return quote! {{
            <polars::prelude::Series as polars::prelude::NamedFrom<_, _>>::new(
                "".into(),
                &(#acc),
            )
        }};
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
    let tail_vec_count = tail.iter().filter(|w| matches!(w, Wrapper::Vec)).count();
    let mut empty_dtype = elem_dtype.clone();
    for _ in 0..tail_vec_count {
        empty_dtype =
            quote! { polars::prelude::DataType::List(::std::boxed::Box::new(#empty_dtype)) };
    }
    // Bind the cloned element to a named local so the inner recursion sees
    // a place expression rather than the `(*elem).clone()` temporary. The
    // inner-Vec borrowing path (`generate_inner_series_from_vec`) takes a
    // `&str` view into the access, and that borrow must outlive the
    // `Vec<&str>` it builds — a bare temp would drop at the previous `;`
    // and dangle. The binding has no effect on owning paths.
    let elem_owned_ident =
        syn::Ident::new("__df_derive_elem_owned", proc_macro2::Span::call_site());
    let elem_owned_access = quote! { #elem_owned_ident };
    let recur_elem_tokens_ts = super::wrapped_codegen::generate_primitive_for_anyvalue(
        &per_item_vals_ident,
        &elem_owned_access,
        base_type,
        transform,
        tail,
    );
    quote! {{
        let mut #list_vals_ident: ::std::vec::Vec<polars::prelude::AnyValue> = ::std::vec::Vec::with_capacity((#acc).len());
        for #elem_ident in (#acc).iter() {
            let #elem_owned_ident = (*#elem_ident).clone();
            let mut #per_item_vals_ident: ::std::vec::Vec<polars::prelude::AnyValue> = ::std::vec::Vec::new();
            { #recur_elem_tokens_ts }
            let __df_derive_elem_av = #per_item_vals_ident.pop().ok_or_else(|| polars::prelude::polars_err!(
                ComputeError: "df-derive: expected single AnyValue for primitive vec element (codegen invariant violation)"
            ))?;
            #list_vals_ident.push(__df_derive_elem_av);
        }
        let __df_derive_inner = if #list_vals_ident.is_empty() {
            polars::prelude::Series::new_empty("".into(), &#empty_dtype)
        } else {
            let mut __df_derive_s = <polars::prelude::Series as polars::prelude::NamedFrom<_, _>>::new("".into(), &#list_vals_ident);
            if #do_cast { __df_derive_s = __df_derive_s.cast(&#elem_dtype)?; }
            __df_derive_s
        };
        __df_derive_inner
    }}
}

/// Build tokens that evaluate to `Vec<polars::prelude::AnyValue>`, where each element
/// is a `AnyValue::List(inner_series)` for one nested struct field across the vector.
#[allow(clippy::too_many_lines)]
fn gen_nested_vec_to_list_anyvalues(
    ty: &TokenStream,
    acc: &TokenStream,
    tail: &[Wrapper],
) -> TokenStream {
    if tail.is_empty() {
        // Fast path: Vec<Struct>
        quote! { #ty::__df_derive_vec_to_inner_list_values(&(#acc))? }
    } else if tail.len() == 1 && matches!(tail[0], Wrapper::Option) {
        // Semi-optimized: Vec<Option<Struct>>. Build the inner DataFrame
        // once over the non-null subset, then per inner column scatter back
        // over the original `Vec<Option<Struct>>` positions via
        // `Series::take(&IdxCa)` — typed Series in, typed Series out, no
        // `Vec<AnyValue>` round-trip (which previously paid for an
        // AnyValue dispatch per outer position plus an inferring scan when
        // the outer Series was rebuilt).
        let schema_ident = syn::Ident::new("__df_derive_schema", proc_macro2::Span::call_site());
        let pos_ident = syn::Ident::new("__df_derive_pos", proc_macro2::Span::call_site());
        let nn_ident = syn::Ident::new("__df_derive_nn", proc_macro2::Span::call_site());
        let vals_ident = syn::Ident::new("__df_derive_vals", proc_macro2::Span::call_site());
        let take_ident = syn::Ident::new("__df_derive_take", proc_macro2::Span::call_site());
        quote! {{
            let #schema_ident = #ty::schema()?;
            let mut #pos_ident: ::std::vec::Vec<::std::option::Option<polars::prelude::IdxSize>> =
                ::std::vec::Vec::with_capacity((#acc).len());
            let mut #nn_ident: ::std::vec::Vec<#ty> = ::std::vec::Vec::new();
            for __df_derive_maybe in (#acc).iter() {
                match __df_derive_maybe {
                    ::std::option::Option::Some(v) => {
                        #pos_ident.push(::std::option::Option::Some(
                            #nn_ident.len() as polars::prelude::IdxSize,
                        ));
                        #nn_ident.push((*v).clone());
                    }
                    ::std::option::Option::None => #pos_ident.push(::std::option::Option::None),
                }
            }
            if #nn_ident.is_empty() {
                // All-None path: produce one outer-list cell per inner schema
                // column, each a typed-null Series of length `(#acc).len()`.
                // Pre-typing avoids feeding `dtype Null` into a list builder
                // that expects e.g. `list[Float64]` (which `ListPrimitiveChunkedBuilder::append_series` rejects).
                let mut __df_derive_out: ::std::vec::Vec<polars::prelude::AnyValue> =
                    ::std::vec::Vec::with_capacity(#schema_ident.len());
                for (_inner_name, __df_derive_inner_dtype) in #schema_ident.iter() {
                    let inner = polars::prelude::Series::new_empty("".into(), __df_derive_inner_dtype)
                        .extend_constant(polars::prelude::AnyValue::Null, (#acc).len())?;
                    __df_derive_out.push(polars::prelude::AnyValue::List(inner));
                }
                __df_derive_out
            } else {
                let #vals_ident = #ty::__df_derive_vec_to_inner_list_values(&#nn_ident)?;
                let #take_ident: polars::prelude::IdxCa =
                    <polars::prelude::IdxCa as polars::prelude::NewChunkedArray<_, _>>::from_iter_options(
                        "".into(),
                        #pos_ident.iter().copied(),
                    );
                let mut __df_derive_out: ::std::vec::Vec<polars::prelude::AnyValue> =
                    ::std::vec::Vec::with_capacity(#schema_ident.len());
                for j in 0..#schema_ident.len() {
                    let inner = match &#vals_ident[j] {
                        polars::prelude::AnyValue::List(__df_derive_inner_full) => {
                            __df_derive_inner_full.take(&#take_ident)?
                        }
                        _ => return ::std::result::Result::Err(polars::prelude::polars_err!(
                            ComputeError: "df-derive: expected list AnyValue from __df_derive_vec_to_inner_list_values (codegen invariant violation)"
                        )),
                    };
                    __df_derive_out.push(polars::prelude::AnyValue::List(inner));
                }
                __df_derive_out
            }
        }}
    } else {
        // Fallback: recursive per-element. Used for rarer nestings
        // (e.g. `Vec<Vec<Struct>>`). We accept a per-element AnyValue
        // round-trip here — the outer aggregation still uses typed Series,
        // and we explicitly type the empty case so list builders that demand
        // a specific inner dtype don't reject an inferred-Null Series.
        let schema_ident = syn::Ident::new("__df_derive_schema", proc_macro2::Span::call_site());
        let cols_buf_ident =
            syn::Ident::new("__df_derive_cols_buf", proc_macro2::Span::call_site());
        let elem_ident = syn::Ident::new("__df_derive_vec_elem", proc_macro2::Span::call_site());
        let per_item_vals_ident =
            syn::Ident::new("__df_derive_elem_values", proc_macro2::Span::call_site());

        let recur_elem = || {
            let elem_access = quote! { #elem_ident };
            super::wrapped_codegen::generate_nested_for_anyvalue(
                ty,
                &per_item_vals_ident,
                &elem_access,
                tail,
                false,
            )
        };
        let recur_elem_ts = recur_elem();

        // Each `Vec` in `tail` adds one extra `List<>` layer to the
        // inferred dtype of the inner Series rebuilt below. Skip the wrap
        // loop in generated tokens when `tail_vec_count == 0` so we don't
        // emit a `for _ in 0..0` that trips `clippy::reversed_empty_ranges`.
        let tail_vec_count = tail.iter().filter(|w| matches!(w, Wrapper::Vec)).count();
        let wrap_extra_for_empty = if tail_vec_count == 0 {
            quote! {}
        } else {
            let count_lit = tail_vec_count;
            quote! {
                for _ in 0..#count_lit {
                    __df_derive_wrapped = polars::prelude::DataType::List(
                        ::std::boxed::Box::new(__df_derive_wrapped),
                    );
                }
            }
        };

        quote! {{
            let #schema_ident = #ty::schema()?;
            let mut #cols_buf_ident: ::std::vec::Vec<::std::vec::Vec<polars::prelude::AnyValue>> = #schema_ident.iter().map(|_| ::std::vec::Vec::with_capacity((#acc).len())).collect();
            for #elem_ident in (#acc).iter() {
                let mut #per_item_vals_ident: ::std::vec::Vec<polars::prelude::AnyValue> = ::std::vec::Vec::new();
                { #recur_elem_ts }
                for (j, v) in #per_item_vals_ident.into_iter().enumerate() { #cols_buf_ident[j].push(v); }
            }
            let mut __df_derive_out: ::std::vec::Vec<polars::prelude::AnyValue> = ::std::vec::Vec::with_capacity(#schema_ident.len());
            for (j, (_inner_name, __df_derive_inner_dtype)) in #schema_ident.iter().enumerate() {
                let inner = if #cols_buf_ident[j].is_empty() {
                    let mut __df_derive_wrapped = __df_derive_inner_dtype.clone();
                    #wrap_extra_for_empty
                    polars::prelude::Series::new_empty("".into(), &__df_derive_wrapped)
                } else {
                    <polars::prelude::Series as polars::prelude::NamedFrom<_, _>>::new("".into(), &#cols_buf_ident[j])
                };
                __df_derive_out.push(polars::prelude::AnyValue::List(inner));
            }
            __df_derive_out
        }}
    }
}

/// Trait-only equivalent of `gen_nested_vec_to_list_anyvalues` for fields whose
/// base type is a generic type parameter. Avoids any inherent helpers and uses
/// only `ToDataFrame` / `Columnar` trait methods.
fn gen_generic_vec_to_list_anyvalues(
    ty: &TokenStream,
    acc: &TokenStream,
    tail: &[Wrapper],
) -> TokenStream {
    let schema_ident = syn::Ident::new("__df_derive_schema", proc_macro2::Span::call_site());
    let cols_buf_ident = syn::Ident::new("__df_derive_cols_buf", proc_macro2::Span::call_site());
    let elem_ident = syn::Ident::new("__df_derive_vec_elem", proc_macro2::Span::call_site());
    let per_item_vals_ident =
        syn::Ident::new("__df_derive_elem_values", proc_macro2::Span::call_site());

    let recur_elem_ts =
        generate_generic_for_anyvalue(ty, &per_item_vals_ident, &quote! { #elem_ident }, tail);

    // Each `Vec` in `tail` adds one `List<>` layer to the inferred dtype of
    // the inner Series rebuilt below (each per-element recursion pushes an
    // `AnyValue::List(...)` into `cols_buf`). The empty-input branch has to
    // produce that same wrapped dtype explicitly so a downstream typed list
    // builder doesn't reject the empty Series with `SchemaMismatch`.
    //
    // The wrap loop runs at macro time (not in generated tokens) so we don't
    // emit a `for _ in 0..0` for a zero-Vec tail — that would trip
    // `clippy::reversed_empty_ranges` inside the user's expanded code.
    let tail_vec_count = tail.iter().filter(|w| matches!(w, Wrapper::Vec)).count();
    let wrap_extra_for_empty = if tail_vec_count == 0 {
        quote! {}
    } else {
        let count_lit = tail_vec_count;
        quote! {
            for _ in 0..#count_lit {
                __df_derive_wrapped = polars::prelude::DataType::List(
                    ::std::boxed::Box::new(__df_derive_wrapped),
                );
            }
        }
    };

    quote! {{
        let #schema_ident = #ty::schema()?;
        let mut #cols_buf_ident: ::std::vec::Vec<::std::vec::Vec<polars::prelude::AnyValue>> =
            #schema_ident.iter().map(|_| ::std::vec::Vec::with_capacity((#acc).len())).collect();
        for #elem_ident in (#acc).iter() {
            let mut #per_item_vals_ident: ::std::vec::Vec<polars::prelude::AnyValue> = ::std::vec::Vec::new();
            { #recur_elem_ts }
            for (j, v) in #per_item_vals_ident.into_iter().enumerate() { #cols_buf_ident[j].push(v); }
        }
        let mut __df_derive_out: ::std::vec::Vec<polars::prelude::AnyValue> = ::std::vec::Vec::with_capacity(#schema_ident.len());
        for (j, (_inner_name, __df_derive_inner_dtype)) in #schema_ident.iter().enumerate() {
            let inner = if #cols_buf_ident[j].is_empty() {
                let mut __df_derive_wrapped = __df_derive_inner_dtype.clone();
                #wrap_extra_for_empty
                polars::prelude::Series::new_empty("".into(), &__df_derive_wrapped)
            } else {
                <polars::prelude::Series as polars::prelude::NamedFrom<_, _>>::new("".into(), &#cols_buf_ident[j])
            };
            __df_derive_out.push(polars::prelude::AnyValue::List(inner));
        }
        __df_derive_out
    }}
}

/// Trait-only on-leaf body for converting a single nested value into `AnyValues`
/// pushed onto `values_vec_ident` via `to_dataframe()`.
fn generic_leaf_to_anyvalues(values_vec_ident: &Ident, acc: &TokenStream) -> TokenStream {
    quote! {
        let __df_derive_tmp_df = (#acc).to_dataframe()?;
        let __df_derive_names: ::std::vec::Vec<String> = __df_derive_tmp_df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        for __df_derive_name in __df_derive_names.iter() {
            let __df_derive_v = __df_derive_tmp_df.column(__df_derive_name.as_str())?.get(0)?;
            #values_vec_ident.push(__df_derive_v.into_static());
        }
    }
}

/// Trait-only equivalent of `generate_nested_for_anyvalue` for generic params.
fn generate_generic_for_anyvalue(
    type_path: &TokenStream,
    values_vec_ident: &Ident,
    access: &TokenStream,
    wrappers: &[Wrapper],
) -> TokenStream {
    let ty = type_path.clone();

    let on_leaf = |acc: &TokenStream| generic_leaf_to_anyvalues(values_vec_ident, acc);

    let on_option_none = |_tail: &[Wrapper]| {
        quote! {
            let schema = #ty::schema()?;
            for _ in 0..schema.len() { #values_vec_ident.push(polars::prelude::AnyValue::Null); }
        }
    };

    let on_vec = |acc: &TokenStream, tail: &[Wrapper]| {
        let list_vals_ts = gen_generic_vec_to_list_anyvalues(&ty, acc, tail);
        quote! {{
            let __df_derive_vals: ::std::vec::Vec<polars::prelude::AnyValue> = { #list_vals_ts };
            for v in __df_derive_vals.into_iter() { #values_vec_ident.push(v); }
        }}
    };

    super::wrapper_processor::process_wrappers(access, wrappers, &on_leaf, &on_option_none, &on_vec)
}

// --- Generic-field bulk emitters ---
//
// These produce the entire builder/finisher token stream for a generic-typed
// field (i.e., the field's base type is a type parameter of the enclosing
// struct). They sidestep the per-row push pipeline entirely by collecting
// values into a contiguous slice once and calling
// `<T as Columnar>::columnar_to_dataframe(slice)` exactly once. From there
// each schema column of `T` is sliced/scattered into the parent's columns or
// `AnyValue::List` entries.
//
// Three wrapper shapes are supported as bulk: the bare leaf (`payload: T`),
// `Option<T>`, and `Vec<T>`. Deeper nestings (`Option<Vec<T>>`, etc.) keep
// using the per-row trait-only paths defined elsewhere in this module — those
// are rare and the bulk variants would need offset+position bookkeeping that
// isn't worth the added complexity.

/// Codegen context for the bulk emitters. Each variant carries exactly the
/// data its consumer needs: the columnar arm prefixes inner Series names with
/// the parent field name and pushes them onto the `columns` Vec; the
/// vec-anyvalues arm wraps each inner Series in `AnyValue::List` and pushes
/// onto `out_values`. Keeping `parent_name` inside `Columnar` (rather than as
/// a separate parameter on `bulk_consume_inner_series`) prevents the vec
/// path from accidentally depending on a value it can't use.
#[derive(Clone, Copy)]
pub enum BulkContext<'a> {
    /// Builder-position emit inside `columnar_to_dataframe`. Pushes prefixed
    /// columns onto the in-scope `columns` Vec.
    Columnar { parent_name: &'a str },
    /// Finisher-position emit inside `__df_derive_vec_to_inner_list_values`.
    /// Pushes `AnyValue::List(inner)` onto the in-scope `out_values` Vec.
    VecAnyvalues,
}

/// Build the per-column emit body that adapts an inner `Series` to the given
/// context. `series_expr` must evaluate to an owned `polars::prelude::Series`.
///
/// Note: this helper assumes `<T as Columnar>::columnar_to_dataframe` returns a
/// `DataFrame` whose columns appear in the same order as `T::schema()` —
/// every call iterates `T::schema()` and looks up by column name, so a
/// mismatch wouldn't crash but would produce a parent `DataFrame` whose column
/// order silently diverges from the declared schema. The derive enforces this
/// for its own generated impls; user-written `Columnar` impls must do the
/// same to be compatible with derives that use the type as a generic
/// parameter.
fn bulk_consume_inner_series(
    ctx: BulkContext<'_>,
    col_name_var: &TokenStream,
    series_expr: &TokenStream,
) -> TokenStream {
    match ctx {
        BulkContext::Columnar { parent_name } => quote! {{
            let __df_derive_prefixed = format!("{}.{}", #parent_name, #col_name_var);
            let __df_derive_inner = #series_expr;
            let __df_derive_named = __df_derive_inner.with_name(__df_derive_prefixed.as_str().into());
            columns.push(__df_derive_named.into());
        }},
        BulkContext::VecAnyvalues => quote! {{
            let __df_derive_inner = #series_expr;
            out_values.push(polars::prelude::AnyValue::List(__df_derive_inner));
        }},
    }
}

/// Bulk emit for a leaf generic field (`payload: T`). Collects `Vec<T>` once,
/// calls `T::columnar_to_dataframe`, then ships each schema column to `ctx`.
pub fn gen_bulk_generic_leaf(
    ty: &TokenStream,
    columnar_trait: &TokenStream,
    to_df_trait: &TokenStream,
    idx: usize,
    access: &TokenStream,
    ctx: BulkContext<'_>,
) -> TokenStream {
    let slice_ident = format_ident!("__df_derive_gen_slice_{}", idx);
    let df_ident = format_ident!("__df_derive_gen_df_{}", idx);
    let consume = bulk_consume_inner_series(
        ctx,
        &quote! { __df_derive_col_name },
        &quote! {
            #df_ident
                .column(__df_derive_col_name)?
                .as_materialized_series()
                .clone()
        },
    );
    quote! {{
        let #slice_ident: ::std::vec::Vec<#ty> = items
            .iter()
            .map(|__df_derive_it| (#access).clone())
            .collect();
        let #df_ident = <#ty as #columnar_trait>::columnar_to_dataframe(&#slice_ident)?;
        for (__df_derive_col_name, _) in <#ty as #to_df_trait>::schema()? {
            let __df_derive_col_name: &str = __df_derive_col_name.as_str();
            #consume
        }
    }}
}

/// Bulk emit for `payload: Option<T>`. Builds the gather (`nn`) plus position
/// (`pos`) vectors, calls `T::columnar_to_dataframe(&nn)` once (skipping when
/// every item is `None`), then scatters each `T` column back over the
/// original positions, emitting nulls where the source was `None`. Builds the
/// scattered Series via `Series::take` over an indices `IdxCa` instead of a
/// `Vec<AnyValue>` round-trip — typed buffers stay typed and we skip the
/// `AnyValue` dispatch per element.
pub fn gen_bulk_generic_option(
    ty: &TokenStream,
    columnar_trait: &TokenStream,
    to_df_trait: &TokenStream,
    idx: usize,
    access: &TokenStream,
    ctx: BulkContext<'_>,
) -> TokenStream {
    let nn_ident = format_ident!("__df_derive_gen_nn_{}", idx);
    let pos_ident = format_ident!("__df_derive_gen_pos_{}", idx);
    let take_ident = format_ident!("__df_derive_gen_take_{}", idx);
    let df_ident = format_ident!("__df_derive_gen_df_{}", idx);
    let dtype_var = quote! { __df_derive_dtype };
    let col_name_var = quote! { __df_derive_col_name };

    let consume_filled = bulk_consume_inner_series(
        ctx,
        &col_name_var,
        &quote! {{
            let __df_derive_inner_col = #df_ident
                .column(#col_name_var)?
                .as_materialized_series();
            __df_derive_inner_col.take(&#take_ident)?
        }},
    );
    let consume_empty = bulk_consume_inner_series(
        ctx,
        &col_name_var,
        &quote! {
            polars::prelude::Series::new_empty("".into(), &#dtype_var)
                .extend_constant(polars::prelude::AnyValue::Null, items.len())?
        },
    );

    quote! {{
        let mut #nn_ident: ::std::vec::Vec<#ty> = ::std::vec::Vec::new();
        let mut #pos_ident: ::std::vec::Vec<::std::option::Option<polars::prelude::IdxSize>> =
            ::std::vec::Vec::with_capacity(items.len());
        for __df_derive_it in items {
            match &(#access) {
                ::std::option::Option::Some(__df_derive_v) => {
                    #pos_ident.push(::std::option::Option::Some(
                        #nn_ident.len() as polars::prelude::IdxSize,
                    ));
                    #nn_ident.push(__df_derive_v.clone());
                }
                ::std::option::Option::None => {
                    #pos_ident.push(::std::option::Option::None);
                }
            }
        }
        if #nn_ident.is_empty() {
            for (#col_name_var, #dtype_var) in <#ty as #to_df_trait>::schema()? {
                let #col_name_var: &str = #col_name_var.as_str();
                let #dtype_var = &#dtype_var;
                #consume_empty
            }
        } else {
            let #df_ident = <#ty as #columnar_trait>::columnar_to_dataframe(&#nn_ident)?;
            let #take_ident: polars::prelude::IdxCa =
                <polars::prelude::IdxCa as polars::prelude::NewChunkedArray<_, _>>::from_iter_options(
                    "".into(),
                    #pos_ident.iter().copied(),
                );
            for (#col_name_var, _) in <#ty as #to_df_trait>::schema()? {
                let #col_name_var: &str = #col_name_var.as_str();
                #consume_filled
            }
        }
    }}
}

/// Build the per-inner-column emit for a bulk-vec context: given the inner
/// `DataFrame` (already computed from the flat slice) plus the `offsets`
/// array, slice the inner column per parent row and feed each slice to a
/// typed `ListBuilder`. The finisher produces a `ListChunked` that goes
/// straight into either the parent `columns` Vec or an outer `AnyValue::List`.
///
/// Going through `ListBuilderTrait` directly skips the dtype-inference scan
/// and per-row `cast(inner_type)` Polars does inside
/// `any_values_to_list` when consuming a `Vec<AnyValue::List>`. The
/// `ListPrimitiveChunkedBuilder` / `ListStringChunkedBuilder` selected by
/// `get_list_builder` for primitive inner dtypes copies elements with
/// `extend_from_slice` rather than per-element `AnyValue` dispatch; the
/// `AnonymousOwnedListBuilder` selected for struct/list inner dtypes
/// Arc-shares chunks rather than wrapping them in an `AnyValue` envelope.
fn bulk_vec_consume_inner_columns(
    ctx: BulkContext<'_>,
    df_ident: &Ident,
    offsets_ident: &Ident,
    schema_iter_ts: &TokenStream,
) -> TokenStream {
    let dtype_var = quote! { __df_derive_dtype };
    let col_name_var = quote! { __df_derive_col_name };
    let consume_filled = bulk_consume_inner_series(
        ctx,
        &col_name_var,
        &quote! {{
            let __df_derive_inner_col = #df_ident
                .column(#col_name_var)?
                .as_materialized_series();
            let mut __df_derive_lb = polars::chunked_array::builder::get_list_builder(
                #dtype_var,
                __df_derive_inner_col.len(),
                items.len(),
                "".into(),
            );
            for __df_derive_i in 0..items.len() {
                let __df_derive_start = #offsets_ident[__df_derive_i];
                let __df_derive_end = #offsets_ident[__df_derive_i + 1];
                let __df_derive_slice = __df_derive_inner_col
                    .slice(__df_derive_start as i64, __df_derive_end - __df_derive_start);
                polars::prelude::ListBuilderTrait::append_series(
                    &mut *__df_derive_lb,
                    &__df_derive_slice,
                )?;
            }
            polars::prelude::IntoSeries::into_series(
                polars::prelude::ListBuilderTrait::finish(&mut *__df_derive_lb),
            )
        }},
    );
    quote! {
        for (#col_name_var, #dtype_var) in #schema_iter_ts {
            let #col_name_var: &str = #col_name_var.as_str();
            let #dtype_var: &polars::prelude::DataType = &#dtype_var;
            #consume_filled
        }
    }
}

/// Build the all-empty-rows emit for a bulk-vec context: when every parent
/// row's inner Vec is empty (so the flat slice is empty and we skip the
/// inner `columnar_to_dataframe` call), still produce one outer-list column
/// per inner schema entry, each containing `items.len()` empty inner lists.
fn bulk_vec_consume_empty_columns(
    ctx: BulkContext<'_>,
    schema_iter_ts: &TokenStream,
) -> TokenStream {
    let dtype_var = quote! { __df_derive_dtype };
    let col_name_var = quote! { __df_derive_col_name };
    let consume_empty = bulk_consume_inner_series(
        ctx,
        &col_name_var,
        &quote! {{
            let __df_derive_empty = polars::prelude::Series::new_empty("".into(), #dtype_var);
            let mut __df_derive_lb = polars::chunked_array::builder::get_list_builder(
                #dtype_var,
                0,
                items.len(),
                "".into(),
            );
            for _ in 0..items.len() {
                polars::prelude::ListBuilderTrait::append_series(
                    &mut *__df_derive_lb,
                    &__df_derive_empty,
                )?;
            }
            polars::prelude::IntoSeries::into_series(
                polars::prelude::ListBuilderTrait::finish(&mut *__df_derive_lb),
            )
        }},
    );
    quote! {
        for (#col_name_var, #dtype_var) in #schema_iter_ts {
            let #col_name_var: &str = #col_name_var.as_str();
            let #dtype_var: &polars::prelude::DataType = &#dtype_var;
            #consume_empty
        }
    }
}

/// Bulk emit for `payload: Vec<T>` where `T` is a generic type parameter.
/// Flattens via `T: Clone` (already a macro-injected bound on `T`), calls
/// `<T as Columnar>::columnar_to_dataframe(&flat)` once, then ships each
/// inner column to `ctx` via `bulk_vec_consume_inner_columns`.
pub fn gen_bulk_generic_vec(
    ty: &TokenStream,
    columnar_trait: &TokenStream,
    to_df_trait: &TokenStream,
    idx: usize,
    access: &TokenStream,
    ctx: BulkContext<'_>,
) -> TokenStream {
    let flat_ident = format_ident!("__df_derive_gen_flat_{}", idx);
    let offsets_ident = format_ident!("__df_derive_gen_offsets_{}", idx);
    let df_ident = format_ident!("__df_derive_gen_df_{}", idx);
    let schema_iter = quote! { <#ty as #to_df_trait>::schema()? };
    let consume_filled =
        bulk_vec_consume_inner_columns(ctx, &df_ident, &offsets_ident, &schema_iter);
    let consume_empty = bulk_vec_consume_empty_columns(ctx, &schema_iter);

    quote! {{
        let mut #flat_ident: ::std::vec::Vec<#ty> = ::std::vec::Vec::new();
        let mut #offsets_ident: ::std::vec::Vec<usize> =
            ::std::vec::Vec::with_capacity(items.len() + 1);
        #offsets_ident.push(0);
        for __df_derive_it in items {
            for __df_derive_v in (&(#access)).iter() {
                #flat_ident.push((*__df_derive_v).clone());
            }
            #offsets_ident.push(#flat_ident.len());
        }
        if #flat_ident.is_empty() {
            #consume_empty
        } else {
            let #df_ident = <#ty as #columnar_trait>::columnar_to_dataframe(&#flat_ident)?;
            #consume_filled
        }
    }}
}

/// Bulk emit for `payload: Vec<T>` where `T` is a concrete derived struct
/// type (i.e. has the inherent `__df_derive_columnar_from_refs` helper).
/// Flattens via `&T` references — no `T: Clone` requirement — and calls
/// the inherent helper directly, then ships each inner column to `ctx` via
/// `bulk_vec_consume_inner_columns`.
pub fn gen_bulk_concrete_vec(
    ty: &TokenStream,
    to_df_trait: &TokenStream,
    idx: usize,
    access: &TokenStream,
    ctx: BulkContext<'_>,
) -> TokenStream {
    let flat_ident = format_ident!("__df_derive_gen_flat_{}", idx);
    let offsets_ident = format_ident!("__df_derive_gen_offsets_{}", idx);
    let df_ident = format_ident!("__df_derive_gen_df_{}", idx);
    let schema_iter = quote! { <#ty as #to_df_trait>::schema()? };
    let consume_filled =
        bulk_vec_consume_inner_columns(ctx, &df_ident, &offsets_ident, &schema_iter);
    let consume_empty = bulk_vec_consume_empty_columns(ctx, &schema_iter);

    quote! {{
        let mut #flat_ident: ::std::vec::Vec<&#ty> = ::std::vec::Vec::new();
        let mut #offsets_ident: ::std::vec::Vec<usize> =
            ::std::vec::Vec::with_capacity(items.len() + 1);
        #offsets_ident.push(0);
        for __df_derive_it in items {
            for __df_derive_v in (&(#access)).iter() {
                #flat_ident.push(__df_derive_v);
            }
            #offsets_ident.push(#flat_ident.len());
        }
        if #flat_ident.is_empty() {
            #consume_empty
        } else {
            let #df_ident = #ty::__df_derive_columnar_from_refs(&#flat_ident)?;
            #consume_filled
        }
    }}
}

// --- Primitive: context-specific generators ---

#[allow(clippy::too_many_lines)]
pub fn generate_primitive_for_series(
    series_name: &str,
    access: &TokenStream,
    base_type: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> TokenStream {
    let mapping = crate::codegen::type_registry::compute_mapping(base_type, transform, wrappers);
    let dtype = mapping.full_dtype.clone();
    let elem_rust_ty = mapping.rust_element_type;
    let do_cast = crate::codegen::type_registry::needs_cast(transform);

    let on_leaf = |acc: &TokenStream| {
        // Borrowing fast path: build the 1-row Series from `&[&str]` so the
        // per-row `to_dataframe(&self)` API doesn't allocate before handing
        // bytes to Polars.
        match classify_borrow(base_type, transform, wrappers) {
            Some(BorrowKind::StringLeaf) => quote! {
                vec![{
                    let s = <polars::prelude::Series as polars::prelude::NamedFrom<_, _>>::new(
                        #series_name.into(),
                        &[(#acc).as_str()],
                    );
                    s.into()
                }]
            },
            Some(BorrowKind::AsStr(ty_path)) => quote! {
                vec![{
                    let s = <polars::prelude::Series as polars::prelude::NamedFrom<_, _>>::new(
                        #series_name.into(),
                        &[<#ty_path as ::core::convert::AsRef<str>>::as_ref(&(#acc))],
                    );
                    s.into()
                }]
            },
            None => {
                let mapped = super::common::generate_primitive_access_expr(acc, transform);
                let cast_ts = if do_cast {
                    quote! { s = s.cast(&#dtype)?; }
                } else {
                    quote! {}
                };
                quote! {
                    vec![{
                        let mut s = <polars::prelude::Series as polars::prelude::NamedFrom<_, _>>::new(#series_name.into(), ::std::slice::from_ref(&{ #mapped }));
                        #cast_ts
                        s.into()
                    }]
                }
            }
        }
    };

    let on_option_none = |tail: &[Wrapper]| {
        let tail_has_vec = tail.iter().any(|w| matches!(w, Wrapper::Vec));
        if tail_has_vec {
            quote! {
                vec![{
                    let list_any_value = polars::prelude::AnyValue::Null;
                    <polars::prelude::Series as polars::prelude::NamedFrom<_, _>>::new(#series_name.into(), &[list_any_value]).into()
                }]
            }
        } else {
            quote! {
                vec![{
                    let __df_derive_tmp_opt: ::std::option::Option<#elem_rust_ty> = ::std::option::Option::None;
                    let mut s = <polars::prelude::Series as polars::prelude::NamedFrom<_, _>>::new(#series_name.into(), std::slice::from_ref(&__df_derive_tmp_opt));
                    if #do_cast { s = s.cast(&#dtype)?; }
                    s.into()
                }]
            }
        }
    };

    let on_vec = |acc: &TokenStream, tail: &[Wrapper]| {
        let inner_series_ts = gen_primitive_vec_inner_series(acc, base_type, transform, tail);
        quote! {{
            let inner_series = { #inner_series_ts };
            vec![{
                let list_any_value = polars::prelude::AnyValue::List(inner_series);
                <polars::prelude::Series as polars::prelude::NamedFrom<_, _>>::new(#series_name.into(), &[list_any_value]).into()
            }]
        }}
    };

    super::wrapper_processor::process_wrappers(access, wrappers, &on_leaf, &on_option_none, &on_vec)
}

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
        let opt = is_option(wrappers);
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

    let mapping = crate::codegen::type_registry::compute_mapping(base_type, transform, wrappers);
    let _dtype = mapping.full_dtype.clone();
    let _elem_rust_ty = mapping.rust_element_type;
    let _do_cast = crate::codegen::type_registry::needs_cast(transform);
    let opt_scalar = is_option(wrappers) && !is_vec(wrappers);

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
        let tail_has_vec = tail.iter().any(|w| matches!(w, Wrapper::Vec));
        if tail_has_vec {
            let lb_ident = PopulatorIdents::primitive_list_builder(idx);
            quote! { polars::prelude::ListBuilderTrait::append_null(&mut *#lb_ident); }
        } else {
            let vec_ident = PopulatorIdents::primitive_buf(idx);
            quote! { #vec_ident.push(::std::option::Option::None); }
        }
    };

    let on_vec = |acc: &TokenStream, tail: &[Wrapper]| {
        let inner_series_ts = gen_primitive_vec_inner_series(acc, base_type, transform, tail);
        let lb_ident = PopulatorIdents::primitive_list_builder(idx);
        quote! {{
            let inner_series = { #inner_series_ts };
            polars::prelude::ListBuilderTrait::append_series(&mut *#lb_ident, &inner_series)?;
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
    let mapping = crate::codegen::type_registry::compute_mapping(base_type, transform, wrappers);
    let dtype = mapping.full_dtype.clone();
    let _elem_rust_ty = mapping.rust_element_type;
    let do_cast = crate::codegen::type_registry::needs_cast(transform);

    let on_leaf = |acc: &TokenStream| {
        // Borrowing fast path: skip the user-side allocation by building the
        // 1-element Series from `&[&str]`. The Series owns its own Arrow
        // buffer once `from_slice` returns, so `s.get(0)?.into_static()` is
        // safe to call after the borrow's scope.
        match classify_borrow(base_type, transform, wrappers) {
            Some(BorrowKind::StringLeaf) => quote! {
                let s = <polars::prelude::Series as polars::prelude::NamedFrom<_, _>>::new(
                    "".into(),
                    &[(#acc).as_str()],
                );
                #values_vec_ident.push(s.get(0)?.into_static());
            },
            Some(BorrowKind::AsStr(ty_path)) => quote! {
                let s = <polars::prelude::Series as polars::prelude::NamedFrom<_, _>>::new(
                    "".into(),
                    &[<#ty_path as ::core::convert::AsRef<str>>::as_ref(&(#acc))],
                );
                #values_vec_ident.push(s.get(0)?.into_static());
            },
            None => {
                let mapped = super::common::generate_primitive_access_expr(acc, transform);
                let cast_ts = if do_cast {
                    quote! { s = s.cast(&#dtype)?; }
                } else {
                    quote! {}
                };
                quote! {
                    let mut s = <polars::prelude::Series as polars::prelude::NamedFrom<_, _>>::new("".into(), std::slice::from_ref(&{ #mapped }));
                    #cast_ts
                    #values_vec_ident.push(s.get(0)?.into_static());
                }
            }
        }
    };

    let on_option_none = |_tail: &[Wrapper]| {
        quote! { #values_vec_ident.push(polars::prelude::AnyValue::Null); }
    };

    let on_vec = |acc: &TokenStream, tail: &[Wrapper]| {
        let inner_series_ts = gen_primitive_vec_inner_series(acc, base_type, transform, tail);
        quote! {{
            let inner_series = { #inner_series_ts };
            #values_vec_ident.push(polars::prelude::AnyValue::List(inner_series));
        }}
    };

    super::wrapper_processor::process_wrappers(access, wrappers, &on_leaf, &on_option_none, &on_vec)
}

// --- Nested: context-specific generators ---

pub fn generate_nested_for_series(
    type_path: &TokenStream,
    series_name: &str,
    access: &TokenStream,
    wrappers: &[Wrapper],
    is_generic: bool,
) -> TokenStream {
    #![allow(clippy::too_many_lines)]
    let ty = type_path.clone();

    let on_leaf = |acc: &TokenStream| {
        let main_logic = generate_scalar_struct_logic(series_name, acc);
        quote! { #main_logic }
    };

    let on_option_none = |tail: &[Wrapper]| {
        let as_list_none = tail.iter().any(|w| matches!(w, Wrapper::Vec));
        generate_null_series_for_struct(&ty, series_name, as_list_none)
    };

    let on_vec = |acc: &TokenStream, tail: &[Wrapper]| {
        if tail.is_empty() && !is_generic {
            quote! {{
                #ty::__df_derive_collect_vec_as_prefixed_list_series(&(#acc), #series_name)?
            }}
        } else {
            let schema_ident =
                syn::Ident::new("__df_derive_schema", proc_macro2::Span::call_site());
            let vals_ident = syn::Ident::new("__df_derive_vals", proc_macro2::Span::call_site());
            let list_vals_ts = if is_generic {
                gen_generic_vec_to_list_anyvalues(&ty, acc, tail)
            } else {
                gen_nested_vec_to_list_anyvalues(&ty, acc, tail)
            };
            quote! {{
                let #schema_ident = #ty::schema()?;
                let #vals_ident: ::std::vec::Vec<polars::prelude::AnyValue> = { #list_vals_ts };
                let mut nested_series: ::std::vec::Vec<polars::prelude::Column> = ::std::vec::Vec::with_capacity(#schema_ident.len());
                for (j, (inner_name, _dtype)) in #schema_ident.iter().enumerate() {
                    let prefixed_name = format!("{}.{}", #series_name, inner_name);
                    let s = <polars::prelude::Series as polars::prelude::NamedFrom<_, _>>::new(prefixed_name.as_str().into(), &[#vals_ident[j].clone()]);
                    nested_series.push(s.into());
                }
                nested_series
            }}
        }
    };

    super::wrapper_processor::process_wrappers(access, wrappers, &on_leaf, &on_option_none, &on_vec)
}

pub fn generate_nested_for_columnar_push(
    type_path: &TokenStream,
    access: &TokenStream,
    wrappers: &[Wrapper],
    idx: usize,
    is_generic: bool,
) -> TokenStream {
    let ty = type_path.clone();
    let vec = is_vec(wrappers);

    // For non-vec shapes the populator is `Vec<Vec<AnyValue>>` (one inner
    // Vec per inner schema column, accumulating one AnyValue per outer row).
    // For vec shapes the populator is `Vec<Box<dyn ListBuilderTrait>>`
    // (one builder per inner schema column, accumulating one outer-list
    // entry per outer row). The on-leaf branch only runs for non-vec
    // shapes — `process_wrappers` reaches the leaf only when no `Vec`
    // wrapper is present.
    let cols_ident = PopulatorIdents::nested_struct_cols(idx);
    let lbs_ident = PopulatorIdents::nested_list_builders(idx);

    let on_leaf = |acc: &TokenStream| {
        let cols_ident = cols_ident.clone();
        if is_generic {
            quote! {
                let __df_derive_tmp_df = (#acc).to_dataframe()?;
                let __df_derive_names: ::std::vec::Vec<String> = __df_derive_tmp_df
                    .get_column_names()
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                for (j, __df_derive_name) in __df_derive_names.iter().enumerate() {
                    let __df_derive_v = __df_derive_tmp_df
                        .column(__df_derive_name.as_str())?
                        .get(0)?
                        .into_static();
                    #cols_ident[j].push(__df_derive_v);
                }
            }
        } else {
            quote! {
                let nested_values = (#acc).__df_derive_to_anyvalues()?;
                for (j, value) in nested_values.into_iter().enumerate() {
                    #cols_ident[j].push(value);
                }
            }
        }
    };

    let on_option_none = |_tail: &[Wrapper]| {
        if vec {
            let lbs_ident = lbs_ident.clone();
            quote! {
                for j in 0..#lbs_ident.len() {
                    polars::prelude::ListBuilderTrait::append_null(&mut *#lbs_ident[j]);
                }
            }
        } else {
            let cols_ident = cols_ident.clone();
            quote! {
                for j in 0..#cols_ident.len() {
                    #cols_ident[j].push(polars::prelude::AnyValue::Null);
                }
            }
        }
    };

    let on_vec = |acc: &TokenStream, tail: &[Wrapper]| {
        let lbs_ident = lbs_ident.clone();
        let list_vals_ts = if is_generic {
            gen_generic_vec_to_list_anyvalues(&ty, acc, tail)
        } else {
            gen_nested_vec_to_list_anyvalues(&ty, acc, tail)
        };
        quote! {{
            let __df_derive_vals: ::std::vec::Vec<polars::prelude::AnyValue> = { #list_vals_ts };
            for (j, __df_derive_v) in __df_derive_vals.into_iter().enumerate() {
                match __df_derive_v {
                    polars::prelude::AnyValue::List(__df_derive_inner) => {
                        polars::prelude::ListBuilderTrait::append_series(
                            &mut *#lbs_ident[j],
                            &__df_derive_inner,
                        )?;
                    }
                    polars::prelude::AnyValue::Null => {
                        polars::prelude::ListBuilderTrait::append_null(&mut *#lbs_ident[j]);
                    }
                    _ => {
                        return ::std::result::Result::Err(polars::prelude::polars_err!(
                            ComputeError: "df-derive: expected list or null AnyValue from nested vec helper (codegen invariant violation)"
                        ));
                    }
                }
            }
        }}
    };

    super::wrapper_processor::process_wrappers(access, wrappers, &on_leaf, &on_option_none, &on_vec)
}

pub fn generate_nested_for_anyvalue(
    type_path: &TokenStream,
    values_vec_ident: &Ident,
    access: &TokenStream,
    wrappers: &[Wrapper],
    is_generic: bool,
) -> TokenStream {
    let ty = type_path.clone();

    let on_leaf = |acc: &TokenStream| {
        quote! {
            let tmp_df = (#acc).to_dataframe()?;
            for col_name in tmp_df.get_column_names() {
                let v = tmp_df.column(col_name)?.get(0)?;
                #values_vec_ident.push(v.into_static());
            }
        }
    };

    let on_option_none = |_tail: &[Wrapper]| {
        quote! {
            let schema = #ty::schema()?;
            for _ in 0..schema.len() { #values_vec_ident.push(polars::prelude::AnyValue::Null); }
        }
    };

    let on_vec = |acc: &TokenStream, tail: &[Wrapper]| {
        let list_vals_ts = if is_generic {
            gen_generic_vec_to_list_anyvalues(&ty, acc, tail)
        } else {
            gen_nested_vec_to_list_anyvalues(&ty, acc, tail)
        };
        quote! {{
            let __df_derive_vals: ::std::vec::Vec<polars::prelude::AnyValue> = { #list_vals_ts };
            for v in __df_derive_vals.into_iter() { #values_vec_ident.push(v); }
        }}
    };

    super::wrapper_processor::process_wrappers(access, wrappers, &on_leaf, &on_option_none, &on_vec)
}
// --- Primitive: Columnar (centralized move from wrapper_logic.rs) ---

pub fn primitive_decls(
    wrappers: &[Wrapper],
    base_type: &BaseType,
    transform: Option<&PrimitiveTransform>,
    idx: usize,
) -> Vec<TokenStream> {
    let mut decls: Vec<TokenStream> = Vec::new();
    let opt = is_option(wrappers);
    let vec = is_vec(wrappers);

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
        decls.push(quote! {
            let mut #lb_ident: ::std::boxed::Box<dyn polars::prelude::ListBuilderTrait> =
                polars::chunked_array::builder::get_list_builder(
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
    let vec = is_vec(wrappers);
    let mapping = crate::codegen::type_registry::compute_mapping(base_type, transform, wrappers);
    let needs_cast = crate::codegen::type_registry::needs_cast(transform);
    if vec {
        let lb_ident = PopulatorIdents::primitive_list_builder(idx);
        quote! {
            let inner = polars::prelude::IntoSeries::into_series(
                polars::prelude::ListBuilderTrait::finish(&mut *#lb_ident),
            );
            out_values.push(polars::prelude::AnyValue::List(inner));
        }
    } else {
        let dtype = mapping.full_dtype;
        let vec_ident = PopulatorIdents::primitive_buf(idx);
        quote! {
            let mut inner = <polars::prelude::Series as polars::prelude::NamedFrom<_, _>>::new("".into(), &#vec_ident);
            if #needs_cast { inner = inner.cast(&#dtype)?; }
            out_values.push(polars::prelude::AnyValue::List(inner));
        }
    }
}

// --- Nested Structs: Row-wise and Columnar (centralized) ---

pub fn nested_empty_series_row(
    type_path: &TokenStream,
    name: &str,
    wrappers: &[Wrapper],
) -> TokenStream {
    generate_empty_series_for_struct(type_path, name, is_vec(wrappers))
}

pub fn nested_decls(wrappers: &[Wrapper], type_path: &TokenStream, idx: usize) -> Vec<TokenStream> {
    let mut decls: Vec<TokenStream> = Vec::new();
    let vec = is_vec(wrappers);
    if vec {
        // Vec<Struct> shapes that didn't take the bulk-concrete fast path
        // (i.e. `Vec<Option<Struct>>`, `Vec<Vec<Struct>>`, etc.). Per parent
        // row we still call the inner helper to get one inner Series per
        // schema column; instead of accumulating those into a
        // `Vec<AnyValue::List>` and rebuilding the outer list series via
        // `Series::new`, we feed each inner Series straight into a typed
        // `ListBuilder` per inner column. Skips the AnyValue inference scan
        // and per-row `cast(inner_type)` Polars does inside
        // `any_values_to_list`.
        //
        // For nested-Vec shapes (`Vec<Vec<Struct>>`, …), the per-row inner
        // Series feeding the builder is itself `List<…>`-shaped, so the
        // builder's inner dtype must include `(vec_count - 1)` extra
        // `List<>` layers around the inner-struct schema dtype — see the
        // analogous wrap in `outer_list_inner_dtype` for primitives. Without
        // this, `ListPrimitiveChunkedBuilder` rejects the deeper `list[…]`
        // slice with a `SchemaMismatch`.
        let schema_ident = PopulatorIdents::nested_vec_schema(idx);
        let lbs_ident = PopulatorIdents::nested_list_builders(idx);
        let extra_list_layers = wrappers
            .iter()
            .filter(|w| matches!(w, Wrapper::Vec))
            .count()
            .saturating_sub(1);
        // Emit the wrap loop only when there's something to wrap. `for _ in
        // 0..0` is technically fine but trips `clippy::reversed_empty_ranges`
        // inside the user's monomorphized code, which we can't easily silence
        // from a macro without leaking attribute annotations into user types.
        let wrap_extra = if extra_list_layers == 0 {
            quote! {}
        } else {
            quote! {
                for _ in 0..#extra_list_layers {
                    __df_derive_wrapped = polars::prelude::DataType::List(
                        ::std::boxed::Box::new(__df_derive_wrapped),
                    );
                }
            }
        };
        decls.push(quote! { let #schema_ident = #type_path::schema()?; });
        decls.push(quote! {
            let mut #lbs_ident: ::std::vec::Vec<
                ::std::boxed::Box<dyn polars::prelude::ListBuilderTrait>,
            > = #schema_ident
                .iter()
                .map(|(_, __df_derive_inner_dtype)| {
                    let mut __df_derive_wrapped = __df_derive_inner_dtype.clone();
                    #wrap_extra
                    polars::chunked_array::builder::get_list_builder(
                        &__df_derive_wrapped,
                        items.len() * 4,
                        items.len(),
                        "".into(),
                    )
                })
                .collect();
        });
    } else {
        let schema_ident = PopulatorIdents::nested_struct_schema(idx);
        let cols_ident = PopulatorIdents::nested_struct_cols(idx);
        decls.push(quote! { let #schema_ident = #type_path::schema()?; });
        decls.push(quote! {
            let mut #cols_ident: ::std::vec::Vec<::std::vec::Vec<polars::prelude::AnyValue>> =
                #schema_ident
                    .iter()
                    .map(|_| ::std::vec::Vec::with_capacity(items.len()))
                    .collect();
        });
    }
    decls
}

pub fn nested_finishers_for_vec_anyvalues(wrappers: &[Wrapper], idx: usize) -> TokenStream {
    let vec = is_vec(wrappers);
    let schema_ident = if vec {
        PopulatorIdents::nested_vec_schema(idx)
    } else {
        PopulatorIdents::nested_struct_schema(idx)
    };
    if vec {
        let lbs_ident = PopulatorIdents::nested_list_builders(idx);
        quote! {
            for j in 0..#schema_ident.len() {
                let inner = polars::prelude::IntoSeries::into_series(
                    polars::prelude::ListBuilderTrait::finish(&mut *#lbs_ident[j]),
                );
                out_values.push(polars::prelude::AnyValue::List(inner));
            }
        }
    } else {
        let cols_ident = PopulatorIdents::nested_struct_cols(idx);
        quote! {
            for j in 0..#schema_ident.len() {
                let inner = <polars::prelude::Series as polars::prelude::NamedFrom<_, _>>::new("".into(), &#cols_ident[j]);
                out_values.push(polars::prelude::AnyValue::List(inner));
            }
        }
    }
}

pub fn nested_columnar_builders(
    wrappers: &[Wrapper],
    idx: usize,
    field_name: &str,
) -> Vec<TokenStream> {
    let vec = is_vec(wrappers);
    let schema_ident = if vec {
        PopulatorIdents::nested_vec_schema(idx)
    } else {
        PopulatorIdents::nested_struct_schema(idx)
    };
    let name = field_name;
    if vec {
        let lbs_ident = PopulatorIdents::nested_list_builders(idx);
        vec![quote! {{
            for (j, (col_name, _)) in #schema_ident.iter().enumerate() {
                let full_name = format!("{}.{}", #name, col_name);
                let s = polars::prelude::IntoSeries::into_series(
                    polars::prelude::ListBuilderTrait::finish(&mut *#lbs_ident[j]),
                )
                .with_name(full_name.as_str().into());
                columns.push(s.into());
            }
        }}]
    } else {
        let cols_ident = PopulatorIdents::nested_struct_cols(idx);
        vec![quote! {{
            for (j, (col_name, _)) in #schema_ident.iter().enumerate() {
                let full_name = format!("{}.{}", #name, col_name);
                let s = <polars::prelude::Series as polars::prelude::NamedFrom<_, _>>::new(full_name.as_str().into(), &#cols_ident[j]);
                columns.push(s.into());
            }
        }}]
    }
}

pub fn generate_schema_entries_for_struct(
    type_path: &TokenStream,
    column_name: &str,
    as_list: bool,
) -> TokenStream {
    quote! {
        {
            let mut nested_fields: ::std::vec::Vec<(::std::string::String, polars::prelude::DataType)> = ::std::vec::Vec::new();
            for (inner_name, inner_dtype) in #type_path::schema()? {
                let prefixed_name = format!("{}.{}", #column_name, inner_name);
                let dtype = if #as_list {
                    polars::prelude::DataType::List(Box::new(inner_dtype))
                } else {
                    inner_dtype
                };
                nested_fields.push((prefixed_name, dtype));
            }
            nested_fields
        }
    }
}

fn generate_empty_series_for_struct(
    type_path: &TokenStream,
    column_name: &str,
    as_list: bool,
) -> TokenStream {
    quote! {
        {
            let mut nested_series = Vec::new();
            for (inner_name, inner_dtype) in #type_path::schema()? {
                let prefixed_name = format!("{}.{}", #column_name, inner_name);
                let dtype = if #as_list {
                    polars::prelude::DataType::List(Box::new(inner_dtype))
                } else {
                    inner_dtype
                };
                let empty_series = polars::prelude::Series::new_empty(prefixed_name.as_str().into(), &dtype);
                nested_series.push(empty_series.into());
            }
            nested_series
        }
    }
}

fn generate_null_series_for_struct(
    type_path: &TokenStream,
    column_name: &str,
    as_list: bool,
) -> TokenStream {
    quote! {
        {
            let mut nested_series = Vec::new();
            for (inner_name, inner_dtype) in #type_path::schema()? {
                let prefixed_name = format!("{}.{}", #column_name, inner_name);
                let dtype = if #as_list {
                    polars::prelude::DataType::List(Box::new(inner_dtype))
                } else {
                    inner_dtype
                };
                let null_series = polars::prelude::Series::new_empty(prefixed_name.as_str().into(), &dtype);
                let null_series_with_value = null_series.extend_constant(polars::prelude::AnyValue::Null, 1)?;
                nested_series.push(null_series_with_value.into());
            }
            nested_series
        }
    }
}

fn generate_scalar_struct_logic(column_name: &str, access_path: &TokenStream) -> TokenStream {
    quote! {
        {
            let nested_df = (#access_path).to_dataframe()?;
            let mut nested_series = Vec::new();

            for col_name in nested_df.get_column_names() {
                let prefixed_name = format!("{}.{}", #column_name, col_name);
                let series = nested_df.column(col_name)?.clone().with_name(prefixed_name.as_str().into());
                nested_series.push(series.into());
            }
            nested_series
        }
    }
}

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

/// Top-level `String` (no transform, no `Vec<…>`, no `Option<…>`) bypasses
/// the `Vec<&str>` intermediate buffer entirely and accumulates straight
/// into a `MutableBinaryViewArray<str>` during the items loop. The finisher
/// then wraps it in `StringChunked::with_chunk` — same `Utf8ViewArray`-backed
/// column the `Series::new(&Vec<&str>)` path produces, minus the second walk
/// that `from_slice_values` would do.
///
/// `Option<String>` is handled separately by `is_direct_view_option_string_leaf`
/// via a split-buffer pattern (values into `MutableBinaryViewArray<str>`,
/// validity into `MutableBitmap`). A previous single-buffer attempt that
/// routed the per-row `match` through `push_value`/`push_null` measured a
/// ~7-8% regression on `string_columns_optional` because the validity
/// bookkeeping inside `MutableBinaryViewArray` is per-row branchy on
/// `validity.is_some()`. Splitting validity into its own bitmap mirrors the
/// `Option<numeric>` fix and decouples the two pushes so each one is a
/// straight-line append.
pub(super) const fn is_direct_view_string_leaf(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> bool {
    transform.is_none() && matches!(base, BaseType::String) && wrappers.is_empty()
}

/// Top-level `Option<String>` leaf (no transform, exactly `[Option]`). The
/// fast path declares a `MutableBinaryViewArray<str>` for values plus a
/// `MutableBitmap` for validity, with the bitmap pre-filled to all-`true`
/// at decl time. Per-row, `Some` pushes the row's `&str` to the view array
/// and does no validity work (the bit is already set); `None` pushes an
/// empty placeholder string (which `push_value_into_buffer` handles as an
/// inline view, no buffer allocation) and flips the corresponding bit to
/// `false` via the safe `MutableBitmap::set`. A row counter tracks the
/// index so the `None` arm can address the bit directly without a per-row
/// push. The safe `set` is used (not `set_unchecked`) so no `unsafe` lands
/// inside the user's `Columnar::columnar_from_refs` impl method — that
/// would trip `clippy::unsafe_derive_deserialize` downstream.
///
/// The pre-fill+set scheme dominates the symmetric `validity.push(true/false)`
/// alternative because most rows in real data are `Some`, so the common case
/// becomes zero validity work per row. The finisher attaches the bitmap via
/// `Utf8ViewArray::with_validity` — `Option<Bitmap>::from(MutableBitmap)`
/// collapses to `None` when no bits are unset, preserving the no-null fast
/// path.
pub(super) const fn is_direct_view_option_string_leaf(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> bool {
    transform.is_none() && matches!(base, BaseType::String) && matches!(wrappers, [Wrapper::Option])
}

/// Top-level `as_string` leaf (`#[df_derive(as_string)]`) with bare or
/// `Option<…>` wrappers, any base whose `Display::fmt` produces a string.
/// Replaces the per-row `(field).to_string()` allocation with a reused
/// `String` scratch + `MutableBinaryViewArray<str>` accumulator: each row
/// clears the scratch, runs `write!(scratch, "{}", field)` (Display writes
/// straight into the scratch), then `push_value(scratch.as_str())` copies
/// the bytes into the view array — so reusing the scratch the next row is
/// safe. Replaces `items.len()` heap allocations with the scratch's
/// log-shaped growth (effectively constant for fixed-format Displays like
/// enum names), and skips the second walk over `Vec<String>` that
/// `Series::new` would do internally.
///
/// Same `wrappers in {[], [Option]}` carve-out as the borrowing/`as_str`
/// path: deeper nestings (`Vec<…>`, `Option<Option<…>>`) keep the existing
/// owning-buffer pipeline because the per-row work to flatten them into a
/// single `MutableBinaryViewArray<str>` would overshoot the saving.
pub(super) const fn is_direct_view_to_string_leaf(
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> bool {
    matches!(transform, Some(PrimitiveTransform::ToString))
        && matches!(wrappers, [] | [Wrapper::Option])
}

/// Per-row push tokens for the `is_direct_view_to_string_leaf` fast path.
/// Caller must have already gated on the predicate. The generated code
/// clears the reused `String` scratch, runs `Display::fmt` into it via
/// `write!`, and pushes the resulting `&str` into the
/// `MutableBinaryViewArray<str>` (`push_value_ignore_validity` copies the
/// bytes, so the scratch is safe to reuse on the next row).
///
/// Both shapes track validity externally — the bare arm has no validity at
/// all (none ever initialized) and the `[Option]` arm uses the same
/// split-buffer pattern as `gen_option_string_direct_push` (a parallel
/// `MutableBitmap` pre-filled to all-`true`, with the `None` arm flipping
/// the row's bit to `false` via the safe `MutableBitmap::set`). Neither
/// shape ever calls `push_null` / `init_validity`, so the buffer's internal
/// validity is always `None` and `push_value_ignore_validity` skips the
/// per-row `if let Some(validity) ... validity.push(true)` branch
/// `push_value` would otherwise do.
fn gen_direct_view_to_string_push(
    access: &TokenStream,
    wrappers: &[Wrapper],
    idx: usize,
) -> TokenStream {
    let buf_ident = PopulatorIdents::primitive_buf(idx);
    let str_ident = PopulatorIdents::primitive_str_scratch(idx);
    if matches!(wrappers, [Wrapper::Option]) {
        let validity_ident = PopulatorIdents::primitive_validity(idx);
        let row_idx = PopulatorIdents::primitive_row_idx(idx);
        return quote! {
            match &(#access) {
                ::std::option::Option::Some(__df_derive_v) => {
                    use ::std::fmt::Write as _;
                    #str_ident.clear();
                    ::std::write!(&mut #str_ident, "{}", __df_derive_v).unwrap();
                    #buf_ident.push_value_ignore_validity(#str_ident.as_str());
                }
                ::std::option::Option::None => {
                    #buf_ident.push_value_ignore_validity("");
                    #validity_ident.set(#row_idx, false);
                }
            }
            #row_idx += 1;
        };
    }
    quote! {
        {
            use ::std::fmt::Write as _;
            #str_ident.clear();
            ::std::write!(&mut #str_ident, "{}", &(#access)).unwrap();
            #buf_ident.push_value_ignore_validity(#str_ident.as_str());
        }
    }
}

/// Top-level bare-numeric leaf (no transform, no `Vec<…>`, no `Option<…>`).
/// The buffer is already `Vec<Native>`; the finisher swaps the
/// `Series::new(&Vec<Native>)` path — which dispatches to `from_slice` and
/// internally `memcpy`s the slice into a fresh `PrimitiveArray::from_slice` —
/// for `<Chunked>::from_vec(name, buf).into_series()`, which consumes the Vec
/// without copying via `to_primitive`.
///
/// `ISize`/`USize` are excluded conservatively: their buffer element type
/// is `i64`/`u64` (chosen by `compute_mapping`) so they would work fine on
/// this path, but no bench currently exercises them and we'd rather measure
/// before extending. `Bool` is excluded because the validity-bit semantics
/// of `BooleanChunked` differ from numeric primitives and the bool slow
/// path is already a tight `from_slice`. Decimal/DateTime are transforms,
/// hence ruled out by the transform check.
pub(super) const fn is_direct_primitive_array_numeric_leaf(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> bool {
    if transform.is_some() || !wrappers.is_empty() {
        return false;
    }
    matches!(
        base,
        BaseType::I8
            | BaseType::I16
            | BaseType::I32
            | BaseType::I64
            | BaseType::U8
            | BaseType::U16
            | BaseType::U32
            | BaseType::U64
            | BaseType::F32
            | BaseType::F64
    )
}

/// Polars chunked-array type token for the bare-numeric direct-finish path.
/// Returns the prelude path to the `*Chunked` alias for each eligible
/// `BaseType` — `Int64Chunked` etc. — paired with the same prelude root.
/// Caller should only invoke after `is_direct_primitive_array_numeric_leaf`
/// or `is_direct_primitive_array_option_numeric_leaf` returns `true`, which
/// restricts inputs to the bases enumerated here.
pub(super) fn numeric_chunked_type(base: &BaseType) -> TokenStream {
    let pp = super::polars_paths::prelude();
    match base {
        BaseType::I8 => quote! { #pp::Int8Chunked },
        BaseType::I16 => quote! { #pp::Int16Chunked },
        BaseType::I32 => quote! { #pp::Int32Chunked },
        BaseType::I64 => quote! { #pp::Int64Chunked },
        BaseType::U8 => quote! { #pp::UInt8Chunked },
        BaseType::U16 => quote! { #pp::UInt16Chunked },
        BaseType::U32 => quote! { #pp::UInt32Chunked },
        BaseType::U64 => quote! { #pp::UInt64Chunked },
        BaseType::F32 => quote! { #pp::Float32Chunked },
        BaseType::F64 => quote! { #pp::Float64Chunked },
        BaseType::Bool
        | BaseType::String
        | BaseType::ISize
        | BaseType::USize
        | BaseType::DateTimeUtc
        | BaseType::Decimal
        | BaseType::Struct(..)
        | BaseType::Generic(_) => unreachable!(
            "numeric_chunked_type called for non-numeric base; \
             callers must gate on is_direct_primitive_array_numeric_leaf"
        ),
    }
}

/// Native Rust type token for the bare-numeric direct-finish path. Used by
/// the `Option<numeric>` direct-`PrimitiveArray::new` fast path to type the
/// `Vec<#native>` values buffer (the `MutableBitmap` carries validity
/// separately so the buffer holds the value-or-default placeholder
/// directly). Caller must gate on `is_direct_primitive_array_numeric_leaf`
/// or `is_direct_primitive_array_option_numeric_leaf`.
pub(super) fn numeric_native_rust_type(base: &BaseType) -> TokenStream {
    match base {
        BaseType::I8 => quote! { i8 },
        BaseType::I16 => quote! { i16 },
        BaseType::I32 => quote! { i32 },
        BaseType::I64 => quote! { i64 },
        BaseType::U8 => quote! { u8 },
        BaseType::U16 => quote! { u16 },
        BaseType::U32 => quote! { u32 },
        BaseType::U64 => quote! { u64 },
        BaseType::F32 => quote! { f32 },
        BaseType::F64 => quote! { f64 },
        BaseType::Bool
        | BaseType::String
        | BaseType::ISize
        | BaseType::USize
        | BaseType::DateTimeUtc
        | BaseType::Decimal
        | BaseType::Struct(..)
        | BaseType::Generic(_) => unreachable!(
            "numeric_native_rust_type called for non-numeric base; \
             callers must gate on is_direct_primitive_array_(option_)numeric_leaf"
        ),
    }
}

/// Top-level `Option<numeric>` leaf (no transform, exactly `[Option]`).
/// Same numeric bases as `is_direct_primitive_array_numeric_leaf`. The fast
/// path declares `Vec<#native>` + `MutableBitmap` instead of
/// `Vec<Option<#native>>` and accumulates value+validity separately during
/// the items loop. The finisher then builds a `PrimitiveArray::new(dtype,
/// values.into(), Some(validity.into()))` and wraps it in
/// `<*Chunked>::with_chunk(name, arr).into_series()` — same column the
/// `Series::new(&Vec<Option<T>>) -> from_slice_options ->
/// PrimitiveChunkedBuilder` slow path produces, but skips the second walk
/// over the `Vec<Option<T>>` and the per-row "is validity active?" branch
/// inside `MutablePrimitiveArray::push(Option<T>)`.
pub(super) const fn is_direct_primitive_array_option_numeric_leaf(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> bool {
    if transform.is_some() || !matches!(wrappers, [Wrapper::Option]) {
        return false;
    }
    matches!(
        base,
        BaseType::I8
            | BaseType::I16
            | BaseType::I32
            | BaseType::I64
            | BaseType::U8
            | BaseType::U16
            | BaseType::U32
            | BaseType::U64
            | BaseType::F32
            | BaseType::F64
    )
}

/// Buffer-pair declarations for the `Option<numeric>` direct fast path: a
/// `Vec<#native>` for values and a `MutableBitmap` for validity, both
/// pre-sized to `items.len()`. Caller must gate on
/// `is_direct_primitive_array_option_numeric_leaf`.
fn gen_option_numeric_direct_decls(base: &BaseType, idx: usize) -> Vec<TokenStream> {
    let buf_ident = PopulatorIdents::primitive_buf(idx);
    let validity_ident = PopulatorIdents::primitive_validity(idx);
    let native = numeric_native_rust_type(base);
    let pa_root = super::polars_paths::polars_arrow_root();
    vec![
        quote! {
            let mut #buf_ident: ::std::vec::Vec<#native> =
                ::std::vec::Vec::with_capacity(items.len());
        },
        quote! {
            let mut #validity_ident: #pa_root::bitmap::MutableBitmap =
                #pa_root::bitmap::MutableBitmap::with_capacity(items.len());
        },
    ]
}

/// Per-row push tokens for the `Option<numeric>` direct fast path. Caller
/// must gate on `is_direct_primitive_array_option_numeric_leaf`. Splits
/// value vs validity into two independent pushes (`Vec::push` for the value,
/// `MutableBitmap::push` for the bit) so the compiler can vectorize the
/// no-null fast path cleanly across rows. The None branch pushes
/// `<#native>::default()` as a placeholder — the bitmap records the row as
/// invalid, so the placeholder is never observed downstream.
pub(super) fn gen_option_numeric_direct_push(
    access: &TokenStream,
    base: &BaseType,
    idx: usize,
) -> TokenStream {
    let buf_ident = PopulatorIdents::primitive_buf(idx);
    let validity_ident = PopulatorIdents::primitive_validity(idx);
    let native = numeric_native_rust_type(base);
    quote! {
        match #access {
            ::std::option::Option::Some(__df_derive_v) => {
                #buf_ident.push(__df_derive_v);
                #validity_ident.push(true);
            }
            ::std::option::Option::None => {
                #buf_ident.push(<#native as ::std::default::Default>::default());
                #validity_ident.push(false);
            }
        }
    }
}

/// Build a `PrimitiveArray<#native>` from the `Vec<#native>` values buffer
/// and `MutableBitmap` validity buffer for the `Option<numeric>` fast path.
/// Caller must gate on `is_direct_primitive_array_option_numeric_leaf`. The
/// dtype is reconstructed via `<#native as NativeType>::PRIMITIVE.into()`
/// (matching what `MutablePrimitiveArray::with_capacity` does internally),
/// so the produced `PrimitiveArray`'s dtype matches the schema's leaf dtype
/// exactly without a post-finish cast. Conversion is zero-copy on values
/// (`Vec<T>` -> `Buffer<T>`) and on validity (`MutableBitmap` ->
/// `Option<Bitmap>` collapses to `None` when no unset bits, preserving the
/// no-null fast path).
fn gen_option_numeric_direct_array_expr(base: &BaseType, idx: usize) -> TokenStream {
    let buf_ident = PopulatorIdents::primitive_buf(idx);
    let validity_ident = PopulatorIdents::primitive_validity(idx);
    let native = numeric_native_rust_type(base);
    let pa_root = super::polars_paths::polars_arrow_root();
    quote! {
        #pa_root::array::PrimitiveArray::<#native>::new(
            <#native as #pa_root::types::NativeType>::PRIMITIVE.into(),
            #buf_ident.into(),
            ::std::convert::Into::<::std::option::Option<#pa_root::bitmap::Bitmap>>::into(
                #validity_ident,
            ),
        )
    }
}

/// Buffer-pair declarations for the `Option<String>` direct fast path: a
/// `MutableBinaryViewArray<str>` for values and a `MutableBitmap` for
/// validity, both pre-sized to `items.len()`. The validity bitmap is
/// pre-filled with `true` so the per-row hot path only needs to flip
/// individual bits to `false` for `None` rows — most rows in real data are
/// `Some`, so the common case is zero validity work per row. Caller must
/// gate on `is_direct_view_option_string_leaf`.
fn gen_option_string_direct_decls(idx: usize) -> Vec<TokenStream> {
    let buf_ident = PopulatorIdents::primitive_buf(idx);
    let validity_ident = PopulatorIdents::primitive_validity(idx);
    let row_idx = PopulatorIdents::primitive_row_idx(idx);
    let pa_root = super::polars_paths::polars_arrow_root();
    vec![
        quote! {
            let mut #buf_ident: #pa_root::array::MutableBinaryViewArray<str> =
                #pa_root::array::MutableBinaryViewArray::<str>::with_capacity(items.len());
        },
        quote! {
            let mut #validity_ident: #pa_root::bitmap::MutableBitmap = {
                let mut __df_derive_b = #pa_root::bitmap::MutableBitmap::with_capacity(items.len());
                __df_derive_b.extend_constant(items.len(), true);
                __df_derive_b
            };
        },
        quote! {
            let mut #row_idx: usize = 0;
        },
    ]
}

/// Per-row push tokens for the `Option<String>` direct fast path. Caller
/// must gate on `is_direct_view_option_string_leaf`. The `Some` branch is
/// the common case and only pushes the value's `&str` to the view array;
/// the validity bitmap was pre-filled with `true` at decl time, so no
/// validity-side work is needed. The `None` branch pushes an empty
/// placeholder string (`push_value_into_buffer` resolves it to an inline
/// zero-length `View` — no buffer allocation, no copy) and unsets the
/// pre-filled validity bit at the row's index via the safe `set` (a
/// bounds-checked single byte write, cheaper than the per-row
/// `MutableBitmap::push`'s `is_multiple_of(8)` check + `Vec::push`). The
/// safe variant is used so no `unsafe` block lands inside the user's
/// `Columnar::columnar_from_refs` impl method, which would trip
/// `clippy::unsafe_derive_deserialize` for any downstream struct that pairs
/// `#[derive(ToDataFrame, Deserialize)]`. The bounds check is a single
/// well-predicted compare against a loop-invariant length, so it doesn't
/// disturb the inner-loop throughput in practice.
///
/// Both pushes use `push_value_ignore_validity`, which skips the per-row
/// `if let Some(validity) ... validity.push(true)` branch `push_value`
/// would otherwise do. The view array's internal validity is never
/// initialized on this path (we track validity externally via the
/// `MutableBitmap`, never calling `push_null` / `init_validity`), so the
/// branch would always take the `None` arm anyway — eliminating it
/// roughly doubles the gain over the `Vec<Option<&str>>` slow path.
///
/// Matches on `&(#access)` so the `Some` arm binds `__df_derive_v: &String`
/// and reaches the value-push directly — one branch, no detour through
/// `Option::as_deref`.
fn gen_option_string_direct_push(access: &TokenStream, idx: usize) -> TokenStream {
    let buf_ident = PopulatorIdents::primitive_buf(idx);
    let validity_ident = PopulatorIdents::primitive_validity(idx);
    let row_idx = PopulatorIdents::primitive_row_idx(idx);
    quote! {
        match &(#access) {
            ::std::option::Option::Some(__df_derive_v) => {
                #buf_ident.push_value_ignore_validity(__df_derive_v.as_str());
            }
            ::std::option::Option::None => {
                #buf_ident.push_value_ignore_validity("");
                #validity_ident.set(#row_idx, false);
            }
        }
        #row_idx += 1;
    }
}

/// Build a `Utf8ViewArray` from the values buffer and `MutableBitmap`
/// validity buffer for the `Option<String>` fast path. Caller must gate on
/// `is_direct_view_option_string_leaf`. Conversion via
/// `Option<Bitmap>::from(MutableBitmap)` collapses to `None` when no bits
/// are unset, preserving the no-null fast path; `Utf8ViewArray::with_validity`
/// then attaches the validity to the frozen array in place.
fn gen_option_string_direct_array_expr(idx: usize) -> TokenStream {
    let buf_ident = PopulatorIdents::primitive_buf(idx);
    let validity_ident = PopulatorIdents::primitive_validity(idx);
    let pa_root = super::polars_paths::polars_arrow_root();
    quote! {
        #buf_ident.freeze().with_validity(
            ::std::convert::Into::<::std::option::Option<#pa_root::bitmap::Bitmap>>::into(
                #validity_ident,
            ),
        )
    }
}

/// Columnar finish tokens for the bare-`String` and bare-`as_string`
/// direct fast paths. Both share the same single-buffer layout
/// (`MutableBinaryViewArray<str>` only — no validity is ever initialized)
/// so the finisher is identical: freeze the values into a `Utf8ViewArray`
/// and wrap in `StringChunked::with_chunk`. Returns `None` for any shape
/// that doesn't match either carve-out.
pub(super) fn gen_bare_view_string_direct_columnar_finish(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
    idx: usize,
    name: &str,
) -> Option<TokenStream> {
    let bare_to_string = is_direct_view_to_string_leaf(transform, wrappers) && !has_option(wrappers);
    if !is_direct_view_string_leaf(base, transform, wrappers) && !bare_to_string {
        return None;
    }
    let pp = super::polars_paths::prelude();
    let buf_ident = PopulatorIdents::primitive_buf(idx);
    Some(quote! {{
        let s = #pp::IntoSeries::into_series(
            #pp::StringChunked::with_chunk(#name.into(), #buf_ident.freeze()),
        );
        columns.push(s.into());
    }})
}

/// Columnar finish tokens for the `Option<String>` and `Option<as_string>`
/// direct fast paths. Both share the same split-buffer layout
/// (`MutableBinaryViewArray<str>` + parallel `MutableBitmap`) so the
/// finisher is identical: freeze the values, attach the bitmap via
/// `Utf8ViewArray::with_validity` (collapsing to `None` when no bits are
/// unset), and wrap in `StringChunked::with_chunk`. Returns `None` for any
/// shape that doesn't match either carve-out.
pub(super) fn gen_option_string_direct_columnar_finish(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
    idx: usize,
    name: &str,
) -> Option<TokenStream> {
    let is_option_to_string =
        is_direct_view_to_string_leaf(transform, wrappers) && has_option(wrappers);
    if !is_direct_view_option_string_leaf(base, transform, wrappers) && !is_option_to_string {
        return None;
    }
    let pp = super::polars_paths::prelude();
    let arr_expr = gen_option_string_direct_array_expr(idx);
    Some(quote! {{
        let s = #pp::IntoSeries::into_series(
            #pp::StringChunked::with_chunk(#name.into(), { #arr_expr }),
        );
        columns.push(s.into());
    }})
}

/// Columnar finish tokens for the bare and `Option<numeric>` direct fast
/// paths. Returns `None` for any shape that doesn't match either carve-out.
///
/// - Bare numeric (no Option, no Vec, no transform): consume the
///   `Vec<#native>` zero-copy via `<*Chunked>::from_vec(name, buf)`. See
///   `is_direct_primitive_array_numeric_leaf`.
/// - `Option<numeric>` (exactly `[Option]`, no transform): build a
///   `PrimitiveArray::new` from the `Vec<#native>` + `MutableBitmap` pair
///   and wrap in `<*Chunked>::with_chunk(name, arr)`. See
///   `is_direct_primitive_array_option_numeric_leaf`.
pub(super) fn gen_numeric_direct_columnar_finish(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
    idx: usize,
    name: &str,
) -> Option<TokenStream> {
    let pp = super::polars_paths::prelude();
    let chunked = if is_direct_primitive_array_numeric_leaf(base, transform, wrappers)
        || is_direct_primitive_array_option_numeric_leaf(base, transform, wrappers)
    {
        numeric_chunked_type(base)
    } else {
        return None;
    };
    if is_direct_primitive_array_numeric_leaf(base, transform, wrappers) {
        let buf_ident = PopulatorIdents::primitive_buf(idx);
        return Some(quote! {{
            let s = #pp::IntoSeries::into_series(
                #chunked::from_vec(#name.into(), #buf_ident),
            );
            columns.push(s.into());
        }});
    }
    let arr_expr = gen_option_numeric_direct_array_expr(base, idx);
    Some(quote! {{
        let s = #pp::IntoSeries::into_series(
            #chunked::with_chunk(#name.into(), { #arr_expr }),
        );
        columns.push(s.into());
    }})
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
    // Direct `MutableBinaryViewArray<str>` accumulation for top-level
    // (non-Option) `String`: the buffer copies each row's bytes straight
    // into the view array, skipping a `Vec<&str>` round-trip and the second
    // walk `Series::new(&Vec<&str>)` would do via `from_slice_values`.
    //
    // `push_value_ignore_validity` skips the per-row `if let Some(validity)`
    // branch `push_value` does internally; the buffer's internal validity is
    // never initialized on this path (no `push_null` / `init_validity`), so
    // the branch would always take the `None` arm anyway.
    if is_direct_view_string_leaf(base_type, transform, wrappers) {
        let buf_ident = PopulatorIdents::primitive_buf(idx);
        return quote! { #buf_ident.push_value_ignore_validity((#access).as_str()); };
    }

    // Direct `MutableBinaryViewArray<str>` + pre-filled `MutableBitmap`
    // accumulation for top-level `Option<String>` (no transform). Splits
    // values vs validity into separate buffers — same shape as the
    // `Option<numeric>` fast path — and the bitmap is pre-filled to all-
    // `true` at decl time so `Some` rows do zero validity work and `None`
    // rows just flip a single bit via the safe `MutableBitmap::set`.
    // Avoids the per-row `validity.is_some()` branch
    // `MutableBinaryViewArray::push_value` would do internally, which a
    // single-buffer `push_value`/`push_null` attempt regressed by ~7-8% on.
    if is_direct_view_option_string_leaf(base_type, transform, wrappers) {
        return gen_option_string_direct_push(access, idx);
    }

    // Direct `MutableBinaryViewArray<str>` accumulation for top-level
    // `as_string` leaves: clear the reused `String` scratch, run
    // `Display::fmt` into it, then push the resulting `&str` (the view
    // array copies the bytes). Replaces `items.len()` per-row `String`
    // allocations with the scratch's amortized growth.
    if is_direct_view_to_string_leaf(transform, wrappers) {
        return gen_direct_view_to_string_push(access, wrappers, idx);
    }

    // Direct `Vec<#native>` + `MutableBitmap` accumulation for top-level
    // `Option<numeric>` (no transform): the buffer holds either the row's
    // value (if Some) or `<#native>::default()` (if None), and the bitmap
    // records validity. Splitting value vs validity into two independent
    // pushes lets the compiler vectorize cleanly, and avoids the per-row
    // "is validity active?" branch inside `MutablePrimitiveArray::push`.
    if is_direct_primitive_array_option_numeric_leaf(base_type, transform, wrappers) {
        return gen_option_numeric_direct_push(access, base_type, idx);
    }

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

    // Direct fast path for top-level (non-Option) `String` (see
    // `is_direct_view_string_leaf`): accumulate straight into the view array.
    if is_direct_view_string_leaf(base_type, transform, wrappers) {
        let buf_ident = PopulatorIdents::primitive_buf(idx);
        let pa_root = super::polars_paths::polars_arrow_root();
        decls.push(quote! {
            let mut #buf_ident: #pa_root::array::MutableBinaryViewArray<str> =
                #pa_root::array::MutableBinaryViewArray::<str>::with_capacity(items.len());
        });
        return decls;
    }

    // Direct fast path for top-level `Option<String>` (see
    // `is_direct_view_option_string_leaf`): split-buffer pair —
    // `MutableBinaryViewArray<str>` for values + `MutableBitmap` for
    // validity. Finalized via `Utf8ViewArray::with_validity`.
    if is_direct_view_option_string_leaf(base_type, transform, wrappers) {
        decls.extend(gen_option_string_direct_decls(idx));
        return decls;
    }

    // Direct fast path for top-level `as_string` leaves (see
    // `is_direct_view_to_string_leaf`): accumulate into a view array, with a
    // reused `String` scratch that each row clears and writes Display::fmt
    // into. The view array's `push_value_ignore_validity` copies the bytes,
    // so reusing the scratch on the next row is sound.
    //
    // The `[Option]` shape also declares a parallel `MutableBitmap` pre-
    // filled to all-`true` plus a row counter — same split-buffer layout
    // as `is_direct_view_option_string_leaf` — so per-row pushes can use
    // `push_value_ignore_validity` (skipping the buffer's internal
    // `if let Some(validity)` branch) and `None` rows just flip a single
    // bit via the safe `MutableBitmap::set`.
    if is_direct_view_to_string_leaf(transform, wrappers) {
        let buf_ident = PopulatorIdents::primitive_buf(idx);
        let str_ident = PopulatorIdents::primitive_str_scratch(idx);
        let pa_root = super::polars_paths::polars_arrow_root();
        decls.push(quote! {
            let mut #buf_ident: #pa_root::array::MutableBinaryViewArray<str> =
                #pa_root::array::MutableBinaryViewArray::<str>::with_capacity(items.len());
        });
        decls.push(quote! {
            let mut #str_ident: ::std::string::String = ::std::string::String::new();
        });
        if matches!(wrappers, [Wrapper::Option]) {
            let validity_ident = PopulatorIdents::primitive_validity(idx);
            let row_idx = PopulatorIdents::primitive_row_idx(idx);
            decls.push(quote! {
                let mut #validity_ident: #pa_root::bitmap::MutableBitmap = {
                    let mut __df_derive_b = #pa_root::bitmap::MutableBitmap::with_capacity(items.len());
                    __df_derive_b.extend_constant(items.len(), true);
                    __df_derive_b
                };
            });
            decls.push(quote! {
                let mut #row_idx: usize = 0;
            });
        }
        return decls;
    }

    // Direct fast path for top-level `Option<numeric>` (see
    // `is_direct_primitive_array_option_numeric_leaf`).
    if is_direct_primitive_array_option_numeric_leaf(base_type, transform, wrappers) {
        decls.extend(gen_option_numeric_direct_decls(base_type, idx));
        return decls;
    }

    // Borrowing fast path for any base type with `as_str` (`AsRef<str>` impl):
    // a `Vec<&str>` (or `Vec<Option<&str>>`) buffer borrows from `items`
    // instead of cloning each row's `String`. `Series::new(name, &Vec<&str>)`
    // dispatches to `StringChunked::from_slice` and produces the same
    // `Utf8ViewArray`-backed column the owning path produces.
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
    } else if is_direct_view_string_leaf(base_type, transform, wrappers)
        || (is_direct_view_to_string_leaf(transform, wrappers) && !has_option(wrappers))
    {
        // Direct `MutableBinaryViewArray<str>` accumulation (see
        // `primitive_decls`): freeze into a `Utf8ViewArray`, wrap as a
        // `StringChunked`, convert to Series — same dtype as the
        // `Series::new(&Vec<&str>)` path, minus the second walk through the
        // intermediate `Vec<&str>`. Covers the bare-`String` fast path
        // (`is_direct_view_string_leaf`) and the bare-`as_string` Display
        // fast path (`is_direct_view_to_string_leaf` with no Option). The
        // `Option<as_string>` shape is handled below alongside the
        // `Option<String>` split-buffer pattern.
        let buf_ident = PopulatorIdents::primitive_buf(idx);
        quote! {
            let inner = #pp::IntoSeries::into_series(
                #pp::StringChunked::with_chunk("".into(), #buf_ident.freeze()),
            );
            out_values.push(#pp::AnyValue::List(inner));
        }
    } else if is_direct_view_option_string_leaf(base_type, transform, wrappers)
        || (is_direct_view_to_string_leaf(transform, wrappers) && has_option(wrappers))
    {
        // `Option<String>` and `Option<as_string>` direct fast paths share
        // the same split-buffer layout: a `MutableBinaryViewArray<str>`
        // values buffer with a parallel `MutableBitmap` validity. Freeze
        // the values, attach the bitmap via `Utf8ViewArray::with_validity`
        // (collapsing to `None` if no unset bits), wrap in
        // `StringChunked::with_chunk` — same column dtype as the
        // `Series::new(&Vec<Option<&str>>)` slow path, minus the second
        // walk and the per-row validity branch.
        let arr_expr = gen_option_string_direct_array_expr(idx);
        quote! {
            let inner = #pp::IntoSeries::into_series(
                #pp::StringChunked::with_chunk("".into(), { #arr_expr }),
            );
            out_values.push(#pp::AnyValue::List(inner));
        }
    } else if is_direct_primitive_array_numeric_leaf(base_type, transform, wrappers) {
        // Bare numeric primitive (no transform, no wrappers): the buffer is
        // `Vec<Native>`. Consume it via `<*Chunked>::from_vec` (zero-copy
        // through `to_primitive`) instead of `Series::new(&buf)`'s
        // `from_slice` + `memcpy`. Same dtype — the `*Chunked` alias's
        // static dtype matches the schema's leaf dtype, no cast needed.
        let buf_ident = PopulatorIdents::primitive_buf(idx);
        let chunked = numeric_chunked_type(base_type);
        quote! {
            let inner = #pp::IntoSeries::into_series(
                #chunked::from_vec("".into(), #buf_ident),
            );
            out_values.push(#pp::AnyValue::List(inner));
        }
    } else if is_direct_primitive_array_option_numeric_leaf(base_type, transform, wrappers) {
        // `Option<numeric>` direct fast path: the buffer is `Vec<#native>`
        // with parallel `MutableBitmap` validity. Build a `PrimitiveArray`
        // directly (zero-copy via `Buffer<T>::from(Vec<T>)` and `Option<
        // Bitmap>::from(MutableBitmap)`, which collapses to `None` if no
        // unset bits) and wrap in `<*Chunked>::with_chunk` — same column
        // dtype as the `Series::new(&Vec<Option<T>>)` slow path, minus the
        // second walk and the per-row validity branch.
        let arr_expr = gen_option_numeric_direct_array_expr(base_type, idx);
        let chunked = numeric_chunked_type(base_type);
        quote! {
            let inner = #pp::IntoSeries::into_series(
                #chunked::with_chunk("".into(), { #arr_expr }),
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

/// Native rust + leaf `polars::prelude::DataType` token tree for the
/// `Vec<Vec<T>>` numeric-primitive fast path. Returned together because
/// every emit site needs both: the native splices into
/// `PrimitiveArray::<T>::from_vec` and the flat `Vec<T>` decl, and the leaf
/// dtype splices into the outer Series's logical `List<leaf>` wrap.
struct NestedNumericPrimitive {
    native_rust: TokenStream,
    leaf_dtype: TokenStream,
}

/// Eligible-shape probe for the bulk `Vec<Vec<T>>` numeric-primitive emit.
/// Returns `Some` only when:
/// - Wrappers exactly `[Vec, Vec]` (no Option layers, no transform).
/// - Base is a bare numeric primitive
///   (`i8/i16/i32/i64/u8/u16/u32/u64/f32/f64`).
///
/// Other shapes — `Vec<Vec<Option<T>>>`, `Option`-wrapped variants, strings,
/// datetimes, decimals, bool, isize/usize, anything with a transform — keep
/// the existing typed-`ListBuilder` per-row push.
fn nested_numeric_primitive(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> Option<NestedNumericPrimitive> {
    if !matches!(wrappers, [Wrapper::Vec, Wrapper::Vec]) || transform.is_some() {
        return None;
    }
    let pp = super::polars_paths::prelude();
    let (native_rust, leaf_dtype) = match base {
        BaseType::I8 => (quote! { i8 }, quote! { #pp::DataType::Int8 }),
        BaseType::I16 => (quote! { i16 }, quote! { #pp::DataType::Int16 }),
        BaseType::I32 => (quote! { i32 }, quote! { #pp::DataType::Int32 }),
        BaseType::I64 => (quote! { i64 }, quote! { #pp::DataType::Int64 }),
        BaseType::U8 => (quote! { u8 }, quote! { #pp::DataType::UInt8 }),
        BaseType::U16 => (quote! { u16 }, quote! { #pp::DataType::UInt16 }),
        BaseType::U32 => (quote! { u32 }, quote! { #pp::DataType::UInt32 }),
        BaseType::U64 => (quote! { u64 }, quote! { #pp::DataType::UInt64 }),
        BaseType::F32 => (quote! { f32 }, quote! { #pp::DataType::Float32 }),
        BaseType::F64 => (quote! { f64 }, quote! { #pp::DataType::Float64 }),
        // Bool is excluded: validity bit semantics differ from numeric leaves
        // and the all-non-null case would still need a separate path. Other
        // bases (String, DateTime, Decimal, ISize/USize, Struct/Generic) keep
        // the per-row typed-`ListBuilder` path.
        BaseType::Bool
        | BaseType::String
        | BaseType::ISize
        | BaseType::USize
        | BaseType::DateTimeUtc
        | BaseType::Decimal
        | BaseType::Struct(..)
        | BaseType::Generic(_) => return None,
    };
    Some(NestedNumericPrimitive {
        native_rust,
        leaf_dtype,
    })
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
    let native = &info.native_rust;
    let leaf_dtype = &info.leaf_dtype;
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

/// Eligible-shape probe for the bulk `Vec<Option<T>>` numeric-primitive
/// emit. Returns `true` only when wrappers are exactly `[Vec, Option]`, the
/// base is a bare numeric primitive (`i8/i16/i32/i64/u8/u16/u32/u64/f32/f64`),
/// and there is no transform. Other shapes — `Vec<Option<bool>>`,
/// `Vec<Option<String>>`, `Vec<Option<DateTime>>`, `Vec<Option<Decimal>>`,
/// `Option<Vec<Option<T>>>`, `Vec<Vec<Option<T>>>` — keep the typed-
/// `ListPrimitiveChunkedBuilder` per-row path: bool would need a validity
/// bit per element regardless, the string / datetime / decimal shapes need
/// transforms or different storage, and any extra wrapper level changes the
/// flattening invariants this emitter relies on.
pub(super) const fn is_direct_vec_option_numeric_leaf(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> bool {
    if transform.is_some() || !matches!(wrappers, [Wrapper::Vec, Wrapper::Option]) {
        return false;
    }
    matches!(
        base,
        BaseType::I8
            | BaseType::I16
            | BaseType::I32
            | BaseType::I64
            | BaseType::U8
            | BaseType::U16
            | BaseType::U32
            | BaseType::U64
            | BaseType::F32
            | BaseType::F64
    )
}

/// Native rust + leaf `polars::prelude::DataType` token tree for the
/// `Vec<Option<T>>` numeric-primitive fast path. Caller must gate on
/// `is_direct_vec_option_numeric_leaf` so the non-numeric arms are
/// statically unreachable.
fn vec_option_numeric_leaf_types(base: &BaseType) -> (TokenStream, TokenStream) {
    let pp = super::polars_paths::prelude();
    match base {
        BaseType::I8 => (quote! { i8 }, quote! { #pp::DataType::Int8 }),
        BaseType::I16 => (quote! { i16 }, quote! { #pp::DataType::Int16 }),
        BaseType::I32 => (quote! { i32 }, quote! { #pp::DataType::Int32 }),
        BaseType::I64 => (quote! { i64 }, quote! { #pp::DataType::Int64 }),
        BaseType::U8 => (quote! { u8 }, quote! { #pp::DataType::UInt8 }),
        BaseType::U16 => (quote! { u16 }, quote! { #pp::DataType::UInt16 }),
        BaseType::U32 => (quote! { u32 }, quote! { #pp::DataType::UInt32 }),
        BaseType::U64 => (quote! { u64 }, quote! { #pp::DataType::UInt64 }),
        BaseType::F32 => (quote! { f32 }, quote! { #pp::DataType::Float32 }),
        BaseType::F64 => (quote! { f64 }, quote! { #pp::DataType::Float64 }),
        BaseType::Bool
        | BaseType::String
        | BaseType::ISize
        | BaseType::USize
        | BaseType::DateTimeUtc
        | BaseType::Decimal
        | BaseType::Struct(..)
        | BaseType::Generic(_) => unreachable!(
            "vec_option_numeric_leaf_types called for non-numeric base; \
             callers must gate on is_direct_vec_option_numeric_leaf"
        ),
    }
}

/// Returns `Some(emit)` for the columnar / vec-anyvalues bulk fast path on
/// `Vec<Option<T>>` over a bare numeric primitive base (no transform).
/// `None` for any other shape — caller falls through to the typed
/// `ListPrimitiveChunkedBuilder<Native>::append_iter` per-row path.
///
/// The block scans `items` once, computing the total leaf count, then
/// allocates a flat `Vec<Native>` and a parallel `MutableBitmap` pre-filled
/// with `true` at that capacity. Each `Some(v)` row pushes the value (the
/// validity bit is already set); each `None` row pushes `<Native>::default()`
/// as a placeholder and flips the corresponding bit via the safe
/// `MutableBitmap::set` (a bounds-checked single-byte write — cheaper than
/// the typed-builder's per-element `MutablePrimitiveArray::push(Option<T>)`
/// branching on the validity-active flag and the discriminant). The
/// finisher builds a `PrimitiveArray::<Native>::new(dtype, flat.into(),
/// Some(validity.into()))` — both conversions are zero-copy: `Vec<T> ->
/// Buffer<T>` and `MutableBitmap -> Option<Bitmap>` (the latter collapses
/// to `None` when no bits were unset, preserving the no-null fast path) —
/// wraps it in a `LargeListArray` partitioned by the per-row offsets, and
/// consumes the array via the in-scope free helper
/// `__df_derive_assemble_list_series_unchecked` so the resulting Series's
/// dtype is `List<leaf>` exactly (no post-finish cast).
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
) -> Option<TokenStream> {
    if !is_direct_vec_option_numeric_leaf(base, transform, wrappers) {
        return None;
    }
    let pp = super::polars_paths::prelude();
    let (native, leaf_dtype) = vec_option_numeric_leaf_types(base);
    let series_block = quote! {{
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
                        __df_derive_flat.push(*__df_derive_v);
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

/// Returns `Some(emit)` for the columnar / vec-anyvalues bulk fast path on
/// `Vec<bool>` (wrappers exactly `[Vec]`, base `Bool`, no transform). `None`
/// for any other shape — caller falls back to the boxed-dyn
/// `ListBuilderTrait::append_series` per-row path.
///
/// Sibling shapes (`Vec<Option<bool>>`, `Option<Vec<bool>>`, `Vec<Vec<bool>>`)
/// keep the existing path: the `[Vec, Option]` and `[Option, Vec]` cases
/// would need a validity bitmap that this emitter intentionally omits, and
/// the numeric `Vec<Vec<T>>` fast path explicitly excludes Bool because of
/// the validity-bit cost it would re-introduce there.
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

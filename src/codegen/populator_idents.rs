use quote::format_ident;
use syn::Ident;

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

    /// `MutableBitmap` validity buffer for the
    /// `is_direct_primitive_array_option_numeric_leaf` fast path. Paired
    /// with `primitive_buf` (which holds `Vec<#native>` on that path) so the
    /// finisher can build a `PrimitiveArray::new(dtype, vals, validity)`
    /// directly without a `Vec<Option<T>>` second walk.
    pub(super) fn primitive_validity(idx: usize) -> Ident {
        format_ident!("__df_derive_val_{}", idx)
    }

    /// Reused `String` scratch buffer for the
    /// `is_direct_view_to_string_leaf` fast path. Paired with `primitive_buf`
    /// (which holds `MutableBinaryViewArray<str>` on that path) so each row
    /// can clear-and-write into the scratch via `Display::fmt` and then push
    /// the resulting `&str` into the view array (which copies the bytes),
    /// avoiding a fresh per-row `String` allocation.
    pub(super) fn primitive_str_scratch(idx: usize) -> Ident {
        format_ident!("__df_derive_str_{}", idx)
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

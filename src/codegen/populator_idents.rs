use quote::format_ident;
use syn::Ident;

/// Per-strategy identifier convention for the primitive populator
/// pipeline. Names declared here in `primitive_decls` are referenced by
/// the per-row push helpers (`generate_primitive_for_columnar_push`) and
/// the columnar builders. Funneling every site through this struct turns
/// rename mistakes into a compile error at the helper itself.
///
/// The nested-struct identifier set was retired in Step 4: every
/// nested-struct/generic shape now routes through `encoder.rs`'s depth-N
/// vec encoder, which manages its own per-shape ident bundle
/// (`NestedIdents` / `NestedLayerIdents`).
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

    /// Row counter for the `is_direct_view_option_string_leaf` fast path.
    /// Indexes the pre-filled `MutableBitmap` so the per-row push only
    /// writes a single byte for `None` rows via `set_unchecked`, instead
    /// of pushing both `true` and `false` bits unconditionally.
    pub(super) fn primitive_row_idx(idx: usize) -> Ident {
        format_ident!("__df_derive_ri_{}", idx)
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
}

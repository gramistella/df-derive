use crate::ir::StructIR;
use proc_macro2::TokenStream;
use quote::quote;

/// Generates the `Columnar` trait impl as a thin shim that delegates to
/// `__df_derive_columnar_from_refs` (an inherent helper emitted by
/// `helpers::generate_helpers_impl`).
///
/// The split exists so that the bulk-vec emitter for non-generic
/// `Vec<Struct>` fields can flatten into a `Vec<&Inner>` (no `Inner: Clone`
/// required) and call the inherent helper directly. The trait method handles
/// the slightly-more-common `&[Self]` shape by collecting references on the
/// fly. The Vec-of-pointers allocation costs N usize per call — negligible
/// next to the actual columnar work, and well worth the avoided
/// per-element clone in the bulk-vec path.
pub fn generate_columnar_impl(ir: &StructIR, config: &super::MacroConfig) -> TokenStream {
    let struct_name = &ir.name;
    let columnar_trait = &config.columnar_trait_path;
    let (impl_generics, ty_generics, where_clause) =
        super::impl_parts_with_bounds(&ir.generics, config);

    quote! {
        impl #impl_generics #columnar_trait for #struct_name #ty_generics #where_clause {
            fn columnar_to_dataframe(items: &[Self]) -> polars::prelude::PolarsResult<polars::prelude::DataFrame> {
                let __df_derive_refs: ::std::vec::Vec<&Self> = items.iter().collect();
                Self::__df_derive_columnar_from_refs(&__df_derive_refs)
            }
        }
    }
}

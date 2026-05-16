use crate::ir::{AccessChain, AccessStep};
use proc_macro2::TokenStream;
use quote::quote;

use super::idents;

pub(in crate::codegen) fn idx_size_len_expr(flat: &syn::Ident, pp: &TokenStream) -> TokenStream {
    quote! {
        <#pp::IdxSize as ::core::convert::TryFrom<usize>>::try_from(#flat.len())
            .map_err(|_| #pp::polars_err!(
                ComputeError:
                "df-derive: nested inner-option position {} exceeds IdxSize range",
                #flat.len(),
            ))?
    }
}

pub(in crate::codegen) fn list_offset_i64_expr(
    offset: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    quote! {
        <i64 as ::core::convert::TryFrom<usize>>::try_from(#offset)
            .map_err(|_| #pp::polars_err!(
                ComputeError:
                "df-derive: list offset {} exceeds i64 range",
                #offset,
            ))?
    }
}

pub(in crate::codegen) fn collapse_options_to_ref(base: &TokenStream, n: usize) -> TokenStream {
    if n == 0 {
        return base.clone();
    }
    let param = idents::collapse_option_param();
    let mut out = quote! { (#base).as_ref() };
    for _ in 1..n {
        out = quote! { #out.and_then(|#param| #param.as_ref()) };
    }
    out
}

pub(in crate::codegen) struct ChainRef {
    pub expr: TokenStream,
    pub has_option: bool,
}

fn deref_ref_expr(base_ref: &TokenStream, smart_ptrs: usize) -> TokenStream {
    if smart_ptrs == 0 {
        return base_ref.clone();
    }
    let mut out = quote! { *(#base_ref) };
    for _ in 0..smart_ptrs {
        out = quote! { *(#out) };
    }
    quote! { (&(#out)) }
}

fn apply_pending_smart_ptrs(expr: TokenStream, has_option: bool, smart_ptrs: usize) -> TokenStream {
    if smart_ptrs == 0 {
        return expr;
    }
    if has_option {
        let param = idents::collapse_option_param();
        let derefed = deref_ref_expr(&quote! { #param }, smart_ptrs);
        quote! { (#expr).map(|#param| #derefed) }
    } else {
        deref_ref_expr(&expr, smart_ptrs)
    }
}

pub(in crate::codegen) fn access_chain_to_ref(base: &TokenStream, chain: &AccessChain) -> ChainRef {
    if chain.is_empty() {
        return ChainRef {
            expr: base.clone(),
            has_option: false,
        };
    }
    if chain.is_only_options() {
        let option_layers = chain.option_layers();
        return ChainRef {
            expr: if option_layers == 1 {
                base.clone()
            } else {
                collapse_options_to_ref(base, option_layers)
            },
            has_option: option_layers > 0,
        };
    }

    let mut expr = base.clone();
    let mut has_option = false;
    let mut pending_smart_ptrs = 0usize;

    for step in &chain.steps {
        match step {
            AccessStep::SmartPtr => {
                pending_smart_ptrs += 1;
            }
            AccessStep::Option => {
                expr = apply_pending_smart_ptrs(expr, has_option, pending_smart_ptrs);
                pending_smart_ptrs = 0;
                let param = idents::collapse_option_param();
                expr = if has_option {
                    quote! { (#expr).and_then(|#param| (#param).as_ref()) }
                } else {
                    has_option = true;
                    quote! { (#expr).as_ref() }
                };
            }
        }
    }

    expr = apply_pending_smart_ptrs(expr, has_option, pending_smart_ptrs);
    ChainRef { expr, has_option }
}

pub(in crate::codegen) fn access_chain_to_option_ref(
    base: &TokenStream,
    chain: &AccessChain,
) -> TokenStream {
    debug_assert!(chain.option_layers() > 0);
    if chain.is_only_options() {
        collapse_options_to_ref(base, chain.option_layers())
    } else {
        access_chain_to_ref(base, chain).expr
    }
}

use crate::ir::Wrapper;
use proc_macro2::TokenStream;
use quote::quote;

/// Centralized generator that traverses wrapper layers (Option, Vec) and delegates behavior.
///
/// Parameters:
/// - `access`: token stream that evaluates to the current value expression
/// - `wrappers`: slice of wrappers from outermost to innermost
/// - `on_leaf`: invoked when no wrappers remain; receives the current access expression
/// - `on_option_none`: invoked for an Option None branch; receives the remaining tail wrappers
/// - `on_vec`: invoked when encountering a Vec; receives the current access expression and the remaining tail wrappers
pub fn process_wrappers<FL, FN, FV>(
    access: &TokenStream,
    wrappers: &[Wrapper],
    on_leaf: &FL,
    on_option_none: &FN,
    on_vec: &FV,
) -> TokenStream
where
    FL: Fn(&TokenStream) -> TokenStream,
    FN: Fn(&[Wrapper]) -> TokenStream,
    FV: Fn(&TokenStream, &[Wrapper]) -> TokenStream,
{
    fn recur<FL, FN, FV>(
        access: &TokenStream,
        wrappers: &[Wrapper],
        on_leaf: &FL,
        on_option_none: &FN,
        on_vec: &FV,
    ) -> TokenStream
    where
        FL: Fn(&TokenStream) -> TokenStream,
        FN: Fn(&[Wrapper]) -> TokenStream,
        FV: Fn(&TokenStream, &[Wrapper]) -> TokenStream,
    {
        if let Some((head, tail)) = wrappers.split_first() {
            match head {
                Wrapper::Option => {
                    let inner_ident = syn::Ident::new(
                        "__df_derive_wrapper_inner",
                        proc_macro2::Span::call_site(),
                    );
                    let inner_access = quote! { #inner_ident };
                    let some_branch = recur(&inner_access, tail, on_leaf, on_option_none, on_vec);
                    let none_branch = on_option_none(tail);
                    quote! {
                        match &(#access) {
                            Some(#inner_ident) => { #some_branch }
                            None => { #none_branch }
                        }
                    }
                }
                Wrapper::Vec => on_vec(access, tail),
            }
        } else {
            on_leaf(access)
        }
    }

    recur(access, wrappers, on_leaf, on_option_none, on_vec)
}

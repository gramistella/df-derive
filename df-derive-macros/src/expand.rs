//! Code generation: one builder struct per derived type.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::ext::IdentExt as _;
use syn::spanned::Spanned as _;
use syn::{Data, DeriveInput, Fields, GenericParam, Index, Member};

use crate::attrs::{self, Flatten};
use crate::mirror;

/// One struct field that produces columns.
struct Column<'a> {
    /// How the field is read from the row: `.name` or `.0`.
    member: Member,
    /// The matching field of the generated builder struct.
    slot: syn::Ident,
    ty: &'a syn::Type,
    builder: TokenStream,
    /// The where-clause predicate that makes `builder` usable for `ty`.
    bound: TokenStream,
    /// The column path segment this field adds, if any.
    segment: Option<syn::LitStr>,
}

fn columns<'a>(
    fields: &'a Fields,
    runtime: &TokenStream,
    builders: &TokenStream,
) -> syn::Result<Vec<Column<'a>>> {
    let mut columns = Vec::with_capacity(fields.len());
    for (index, field) in fields.iter().enumerate() {
        let attrs = attrs::field(&field.attrs)?;
        if attrs.skip.is_some() {
            continue;
        }

        let (member, slot, label) = match &field.ident {
            Some(ident) => (
                Member::Named(ident.clone()),
                ident.clone(),
                ident.unraw().to_string(),
            ),
            None => (
                Member::Unnamed(Index::from(index)),
                format_ident!("field_{index}"),
                format!("field_{index}"),
            ),
        };
        let ty = &field.ty;

        let (builder, bound) = match &attrs.encoding {
            None => (
                quote! { <#ty as #runtime::Columnar>::Builder },
                quote! { #ty: #runtime::Columnar },
            ),
            Some((encoding, span)) => {
                if !matches!(encoding, attrs::Encoding::With(_)) && mirror::leaf_is_tuple(ty) {
                    return Err(syn::Error::new(
                        *span,
                        "encoding attributes cannot be applied to a tuple; \
                         move the value into a named struct and annotate its field",
                    ));
                }
                let builder = mirror::builder(ty, encoding, builders);
                let bound = quote! { #builder: #runtime::ColumnBuilder<#ty> };
                (builder, bound)
            }
        };

        let segment = match attrs.flatten {
            None => Some(syn::LitStr::new(&label, field.span())),
            Some((Flatten::Bare, _)) => None,
            Some((Flatten::Prefix(prefix), _)) => Some(prefix),
        };

        columns.push(Column {
            member,
            slot,
            ty,
            builder,
            bound,
            segment,
        });
    }
    Ok(columns)
}

pub fn expand(input: &DeriveInput) -> syn::Result<TokenStream> {
    let container = attrs::container(&input.attrs)?;
    let krate = &container.krate;
    let runtime = quote! { #krate::dataframe };
    let builders = quote! { #runtime::builders };
    let polars = quote! { #runtime::__private::polars::prelude };
    let bitmap = quote! { #runtime::__private::polars_arrow::bitmap::Bitmap };

    let fields = match &input.data {
        Data::Struct(data) => &data.fields,
        Data::Enum(data) => {
            return Err(syn::Error::new(
                data.enum_token.span(),
                "`ToDataFrame` cannot be derived for enums; \
                 annotate fields of this type with `#[df_derive(as_str)]` or `#[df_derive(as_string)]`",
            ));
        }
        Data::Union(data) => {
            return Err(syn::Error::new(
                data.union_token.span(),
                "`ToDataFrame` cannot be derived for unions",
            ));
        }
    };
    let columns = columns(fields, &runtime, &builders)?;

    let name = &input.ident;
    let vis = &input.vis;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let declared = where_clause.map(|clause| &clause.predicates).into_iter().flatten();
    let bounds = columns.iter().map(|column| &column.bound);
    let where_clause = quote! { where #(#declared,)* #(#bounds,)* };

    // Keeps every generic parameter in use even when its only field is
    // skipped.
    let lifetimes = input.generics.params.iter().filter_map(|param| match param {
        GenericParam::Lifetime(param) => Some(&param.lifetime),
        _ => None,
    });
    let type_params = input.generics.params.iter().filter_map(|param| match param {
        GenericParam::Type(param) => Some(&param.ident),
        _ => None,
    });
    let (marker_field, marker_init) = if input.generics.params.is_empty() {
        (TokenStream::new(), TokenStream::new())
    } else {
        (
            quote! {
                __df_derive_marker:
                    ::core::marker::PhantomData<(#(&#lifetimes (),)* #(fn(&#type_params),)*)>,
            },
            quote! { __df_derive_marker: ::core::marker::PhantomData, },
        )
    };

    let slots: Vec<_> = columns.iter().map(|column| &column.slot).collect();
    let builder_types: Vec<_> = columns.iter().map(|column| &column.builder).collect();
    let as_builder: Vec<_> = columns
        .iter()
        .map(|column| {
            let (builder, ty) = (&column.builder, column.ty);
            quote! { <#builder as #runtime::ColumnBuilder<#ty>> }
        })
        .collect();
    let names: Vec<_> = columns
        .iter()
        .map(|column| match &column.segment {
            Some(segment) => quote! { #runtime::__private::child(&name, #segment) },
            None => quote! { ::core::clone::Clone::clone(&name) },
        })
        .collect();
    let pushes = columns.iter().zip(&as_builder).map(|(column, as_builder)| {
        let (slot, member) = (&column.slot, &column.member);
        let push = quote! { #as_builder::push(&mut self.#slot, &value.#member) };
        match &column.segment {
            Some(segment) => quote! {
                if let ::core::result::Result::Err(error) = #push {
                    return ::core::result::Result::Err(
                        #runtime::__private::in_field(error, #segment),
                    );
                }
            },
            None => quote! { #push?; },
        }
    });

    Ok(quote! {
        const _: () = {
            #[doc(hidden)]
            #[allow(non_snake_case, dead_code)]
            #vis struct __DfDeriveBuilder #impl_generics #where_clause {
                #(#slots: #builder_types,)*
                #marker_field
            }

            #[automatically_derived]
            #[allow(unused_variables, non_snake_case, clippy::inline_always)]
            impl #impl_generics #runtime::ColumnBuilder<#name #ty_generics>
                for __DfDeriveBuilder #ty_generics
            #where_clause
            {
                fn schema(name: #polars::PlSmallStr, out: &mut #runtime::Fields) {
                    #(#as_builder::schema(#names, out);)*
                }

                fn with_capacity(rows: usize) -> Self {
                    Self {
                        #(#slots: #as_builder::with_capacity(rows),)*
                        #marker_init
                    }
                }

                #[inline(always)]
                fn push(&mut self, value: &#name #ty_generics) -> #polars::PolarsResult<()> {
                    #(#pushes)*
                    ::core::result::Result::Ok(())
                }

                #[inline(always)]
                fn push_null(&mut self) {
                    #(#as_builder::push_null(&mut self.#slots);)*
                }

                fn finish(
                    self,
                    name: #polars::PlSmallStr,
                    validity: ::core::option::Option<&#bitmap>,
                    out: &mut ::std::vec::Vec<#polars::Column>,
                ) -> #polars::PolarsResult<()> {
                    #(#as_builder::finish(self.#slots, #names, validity, out)?;)*
                    ::core::result::Result::Ok(())
                }
            }

            #[automatically_derived]
            impl #impl_generics #runtime::Columnar for #name #ty_generics #where_clause {
                type Builder = __DfDeriveBuilder #ty_generics;
            }

            #[automatically_derived]
            impl #impl_generics #runtime::ToDataFrame for #name #ty_generics #where_clause {}
        };
    })
}

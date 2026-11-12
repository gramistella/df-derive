//! Code generation: one builder struct per derived type.

use std::collections::HashSet;

use proc_macro2::{Span, TokenStream, TokenTree};
use quote::{ToTokens as _, format_ident, quote};
use syn::ext::IdentExt as _;
use syn::spanned::Spanned as _;
use syn::{Data, DeriveInput, Fields, GenericParam, Index, Member};

use crate::attrs::{self, ContainerAttrs, Flatten, Via};
use crate::mirror;

/// Paths into the runtime, rooted at the crate the container names.
struct Runtime {
    /// `<crate>::dataframe`
    root: TokenStream,
    /// `<crate>::dataframe::builders`
    builders: TokenStream,
    /// `polars::prelude`, through the runtime's re-export.
    polars: TokenStream,
    /// The Arrow validity bitmap type.
    bitmap: TokenStream,
}

impl Runtime {
    fn new(krate: &syn::Path) -> Self {
        let root = quote! { #krate::dataframe };
        Self {
            builders: quote! { #root::builders },
            polars: quote! { #root::__private::polars::prelude },
            bitmap: quote! { #root::__private::polars_arrow::bitmap::Bitmap },
            root,
        }
    }
}

/// One struct field that produces columns.
struct Column<'a> {
    /// How the field is read from the row: `.name` or `.0`.
    member: Member,
    /// The matching field of the generated builder struct.
    slot: syn::Ident,
    ty: &'a syn::Type,
    builder: TokenStream,
    /// What must hold for `builder` to encode `ty`.
    bound: TokenStream,
    /// Whether `ty` names one of the type's generic type parameters. Only
    /// then does `bound` go in the where-clause; a bound on a concrete type
    /// is checked where the builder struct is declared, so a field that
    /// cannot be encoded is reported at the derive instead of at first use.
    generic: bool,
    /// The column path segment this field adds, if any.
    segment: Option<syn::LitStr>,
    /// Set when the field is flattened and so must be a record.
    flattened: bool,
}

/// Gives every token the span of the field's type, so a field that cannot be
/// encoded is reported at the field instead of at the derive.
fn respan(tokens: TokenStream, span: Span) -> TokenStream {
    tokens
        .into_iter()
        .map(|mut token| {
            if let TokenTree::Group(group) = &token {
                let mut inner =
                    proc_macro2::Group::new(group.delimiter(), respan(group.stream(), span));
                inner.set_span(span);
                token = TokenTree::Group(inner);
            } else {
                token.set_span(span);
            }
            token
        })
        .collect()
}

/// Rewrites every lifetime to `'static`.
///
/// A builder owns its buffers and borrows nothing from the rows it reads, so
/// its type does not need the row's lifetimes. Naming it with `'static` makes
/// one builder type serve every lifetime of the row type.
fn erase_lifetimes(tokens: TokenStream) -> TokenStream {
    let mut erased = Vec::new();
    let mut after_quote = false;
    for token in tokens {
        let token = match token {
            TokenTree::Group(group) => {
                let mut inner =
                    proc_macro2::Group::new(group.delimiter(), erase_lifetimes(group.stream()));
                inner.set_span(group.span());
                TokenTree::Group(inner)
            }
            TokenTree::Ident(ident) if after_quote => {
                TokenTree::Ident(proc_macro2::Ident::new("static", ident.span()))
            }
            other => other,
        };
        after_quote = matches!(&token, TokenTree::Punct(punct) if punct.as_char() == '\'');
        erased.push(token);
    }
    erased.into_iter().collect()
}

fn mentions(tokens: TokenStream, params: &HashSet<String>) -> bool {
    tokens.into_iter().any(|token| match token {
        TokenTree::Ident(ident) => params.contains(&ident.to_string()),
        TokenTree::Group(group) => mentions(group.stream(), params),
        TokenTree::Punct(_) | TokenTree::Literal(_) => false,
    })
}

fn columns<'a>(
    name: &syn::Ident,
    fields: &'a Fields,
    transparent: bool,
    type_params: &HashSet<String>,
    runtime: &Runtime,
) -> syn::Result<Vec<Column<'a>>> {
    let root = &runtime.root;
    let mut columns = Vec::with_capacity(fields.len());
    for (index, field) in fields.iter().enumerate() {
        let attrs = attrs::field(&field.attrs)?;
        if attrs.skip.is_some() {
            continue;
        }

        let positional = format_ident!("field_{index}");
        let (member, slot, label) = field.ident.as_ref().map_or_else(
            || {
                (
                    Member::Unnamed(Index::from(index)),
                    positional.clone(),
                    positional.to_string(),
                )
            },
            |ident| {
                (
                    Member::Named(ident.clone()),
                    ident.clone(),
                    ident.unraw().to_string(),
                )
            },
        );
        let ty = &field.ty;
        if mirror::mentions_self(ty, name) {
            return Err(syn::Error::new(
                ty.span(),
                "a row type cannot contain itself: a DataFrame has a fixed set of columns, \
                 so recursive data needs an id field or a separate flat type",
            ));
        }

        let (builder, bound) = match &attrs.encoding {
            None => (
                quote! { <#ty as #root::Columnar>::Builder },
                quote! { #ty: #root::Columnar },
            ),
            Some((encoding, span)) => {
                if mirror::leaf_is_tuple(ty) {
                    return Err(syn::Error::new(
                        *span,
                        "encoding attributes cannot be applied to a tuple; \
                         move the value into a named struct and annotate its field",
                    ));
                }
                let builder = mirror::builder(ty, encoding, &runtime.builders);
                let bound = quote! { #builder: #root::ColumnBuilder<#ty> };
                (builder, bound)
            }
        };
        let builder = respan(builder, ty.span());

        if let (true, Some((_, span))) = (transparent, &attrs.flatten) {
            return Err(syn::Error::new(
                *span,
                "`flatten` has no effect in a `transparent` type: its one field already takes \
                 the name of the field that holds the wrapper",
            ));
        }
        let flattened = attrs.flatten.is_some();
        let segment = match attrs.flatten {
            _ if transparent => None,
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
            generic: mentions(ty.to_token_stream(), type_params),
            segment,
            flattened,
        });
    }
    Ok(columns)
}

/// Splits a run of fields into walks over the rows.
///
/// A run whose buffers number at most the runtime's limit is filled by one
/// loop that pushes every field of the run. A larger run is halved and each
/// half decided the same way; a single field takes the rows itself, which
/// lets a nested struct split its own fields. The buffer counts are
/// associated constants, so the compiler keeps only the chosen branch.
fn passes(
    run: &[usize],
    as_builder: &[TokenStream],
    slots: &[&syn::Ident],
    members: &[&Member],
    checked: &dyn Fn(usize, TokenStream) -> TokenStream,
    root: &TokenStream,
) -> TokenStream {
    let alone = |index: usize| {
        let (builder, slot, member) = (&as_builder[index], slots[index], members[index]);
        checked(
            index,
            quote! {
                #builder::extend_from(
                    &mut self.#slot,
                    ::core::iter::Iterator::map(
                        ::core::clone::Clone::clone(&values),
                        |value| &value.#member,
                    ),
                    narrow,
                )
            },
        )
    };
    match run {
        [] => TokenStream::new(),
        [index] => alone(*index),
        _ => {
            let builders: Vec<_> = run.iter().map(|&index| &as_builder[index]).collect();
            let pushes = run.iter().map(|&index| {
                let (builder, slot, member) = (&as_builder[index], slots[index], members[index]);
                checked(
                    index,
                    quote! { #builder::push(&mut self.#slot, &value.#member) },
                )
            });
            let (left, right) = run.split_at(run.len() / 2);
            let left = passes(left, as_builder, slots, members, checked, root);
            let right = passes(right, as_builder, slots, members, checked, root);
            quote! {
                if 0 #(+ #builders::STREAMS)* <= #root::__private::PASS_STREAMS {
                    for value in ::core::clone::Clone::clone(&values) {
                        #(#pushes)*
                    }
                } else {
                    #left
                    #right
                }
            }
        }
    }
}

/// The generic parameters of the builder struct, and how they are written.
struct BuilderGenerics {
    /// `<...>` on the builder's declaration.
    declaration: TokenStream,
    /// `<...>` where the builder type is named.
    arguments: TokenStream,
    /// Keeps otherwise unused parameters in use.
    marker_field: TokenStream,
    marker_init: TokenStream,
    /// True when the row's lifetimes are left off the builder.
    erased: bool,
}

/// A type whose only generic parameters are lifetimes (and consts) gets a
/// builder without those lifetimes. That is what lets another type encode
/// through it with `via`, where the row borrows from each value in turn.
fn builder_generics(generics: &syn::Generics) -> BuilderGenerics {
    let erased = generics.type_params().next().is_none() && generics.lifetimes().next().is_some();
    if erased {
        let mut kept = generics.clone();
        kept.params = kept
            .params
            .into_iter()
            .filter(|param| !matches!(param, GenericParam::Lifetime(_)))
            .collect();
        kept.where_clause = None;
        let (declaration, arguments, _) = kept.split_for_impl();
        return BuilderGenerics {
            declaration: declaration.to_token_stream(),
            arguments: arguments.to_token_stream(),
            marker_field: TokenStream::new(),
            marker_init: TokenStream::new(),
            erased,
        };
    }

    let (declaration, arguments, _) = generics.split_for_impl();
    let lifetimes = generics.lifetimes().map(|param| &param.lifetime);
    let type_idents = generics.type_params().map(|param| &param.ident);
    let (marker_field, marker_init) = if generics.params.is_empty() {
        (TokenStream::new(), TokenStream::new())
    } else {
        (
            quote! {
                __df_derive_marker:
                    ::core::marker::PhantomData<(#(&#lifetimes (),)* #(fn(&#type_idents),)*)>,
            },
            quote! { __df_derive_marker: ::core::marker::PhantomData, },
        )
    };
    BuilderGenerics {
        declaration: declaration.to_token_stream(),
        arguments: arguments.to_token_stream(),
        marker_field,
        marker_init,
        erased,
    }
}

fn where_clause(
    generics: &syn::Generics,
    container: &ContainerAttrs,
    extra: impl Iterator<Item = TokenStream>,
) -> TokenStream {
    let declared = generics
        .where_clause
        .iter()
        .flat_map(|clause| &clause.predicates);
    let requested = &container.bounds;
    quote! { where #(#declared,)* #(#requested,)* #(#extra,)* }
}

pub fn expand(input: &DeriveInput) -> syn::Result<TokenStream> {
    let container = attrs::container(&input.attrs)?;
    let runtime = Runtime::new(&container.krate);

    if let Some(via) = &container.via {
        return Ok(expand_via(input, &container, via, &runtime));
    }
    match &input.data {
        Data::Struct(data) => expand_struct(input, &data.fields, &container, &runtime),
        Data::Enum(data) => Err(syn::Error::new(
            data.enum_token.span(),
            "`ToDataFrame` cannot be derived for an enum directly. To use it as one string \
             column, implement `Columnar` for it or annotate fields with `#[df_derive(as_str)]` \
             / `#[df_derive(as_string)]`; to spread it over several columns, convert it to a \
             row struct with `#[df_derive(via = \"...\")]`",
        )),
        Data::Union(data) => Err(syn::Error::new(
            data.union_token.span(),
            "`ToDataFrame` cannot be derived for unions",
        )),
    }
}

#[allow(clippy::too_many_lines)]
fn expand_struct(
    input: &DeriveInput,
    fields: &Fields,
    container: &ContainerAttrs,
    runtime: &Runtime,
) -> syn::Result<TokenStream> {
    let Runtime {
        root,
        polars,
        bitmap,
        ..
    } = runtime;

    let type_params: HashSet<String> = input
        .generics
        .type_params()
        .map(|param| param.ident.to_string())
        .collect();
    let columns = columns(
        &input.ident,
        fields,
        container.transparent.is_some(),
        &type_params,
        runtime,
    )?;
    if let Some(span) = container.transparent
        && columns.len() != 1
    {
        return Err(syn::Error::new(
            span,
            format!(
                "a `transparent` type needs exactly one field that is not skipped (found {})",
                columns.len()
            ),
        ));
    }

    let name = &input.ident;
    let vis = &input.vis;
    let (impl_generics, ty_generics, _) = input.generics.split_for_impl();
    let generic_bounds = columns
        .iter()
        .filter(|column| column.generic)
        .flat_map(|column| {
            let ty = column.ty;
            let record = column.flattened.then(|| quote! { #ty: #root::Record });
            [Some(column.bound.clone()), record]
        })
        .flatten();
    let where_clause = where_clause(&input.generics, container, generic_bounds);

    // A flattened field of concrete type is checked here; one of generic type
    // is covered by the where-clause above.
    let record_checks = columns
        .iter()
        .filter(|column| column.flattened && !column.generic)
        .map(|column| {
            let ty = column.ty;
            quote! { #root::__private::assert_record::<#ty>(); }
        });

    let builder = builder_generics(&input.generics);
    let BuilderGenerics {
        declaration,
        arguments,
        marker_field,
        marker_init,
        ..
    } = &builder;
    let builder_where = if builder.erased {
        TokenStream::new()
    } else {
        where_clause.clone()
    };

    let slots: Vec<_> = columns.iter().map(|column| &column.slot).collect();
    let members: Vec<_> = columns.iter().map(|column| &column.member).collect();
    let builder_types: Vec<_> = columns
        .iter()
        .map(|column| {
            if builder.erased {
                erase_lifetimes(column.builder.clone())
            } else {
                column.builder.clone()
            }
        })
        .collect();
    let as_builder: Vec<_> = columns
        .iter()
        .zip(&builder_types)
        .map(|(column, builder)| {
            let ty = column.ty;
            quote! { <#builder as #root::ColumnBuilder<#ty>> }
        })
        .collect();
    let names: Vec<_> = columns
        .iter()
        .map(|column| {
            column.segment.as_ref().map_or_else(
                || quote! { ::core::clone::Clone::clone(&name) },
                |segment| quote! { #root::__private::child(&name, #segment) },
            )
        })
        .collect();
    // Runs a fallible step for one field and, on failure, names the field in
    // the error. A flattened field adds no path segment of its own.
    let checked = |index: usize, step: TokenStream| {
        columns[index].segment.as_ref().map_or_else(
            || quote! { #step?; },
            |segment| {
                quote! {
                    if let ::core::result::Result::Err(error) = #step {
                        return ::core::result::Result::Err(
                            #root::__private::in_field(error, #segment),
                        );
                    }
                }
            },
        )
    };
    let all_fields: Vec<usize> = (0..columns.len()).collect();
    let pushes = all_fields.iter().map(|&index| {
        let (builder, slot, member) = (&as_builder[index], slots[index], members[index]);
        checked(
            index,
            quote! { #builder::push(&mut self.#slot, &value.#member) },
        )
    });
    let field_passes = all_fields.iter().map(|&index| {
        let (builder, slot, member) = (&as_builder[index], slots[index], members[index]);
        checked(
            index,
            quote! {
                #builder::extend_from(
                    &mut self.#slot,
                    ::core::iter::Iterator::map(
                        ::core::clone::Clone::clone(&values),
                        |value| &value.#member,
                    ),
                    narrow,
                )
            },
        )
    });
    let passes = passes(&all_fields, &as_builder, &slots, &members, &checked, root);

    // A transparent wrapper is a value, not a row: it takes the name of the
    // field that holds it, so it is neither a record nor a `ToDataFrame`.
    let row_impls = if container.transparent.is_some() {
        TokenStream::new()
    } else {
        quote! {
            #[automatically_derived]
            impl #impl_generics #root::Record for #name #ty_generics #where_clause {}

            #[automatically_derived]
            impl #impl_generics #root::ToDataFrame for #name #ty_generics #where_clause {}
        }
    };

    Ok(quote! {
        // Field types are repeated inside the generated items with their original
        // spans, so type-shape lints would otherwise fire a second time there.
        #[allow(clippy::all, clippy::pedantic, clippy::nursery)]
        const _: () = {
            #[doc(hidden)]
            #[allow(non_snake_case, dead_code)]
            #vis struct __DfDeriveBuilder #declaration #builder_where {
                #(#slots: #builder_types,)*
                #marker_field
            }

            #[automatically_derived]
            #[allow(unused, non_snake_case, clippy::inline_always)]
            impl #impl_generics #root::ColumnBuilder<#name #ty_generics>
                for __DfDeriveBuilder #arguments
            #where_clause
            {
                fn schema(name: #polars::PlSmallStr, out: &mut #root::Fields) {
                    #(#record_checks)*
                    #(#as_builder::schema(#names, out);)*
                }

                const LISTS: usize = 0 #(+ #as_builder::LISTS)*;
                const STREAMS: usize = 0 #(+ #as_builder::STREAMS)*;

                fn with_capacity(rows: usize) -> Self {
                    Self {
                        #(#slots: #as_builder::with_capacity(rows),)*
                        #marker_init
                    }
                }

                #[inline(always)]
                fn measure(value: &#name #ty_generics, counts: &mut [usize]) {
                    let mut at = 0usize;
                    #(
                        if #as_builder::LISTS > 0 {
                            #as_builder::measure(
                                &value.#members,
                                &mut counts[at..at + #as_builder::LISTS],
                            );
                        }
                        at += #as_builder::LISTS;
                    )*
                }

                fn with_capacities(rows: usize, counts: &[usize]) -> Self {
                    let mut at = 0usize;
                    let mut take = |lists: usize| {
                        let start = at;
                        at += lists;
                        &counts[start..at]
                    };
                    Self {
                        #(
                            #slots: #as_builder::with_capacities(
                                rows,
                                take(#as_builder::LISTS),
                            ),
                        )*
                        #marker_init
                    }
                }

                #[inline(always)]
                fn push(&mut self, value: &#name #ty_generics) -> #polars::PolarsResult<()> {
                    #(#pushes)*
                    ::core::result::Result::Ok(())
                }

                #[inline(always)]
                fn extend_from<'__df_derive_rows, __DfDeriveRows>(
                    &mut self,
                    values: __DfDeriveRows,
                    narrow: bool,
                ) -> #polars::PolarsResult<()>
                where
                    #name #ty_generics: '__df_derive_rows,
                    __DfDeriveRows: ::core::iter::Iterator<
                            Item = &'__df_derive_rows #name #ty_generics,
                        > + ::core::clone::Clone,
                {
                    if narrow {
                        // Small rows: one tight loop per field.
                        #(#field_passes)*
                    } else {
                        // Larger rows: as many neighbouring fields per walk
                        // as the buffer limit allows.
                        #passes
                    }
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
            impl #impl_generics #root::Columnar for #name #ty_generics #where_clause {
                type Builder = __DfDeriveBuilder #arguments;
            }

            #row_impls
        };
    })
}

/// A type that encodes as another row type built from a reference to it.
///
/// Each value is converted to the row with `From<&T>` (or `TryFrom<&T>`) and
/// pushed straight into the row type's builder, so no intermediate collection
/// of rows is built. The row may borrow from the value: write its borrowed
/// lifetimes as `'_`.
fn expand_via(
    input: &DeriveInput,
    container: &ContainerAttrs,
    via: &Via,
    runtime: &Runtime,
) -> TokenStream {
    let Runtime {
        root,
        polars,
        bitmap,
        ..
    } = runtime;
    let name = &input.ident;
    let vis = &input.vis;
    let (impl_generics, ty_generics, _) = input.generics.split_for_impl();
    let where_clause = where_clause(&input.generics, container, std::iter::empty());

    let row = respan(via.row.to_token_stream(), via.span);
    let stored = erase_lifetimes(row.clone());
    let inner = quote! { <#stored as #root::Columnar>::Builder };
    let convert = if via.fallible {
        quote! {
            match ::core::convert::TryFrom::try_from(value) {
                ::core::result::Result::Ok(row) => row,
                ::core::result::Result::Err(error) => {
                    return ::core::result::Result::Err(
                        #root::__private::conversion_error(&error),
                    );
                }
            }
        }
    } else {
        quote! { ::core::convert::From::from(value) }
    };

    let (declaration, arguments, _) = input.generics.split_for_impl();
    let lifetimes = input.generics.lifetimes().map(|param| &param.lifetime);
    let type_idents = input.generics.type_params().map(|param| &param.ident);
    let marker = quote! {
        ::core::marker::PhantomData<(#(&#lifetimes (),)* #(fn(&#type_idents),)*)>
    };

    quote! {
        // Field types are repeated inside the generated items with their original
        // spans, so type-shape lints would otherwise fire a second time there.
        #[allow(clippy::all, clippy::pedantic, clippy::nursery)]
        const _: () = {
            #[doc(hidden)]
            #vis struct __DfDeriveBuilder #declaration (#inner, #marker) #where_clause;

            #[automatically_derived]
            #[allow(clippy::inline_always)]
            impl #impl_generics #root::ColumnBuilder<#name #ty_generics>
                for __DfDeriveBuilder #arguments
            #where_clause
            {
                fn schema(name: #polars::PlSmallStr, out: &mut #root::Fields) {
                    <#inner as #root::ColumnBuilder<#stored>>::schema(name, out);
                }

                fn with_capacity(rows: usize) -> Self {
                    Self(
                        <#inner as #root::ColumnBuilder<#stored>>::with_capacity(rows),
                        ::core::marker::PhantomData,
                    )
                }

                #[inline(always)]
                fn push(&mut self, value: &#name #ty_generics) -> #polars::PolarsResult<()> {
                    let row: #row = #convert;
                    <#inner as #root::ColumnBuilder<#row>>::push(&mut self.0, &row)
                }

                #[inline(always)]
                fn push_null(&mut self) {
                    <#inner as #root::ColumnBuilder<#stored>>::push_null(&mut self.0);
                }

                fn finish(
                    self,
                    name: #polars::PlSmallStr,
                    validity: ::core::option::Option<&#bitmap>,
                    out: &mut ::std::vec::Vec<#polars::Column>,
                ) -> #polars::PolarsResult<()> {
                    <#inner as #root::ColumnBuilder<#stored>>::finish(self.0, name, validity, out)
                }
            }

            #[automatically_derived]
            impl #impl_generics #root::Columnar for #name #ty_generics #where_clause {
                type Builder = __DfDeriveBuilder #arguments;
            }

            #[automatically_derived]
            impl #impl_generics #root::Record for #name #ty_generics #where_clause {}

            #[automatically_derived]
            impl #impl_generics #root::ToDataFrame for #name #ty_generics #where_clause {}
        };
    }
}

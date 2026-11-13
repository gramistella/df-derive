//! Code generation.
//!
//! A derived type gets one builder struct, with a member for every field
//! that produces columns. A builder borrows nothing from the rows it reads,
//! so its type is not tied to one instance of the row type:
//!
//! - A member whose type names no type or const parameter of the row type is
//!   written out, with `'static` in place of every lifetime. The builder of
//!   a `&'a str` is the same type whatever `'a` is, so that member encodes
//!   the field for every `'a`.
//! - A member whose type names such a parameter cannot be written without an
//!   instance of the row type in hand. The builder struct then takes that
//!   instance as its one parameter, `__DfDeriveFor`, and gets the member's
//!   type from it through a hidden trait. Such a builder encodes every
//!   instance whose field the member encodes, not only the one that named
//!   it.
//!
//! So the builder that `Row<'static, &'static str>` names also encodes a
//! `Row<'a, &'a str>`. A projection depends on this: its row borrows from
//! each value in turn, and no type can name that borrow.

use std::collections::HashSet;

use proc_macro2::{Span, TokenStream, TokenTree};
use quote::{ToTokens, format_ident, quote};
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
    /// The builder of the field, as the row type itself would name it.
    builder: TokenStream,
    /// What must hold for `builder` to encode `ty`: one or more predicates.
    bounds: TokenStream,
    /// Whether `ty` or `builder` names a type or const parameter of the row
    /// type. Only then do `bounds` go in the where-clause. A field that does
    /// not depend on a parameter is checked where the builder is declared
    /// and used, so one that cannot be encoded is reported where the type is
    /// defined instead of at first use.
    generic: bool,
    /// Whether `builder` names such a parameter, which makes it a different
    /// type for different instances of the row type.
    per_instance: bool,
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

/// Rewrites the name of every lifetime in `tokens` that `rename` has a new
/// name for.
fn rename_lifetimes(
    tokens: TokenStream,
    rename: &mut impl FnMut(&proc_macro2::Ident) -> Option<String>,
) -> TokenStream {
    let mut renamed = Vec::new();
    let mut after_quote = false;
    for token in tokens {
        let token = match token {
            TokenTree::Group(group) => {
                let mut inner = proc_macro2::Group::new(
                    group.delimiter(),
                    rename_lifetimes(group.stream(), rename),
                );
                inner.set_span(group.span());
                TokenTree::Group(inner)
            }
            TokenTree::Ident(ident) if after_quote => match rename(&ident) {
                Some(name) => TokenTree::Ident(proc_macro2::Ident::new(&name, ident.span())),
                None => TokenTree::Ident(ident),
            },
            other => other,
        };
        after_quote = matches!(&token, TokenTree::Punct(punct) if punct.as_char() == '\'');
        renamed.push(token);
    }
    renamed.into_iter().collect()
}

/// Rewrites every lifetime to `'static`.
///
/// A builder owns its buffers and borrows nothing from the rows it reads, so
/// its type does not need the row's lifetimes. Naming a member with `'static`
/// makes one type serve every lifetime of the field.
fn erase_lifetimes(tokens: TokenStream) -> TokenStream {
    rename_lifetimes(tokens, &mut |_| Some("static".to_owned()))
}

/// Gives every `'_` in `tokens` a name of its own and returns the names, so
/// that a bound can be written that holds whatever each of them is.
fn name_borrows(tokens: TokenStream) -> (TokenStream, Vec<syn::Lifetime>) {
    let mut borrows = Vec::new();
    let renamed = rename_lifetimes(tokens, &mut |ident| {
        (ident == "_").then(|| {
            let name = format!("__df_derive_borrow_{}", borrows.len());
            borrows.push(syn::Lifetime::new(&format!("'{name}"), ident.span()));
            name
        })
    });
    (renamed, borrows)
}

/// The parameters that make an instance of a type a different type: its type
/// and const parameters.
fn parameters(generics: &syn::Generics) -> HashSet<String> {
    generics
        .params
        .iter()
        .filter_map(|param| match param {
            GenericParam::Type(param) => Some(param.ident.to_string()),
            GenericParam::Const(param) => Some(param.ident.to_string()),
            GenericParam::Lifetime(_) => None,
        })
        .collect()
}

/// Whether `tokens` name one of `params`. A lifetime is not one of them,
/// even when it has the name of one.
fn mentions(tokens: TokenStream, params: &HashSet<String>) -> bool {
    let mut after_quote = false;
    for token in tokens {
        let found = match &token {
            TokenTree::Ident(ident) => !after_quote && params.contains(&ident.to_string()),
            TokenTree::Group(group) => mentions(group.stream(), params),
            TokenTree::Punct(_) | TokenTree::Literal(_) => false,
        };
        if found {
            return true;
        }
        after_quote = matches!(&token, TokenTree::Punct(punct) if punct.as_char() == '\'');
    }
    false
}

fn columns<'a>(
    name: &syn::Ident,
    fields: &'a Fields,
    transparent: bool,
    params: &HashSet<String>,
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

        let (builder, bounds) = match &attrs.encoding {
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
                let mut keys = Vec::new();
                let builder = mirror::builder(ty, encoding, root, &runtime.builders, &mut keys);
                // The builder names the builder of every map key, and a key
                // that depends on a parameter has one only if the
                // where-clause says so.
                let keys = keys
                    .into_iter()
                    .filter(|key| mentions(key.to_token_stream(), params));
                let bounds = quote! {
                    #(#keys: #root::Columnar,)*
                    #builder: #root::ColumnBuilder<#ty>
                };
                (builder, bounds)
            }
        };
        let per_instance = mentions(builder.clone(), params);
        let generic = per_instance || mentions(ty.to_token_stream(), params);
        let builder = respan(builder, ty.span());

        if let (true, Some((_, span))) = (transparent, &attrs.flatten) {
            return Err(syn::Error::new(
                *span,
                "`flatten` has no effect in a `transparent` type: its one field already takes \
                 the name of the field that holds the wrapper",
            ));
        }
        if let (true, Some((_, span))) = (transparent, &attrs.rename) {
            return Err(syn::Error::new(
                *span,
                "`rename` has no effect in a `transparent` type: its one field already takes \
                 the name of the field that holds the wrapper",
            ));
        }
        let flattened = attrs.flatten.is_some();
        let segment = match (attrs.flatten, attrs.rename) {
            _ if transparent => None,
            (None, Some((name, _))) => Some(name),
            (None, None) => Some(syn::LitStr::new(&label, field.span())),
            (Some((Flatten::Bare, _)), _) => None,
            (Some((Flatten::Prefix(prefix), _)), _) => Some(prefix),
        };

        columns.push(Column {
            member,
            slot,
            ty,
            builder,
            bounds,
            generic,
            per_instance,
            segment,
            flattened,
        });
    }
    Ok(columns)
}

/// What every generated impl requires whatever the fields are: the type's
/// own where-clause and the bounds asked for with `bound = "..."`.
fn own_predicates(generics: &syn::Generics, container: &ContainerAttrs) -> Vec<TokenStream> {
    let declared = generics
        .where_clause
        .iter()
        .flat_map(|clause| &clause.predicates)
        .map(ToTokens::to_token_stream);
    let requested = container.bounds.iter().map(ToTokens::to_token_stream);
    declared.chain(requested).collect()
}

/// How a builder struct is declared and named.
///
/// A builder with a member that depends on a parameter of the derived type
/// has one parameter of its own: `__DfDeriveFor`, the instance of the type
/// that named it. Any other builder has none.
struct Naming {
    /// The builder's parameter where the struct is declared, if it has one.
    declaration: TokenStream,
    /// The same where the impl that encodes rows names the builder.
    arguments: TokenStream,
    /// The same where the derived type names its builder: itself.
    of_self: TokenStream,
    /// The generic parameters of the impl that encodes rows: the type's own,
    /// and the builder's.
    encoder: syn::Generics,
}

impl Naming {
    /// `members` is the hidden trait through which the naming instance says
    /// what the builder's members are, when some member depends on it.
    fn new(generics: &syn::Generics, members: Option<TokenStream>) -> Self {
        let mut encoder = generics.clone();
        let Some(members) = members else {
            return Self {
                declaration: TokenStream::new(),
                arguments: TokenStream::new(),
                of_self: TokenStream::new(),
                encoder,
            };
        };
        let parameter = quote! { __DfDeriveFor: #members + ?::core::marker::Sized };
        encoder.params.push(syn::parse_quote! { #parameter });
        Self {
            declaration: quote! { <#parameter> },
            arguments: quote! { <__DfDeriveFor> },
            of_self: quote! { <Self> },
            encoder,
        }
    }
}

pub fn expand(input: &DeriveInput) -> syn::Result<TokenStream> {
    let container = attrs::container(&input.attrs)?;
    let runtime = Runtime::new(&container.krate);

    if let Some(via) = &container.via {
        attrs::none_inside(&input.data)?;
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

    let columns = columns(
        &input.ident,
        fields,
        container.transparent.is_some(),
        &parameters(&input.generics),
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
    let own = own_predicates(&input.generics, container);
    // What the type needs to be stored in columns: its own predicates, and
    // what each field that depends on a parameter needs.
    let generic_bounds = columns
        .iter()
        .filter(|column| column.generic)
        .flat_map(|column| {
            let ty = column.ty;
            let record = column.flattened.then(|| quote! { #ty: #root::Record });
            [Some(column.bounds.clone()), record]
        })
        .flatten();
    let where_clause = quote! { where #(#own,)* #(#generic_bounds,)* };

    // A flattened field of concrete type is checked here; one that depends on
    // a parameter is covered by the where-clause above.
    let record_checks = columns
        .iter()
        .filter(|column| column.flattened && !column.generic)
        .map(|column| {
            let ty = column.ty;
            quote! { #root::__private::assert_record::<#ty>(); }
        });

    let per_instance: Vec<_> = columns
        .iter()
        .filter(|column| column.per_instance)
        .collect();
    let naming = Naming::new(
        &input.generics,
        (!per_instance.is_empty()).then(|| quote! { __DfDeriveFields }),
    );
    let Naming {
        declaration,
        arguments,
        of_self,
        ..
    } = &naming;
    let (encoder_generics, _, _) = naming.encoder.split_for_impl();
    // The derived type says which builder each member taken from the naming
    // instance is. Every token of that has the span of the field's type, so
    // a field without a builder is reported at the field. The trait is
    // private because its impl names the builders of fields whose types may
    // be private, which the impl of a public trait must not do.
    let named_members = (!per_instance.is_empty()).then(|| {
        let slots = per_instance.iter().map(|column| &column.slot);
        let named = per_instance.iter().map(|column| {
            let (slot, builder) = (&column.slot, &column.builder);
            respan(quote! { type #slot = #builder; }, column.ty.span())
        });
        quote! {
            #[doc(hidden)]
            #[allow(non_camel_case_types)]
            trait __DfDeriveFields {
                #(type #slots;)*
            }

            #[automatically_derived]
            #[allow(non_camel_case_types)]
            impl #impl_generics __DfDeriveFields for #name #ty_generics #where_clause {
                #(#named)*
            }
        }
    });

    let slots: Vec<_> = columns.iter().map(|column| &column.slot).collect();
    let members: Vec<_> = columns.iter().map(|column| &column.member).collect();
    let builder_types: Vec<_> = columns
        .iter()
        .map(|column| {
            if column.per_instance {
                let slot = &column.slot;
                quote! { <__DfDeriveFor as __DfDeriveFields>::#slot }
            } else {
                erase_lifetimes(column.builder.clone())
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
    // What the fields that depend on a parameter need where rows are
    // encoded. A member written out needs what its field needs. A member
    // taken from the naming instance has to encode the field as this row has
    // it, whichever instance that was.
    let encoder_bounds = columns
        .iter()
        .filter(|column| column.generic)
        .map(|column| {
            if column.per_instance {
                let (ty, slot) = (column.ty, &column.slot);
                quote! { <__DfDeriveFor as __DfDeriveFields>::#slot: #root::ColumnBuilder<#ty> }
            } else {
                column.bounds.clone()
            }
        });
    let encoder_where = quote! { where #(#own,)* #(#encoder_bounds,)* };
    let names: Vec<_> = columns
        .iter()
        .map(|column| {
            column.segment.as_ref().map_or_else(
                || quote! { ::core::clone::Clone::clone(&name) },
                |segment| quote! { #root::__private::child(&name, #segment) },
            )
        })
        .collect();
    // A field of the row in a slot: where the slot is empty, so is the field.
    // Written as a `match` rather than with `Option::map`, which would cost
    // another closure and another instantiation for every field.
    let field_of: Vec<_> = members
        .iter()
        .map(|member| {
            quote! {
                match value {
                    ::core::option::Option::Some(value) => {
                        ::core::option::Option::Some(&value.#member)
                    }
                    ::core::option::Option::None => ::core::option::Option::None,
                }
            }
        })
        .collect();
    // The same field of every row a builder is shown.
    let field_of_each: Vec<_> = field_of
        .iter()
        .map(|field| {
            quote! {
                ::core::iter::Iterator::map(::core::clone::Clone::clone(&values), |value| #field)
            }
        })
        .collect();
    // A failed push names the field it happened in. A flattened field adds no
    // path segment of its own.
    let pushes =
        columns
            .iter()
            .zip(&as_builder)
            .zip(&field_of)
            .map(|((column, builder), field)| {
                let slot = &column.slot;
                let push = quote! { #builder::push(&mut self.#slot, #field) };
                column.segment.as_ref().map_or_else(
                    || quote! { #push?; },
                    |segment| {
                        quote! {
                            if let ::core::result::Result::Err(error) = #push {
                                return ::core::result::Result::Err(
                                    #root::__private::in_field(error, #segment),
                                );
                            }
                        }
                    },
                )
            });
    let has_members = !columns.is_empty();
    let rows = quote! {
        #name #ty_generics: '__df_derive_rows,
        __DfDeriveRows: ::core::iter::Iterator<
                Item = ::core::option::Option<&'__df_derive_rows #name #ty_generics>,
            > + ::core::clone::Clone,
    };

    // A transparent wrapper is a value, not a row: it takes the name of the
    // field that holds it, so it is neither a record nor a `ToDataFrame`.
    let encode = encode_in_place(runtime);
    let row_impls = if container.transparent.is_some() {
        TokenStream::new()
    } else {
        quote! {
            #[automatically_derived]
            impl #impl_generics #root::Record for #name #ty_generics #where_clause {}

            #[automatically_derived]
            impl #impl_generics #root::ToDataFrame for #name #ty_generics #where_clause {
                #encode
            }
        }
    };

    Ok(quote! {
        // Field types are repeated inside the generated items with their original
        // spans, so type-shape lints would otherwise fire a second time there.
        #[allow(clippy::all, clippy::pedantic, clippy::nursery)]
        const _: () = {
            #named_members

            // The last member decides whether a struct is `Sized`. If it were
            // the builder of a field that cannot be encoded, the compiler
            // would report that field again wherever the builder's size is
            // needed. With `()` last, the field is reported at the field and
            // once at the derive.
            #[doc(hidden)]
            #[allow(private_bounds, non_snake_case, dead_code)]
            #vis struct __DfDeriveBuilder #declaration {
                #(#slots: #builder_types,)*
                __df_derive_last: (),
            }

            #[automatically_derived]
            #[allow(unused, non_snake_case, clippy::inline_always)]
            impl #encoder_generics #root::ColumnBuilder<#name #ty_generics>
                for __DfDeriveBuilder #arguments
            #encoder_where
            {
                // A row is fed a row at a time when every member is.
                const ROW_AT_A_TIME: bool = #has_members #(&& #as_builder::ROW_AT_A_TIME)*;

                fn schema(name: #polars::PlSmallStr, out: &mut #root::Fields) {
                    #(#record_checks)*
                    #(#as_builder::schema(#names, out);)*
                }

                fn new() -> Self {
                    Self {
                        #(#slots: #as_builder::new(),)*
                        __df_derive_last: (),
                    }
                }

                #[inline(always)]
                fn measure<'__df_derive_rows, __DfDeriveRows>(&mut self, values: __DfDeriveRows)
                where
                    #rows
                {
                    #(#as_builder::measure(&mut self.#slots, #field_of_each);)*
                }

                fn reserve(&mut self, additional: usize) {
                    #(#as_builder::reserve(&mut self.#slots, additional);)*
                }

                #[inline(always)]
                fn push(
                    &mut self,
                    value: ::core::option::Option<&#name #ty_generics>,
                ) -> #polars::PolarsResult<()> {
                    #(#pushes)*
                    ::core::result::Result::Ok(())
                }

                #[inline(always)]
                fn extend<'__df_derive_rows, __DfDeriveRows>(
                    &mut self,
                    values: __DfDeriveRows,
                ) -> #polars::PolarsResult<()>
                where
                    #rows
                {
                    // The members that take a row at a time are fed first,
                    // together, in one walk over the rows of the block.
                    if false #(|| #as_builder::ROW_AT_A_TIME)* {
                        for value in ::core::clone::Clone::clone(&values) {
                            #(
                                if #as_builder::ROW_AT_A_TIME {
                                    #as_builder::push(&mut self.#slots, #field_of)?;
                                }
                            )*
                        }
                    }
                    // Each of the others gets the block as a run of its own:
                    // one loop per field.
                    #(
                        if !#as_builder::ROW_AT_A_TIME {
                            #as_builder::extend(&mut self.#slots, #field_of_each)?;
                        }
                    )*
                    ::core::result::Result::Ok(())
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
                type Builder = __DfDeriveBuilder #of_self;
            }

            #row_impls
        };
    })
}

/// The one `ToDataFrame` method the derive writes out.
///
/// Its body is what the trait's default does anyway. Written out in the impl,
/// it is an ordinary function of the crate that defines the row type, so the
/// encoder is compiled there once. Left to the default it would be
/// instantiated in every crate that converts the type.
fn encode_in_place(runtime: &Runtime) -> TokenStream {
    let Runtime { root, polars, .. } = runtime;
    quote! {
        #[inline(never)]
        fn __encode(rows: &[Self]) -> #polars::PolarsResult<#polars::DataFrame> {
            #root::__private::encode(rows)
        }
    }
}

/// A type that encodes as another row type built from a reference to it.
///
/// Each value is converted to the row with `From<&T>` (or `TryFrom<&T>`) and
/// pushed straight into the row type's builder, so no intermediate collection
/// of rows is built. The row may borrow from the value: write its borrowed
/// lifetimes as `'_`.
#[allow(clippy::too_many_lines)]
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
    let own = own_predicates(&input.generics, container);
    let where_clause = quote! { where #(#own,)* };

    let row = respan(via.row.to_token_stream(), via.span);
    let stored = erase_lifetimes(row.clone());
    // The builder of the row type is this builder's one member. Like any
    // member, it is written out unless it depends on a parameter.
    let per_instance = mentions(row.clone(), &parameters(&input.generics));
    let naming = Naming::new(
        &input.generics,
        per_instance.then(|| quote! { __DfDeriveVia }),
    );
    let Naming {
        declaration,
        arguments,
        of_self,
        ..
    } = &naming;
    let (encoder_generics, _, _) = naming.encoder.split_for_impl();
    let named_member = per_instance.then(|| {
        // Whatever keeps the row type from having a builder is reported at
        // the attribute that names it. The trait is private because the row
        // type may be.
        let named = respan(
            quote! { type Row = <#stored as #root::Columnar>::Builder; },
            via.span,
        );
        quote! {
            #[doc(hidden)]
            trait __DfDeriveVia {
                type Row;
            }

            #[automatically_derived]
            impl #impl_generics __DfDeriveVia for #name #ty_generics #where_clause {
                #named
            }
        }
    });
    let (inner, encoder_where) = if per_instance {
        // The member has to encode the row whatever the row borrows.
        let inner = quote! { <__DfDeriveFor as __DfDeriveVia>::Row };
        let (any_row, borrows) = name_borrows(row.clone());
        let for_any_borrow = (!borrows.is_empty()).then(|| quote! { for<#(#borrows),*> });
        let encodes = quote! { #for_any_borrow #inner: #root::ColumnBuilder<#any_row> };
        (inner, quote! { where #(#own,)* #encodes })
    } else {
        (
            quote! { <#stored as #root::Columnar>::Builder },
            where_clause.clone(),
        )
    };
    let as_inner = quote! { <#inner as #root::ColumnBuilder<#row>> };
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

    let encode = encode_in_place(runtime);
    // Whatever keeps the row type's builder from encoding the row is
    // reported at the attribute too.
    let named_builder = respan(
        quote! { type Builder = __DfDeriveBuilder #of_self; },
        via.span,
    );

    quote! {
        // Field types are repeated inside the generated items with their original
        // spans, so type-shape lints would otherwise fire a second time there.
        #[allow(clippy::all, clippy::pedantic, clippy::nursery)]
        const _: () = {
            #named_member

            #[doc(hidden)]
            #[allow(private_bounds)]
            #vis struct __DfDeriveBuilder #declaration (#inner);

            #[automatically_derived]
            #[allow(clippy::inline_always)]
            impl #encoder_generics #root::ColumnBuilder<#name #ty_generics>
                for __DfDeriveBuilder #arguments
            #encoder_where
            {
                const ROW_AT_A_TIME: bool =
                    <#inner as #root::ColumnBuilder<#stored>>::ROW_AT_A_TIME;

                fn schema(name: #polars::PlSmallStr, out: &mut #root::Fields) {
                    // A projection is a row type because the type it encodes
                    // as is one.
                    #root::__private::assert_record::<#row>();
                    #as_inner::schema(name, out);
                }

                fn new() -> Self {
                    Self(#as_inner::new())
                }

                fn reserve(&mut self, additional: usize) {
                    #as_inner::reserve(&mut self.0, additional);
                }

                #[inline(always)]
                fn push(
                    &mut self,
                    value: ::core::option::Option<&#name #ty_generics>,
                ) -> #polars::PolarsResult<()> {
                    let row: ::core::option::Option<#row> = match value {
                        ::core::option::Option::Some(value) => {
                            ::core::option::Option::Some(#convert)
                        }
                        ::core::option::Option::None => ::core::option::Option::None,
                    };
                    #as_inner::push(&mut self.0, ::core::option::Option::as_ref(&row))
                }

                fn finish(
                    self,
                    name: #polars::PlSmallStr,
                    validity: ::core::option::Option<&#bitmap>,
                    out: &mut ::std::vec::Vec<#polars::Column>,
                ) -> #polars::PolarsResult<()> {
                    #as_inner::finish(self.0, name, validity, out)
                }
            }

            #[automatically_derived]
            impl #impl_generics #root::Columnar for #name #ty_generics #where_clause {
                #named_builder
            }

            #[automatically_derived]
            impl #impl_generics #root::Record for #name #ty_generics #where_clause {}

            #[automatically_derived]
            impl #impl_generics #root::ToDataFrame for #name #ty_generics #where_clause {
                #encode
            }
        };
    }
}

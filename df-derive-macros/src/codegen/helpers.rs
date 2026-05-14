use crate::codegen::encoder::idents;
use crate::ir::{DisplayBase, LeafSpec, StringyBase, StructIR};
use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::{GenericArgument, GenericParam, Ident, PathArguments, Type};

/// Convert a Rust field identifier into the user-facing `DataFrame` column name.
///
/// `Ident::to_string()` preserves raw identifier spelling (`r#type`), but
/// column names should reflect the logical field name users wrote around the
/// keyword escape (`type`).
pub(in crate::codegen) fn column_name_for_ident(ident: &Ident) -> String {
    let name = ident.to_string();
    name.strip_prefix("r#").unwrap_or(&name).to_owned()
}

#[allow(clippy::struct_field_names)]
struct GenericContext {
    type_params: Vec<Ident>,
    const_params: Vec<Ident>,
    lifetime_params: Vec<Ident>,
}

impl GenericContext {
    fn new(ir: &StructIR) -> Self {
        let mut type_params = Vec::new();
        let mut const_params = Vec::new();
        let mut lifetime_params = Vec::new();

        for param in &ir.generics.params {
            match param {
                GenericParam::Type(type_param) => type_params.push(type_param.ident.clone()),
                GenericParam::Const(const_param) => const_params.push(const_param.ident.clone()),
                GenericParam::Lifetime(lifetime_param) => {
                    lifetime_params.push(lifetime_param.lifetime.ident.clone());
                }
            }
        }

        Self {
            type_params,
            const_params,
            lifetime_params,
        }
    }

    fn has_type_param(&self, ident: &Ident) -> bool {
        self.type_params.iter().any(|param| param == ident)
    }

    fn has_lifetime_param(&self, ident: &Ident) -> bool {
        self.lifetime_params.iter().any(|param| param == ident)
    }

    const fn has_const_params(&self) -> bool {
        !self.const_params.is_empty()
    }
}

pub fn generate_eager_asserts(
    ir: &StructIR,
    to_dataframe_trait: &TokenStream,
    columnar_trait: &TokenStream,
    decimal128_encode_trait: &TokenStream,
) -> TokenStream {
    let generic_ctx = GenericContext::new(ir);
    let mut nested_types = Vec::new();
    let mut decimal_backend_types = Vec::new();
    let mut as_ref_str_types = Vec::new();
    let mut display_types = Vec::new();

    for field in &ir.fields {
        collect_nested_asserts(&field.leaf_spec, &generic_ctx, &mut nested_types);
        if let Some(ty) = &field.decimal_backend_ty
            && !type_depends_on_generics(ty, &generic_ctx)
        {
            push_unique_type(&mut decimal_backend_types, ty);
        }
        collect_as_ref_str_asserts(&field.leaf_spec, &generic_ctx, &mut as_ref_str_types);
        collect_display_asserts(&field.leaf_spec, &generic_ctx, &mut display_types);
    }

    if nested_types.is_empty()
        && decimal_backend_types.is_empty()
        && as_ref_str_types.is_empty()
        && display_types.is_empty()
    {
        return TokenStream::new();
    }

    let nested_asserts = if nested_types.is_empty() {
        TokenStream::new()
    } else {
        let assert_nested_traits = idents::nested_traits_assert_helper();
        quote! {
            const fn #assert_nested_traits<
                __DfDeriveT: #to_dataframe_trait + #columnar_trait
            >() {}

            #(
                #assert_nested_traits::<#nested_types>();
            )*
        }
    };
    let decimal_backend_asserts = if decimal_backend_types.is_empty() {
        TokenStream::new()
    } else {
        let assert_decimal_backend = idents::decimal_backend_assert_helper();
        quote! {
            const fn #assert_decimal_backend<
                __DfDeriveT: #decimal128_encode_trait
            >() {}

            #(
                #assert_decimal_backend::<#decimal_backend_types>();
            )*
        }
    };
    let as_ref_str_asserts = if as_ref_str_types.is_empty() {
        TokenStream::new()
    } else {
        let assert_as_ref_str = idents::as_ref_str_assert_helper();
        quote! {
            const fn #assert_as_ref_str<
                __DfDeriveT: ?::core::marker::Sized + ::core::convert::AsRef<str>
            >() {}

            #(
                #assert_as_ref_str::<#as_ref_str_types>();
            )*
        }
    };
    let display_asserts = if display_types.is_empty() {
        TokenStream::new()
    } else {
        let assert_display = idents::display_assert_helper();
        quote! {
            const fn #assert_display<
                __DfDeriveT: ?::core::marker::Sized + ::core::fmt::Display
            >() {}

            #(
                #assert_display::<#display_types>();
            )*
        }
    };

    quote! {
        #nested_asserts
        #decimal_backend_asserts
        #as_ref_str_asserts
        #display_asserts
    }
}

fn collect_nested_asserts(leaf: &LeafSpec, generic_ctx: &GenericContext, out: &mut Vec<Type>) {
    match leaf {
        LeafSpec::Struct(ty) => {
            if !type_depends_on_generics(ty, generic_ctx) {
                push_unique_type(out, ty);
            }
        }
        LeafSpec::Tuple(elements) => {
            for element in elements {
                collect_nested_asserts(&element.leaf_spec, generic_ctx, out);
            }
        }
        LeafSpec::Numeric(_)
        | LeafSpec::String
        | LeafSpec::Bool
        | LeafSpec::DateTime(_)
        | LeafSpec::NaiveDateTime(_)
        | LeafSpec::NaiveDate
        | LeafSpec::NaiveTime
        | LeafSpec::Duration { .. }
        | LeafSpec::Decimal { .. }
        | LeafSpec::Generic(_)
        | LeafSpec::AsString(_)
        | LeafSpec::AsStr(_)
        | LeafSpec::Binary => {}
    }
}

fn collect_as_ref_str_asserts(leaf: &LeafSpec, generic_ctx: &GenericContext, out: &mut Vec<Type>) {
    match leaf {
        LeafSpec::AsStr(StringyBase::Struct(ty)) => {
            if !type_depends_on_generics(ty, generic_ctx) {
                push_unique_type(out, ty);
            }
        }
        LeafSpec::Tuple(elements) => {
            for element in elements {
                collect_as_ref_str_asserts(&element.leaf_spec, generic_ctx, out);
            }
        }
        LeafSpec::Numeric(_)
        | LeafSpec::String
        | LeafSpec::Bool
        | LeafSpec::DateTime(_)
        | LeafSpec::NaiveDateTime(_)
        | LeafSpec::NaiveDate
        | LeafSpec::NaiveTime
        | LeafSpec::Duration { .. }
        | LeafSpec::Decimal { .. }
        | LeafSpec::Struct(_)
        | LeafSpec::Generic(_)
        | LeafSpec::AsString(_)
        | LeafSpec::AsStr(_)
        | LeafSpec::Binary => {}
    }
}

fn collect_display_asserts(leaf: &LeafSpec, generic_ctx: &GenericContext, out: &mut Vec<Type>) {
    match leaf {
        LeafSpec::AsString(DisplayBase::Struct(ty)) => {
            if !type_depends_on_generics(ty, generic_ctx) {
                push_unique_type(out, ty);
            }
        }
        LeafSpec::Tuple(elements) => {
            for element in elements {
                collect_display_asserts(&element.leaf_spec, generic_ctx, out);
            }
        }
        LeafSpec::Numeric(_)
        | LeafSpec::String
        | LeafSpec::Bool
        | LeafSpec::DateTime(_)
        | LeafSpec::NaiveDateTime(_)
        | LeafSpec::NaiveDate
        | LeafSpec::NaiveTime
        | LeafSpec::Duration { .. }
        | LeafSpec::Decimal { .. }
        | LeafSpec::Struct(_)
        | LeafSpec::Generic(_)
        | LeafSpec::AsString(_)
        | LeafSpec::AsStr(_)
        | LeafSpec::Binary => {}
    }
}

pub(in crate::codegen) fn push_unique_type(out: &mut Vec<Type>, ty: &Type) {
    let key = ty.to_token_stream().to_string();
    if !out
        .iter()
        .any(|existing| existing.to_token_stream().to_string() == key)
    {
        out.push(ty.clone());
    }
}

fn path_depends_on_generics(path: &syn::Path, generic_ctx: &GenericContext) -> bool {
    path.segments.iter().any(|segment| {
        if generic_ctx.has_type_param(&segment.ident) {
            return true;
        }

        let PathArguments::AngleBracketed(args) = &segment.arguments else {
            return false;
        };

        args.args
            .iter()
            .any(|arg| generic_argument_depends_on_generics(arg, generic_ctx))
    })
}

fn generic_argument_depends_on_generics(
    arg: &GenericArgument,
    generic_ctx: &GenericContext,
) -> bool {
    match arg {
        GenericArgument::Lifetime(lifetime) => generic_ctx.has_lifetime_param(&lifetime.ident),
        GenericArgument::Type(ty) => type_depends_on_generics(ty, generic_ctx),
        GenericArgument::Const(_) => generic_ctx.has_const_params(),
        GenericArgument::AssocType(assoc) => type_depends_on_generics(&assoc.ty, generic_ctx),
        GenericArgument::Constraint(constraint) => constraint.bounds.iter().any(|bound| {
            if let syn::TypeParamBound::Trait(trait_bound) = bound {
                path_depends_on_generics(&trait_bound.path, generic_ctx)
            } else {
                false
            }
        }),
        _ => false,
    }
}

fn type_depends_on_generics(ty: &Type, generic_ctx: &GenericContext) -> bool {
    match ty {
        Type::Array(array) => {
            generic_ctx.has_const_params()
                || type_depends_on_generics(array.elem.as_ref(), generic_ctx)
        }
        Type::Group(group) => type_depends_on_generics(group.elem.as_ref(), generic_ctx),
        Type::Paren(paren) => type_depends_on_generics(paren.elem.as_ref(), generic_ctx),
        Type::Path(type_path) => {
            if let Some(qself) = &type_path.qself
                && type_depends_on_generics(qself.ty.as_ref(), generic_ctx)
            {
                return true;
            }

            if type_path.qself.is_none()
                && type_path.path.segments.len() == 1
                && let Some(segment) = type_path.path.segments.last()
                && matches!(segment.arguments, PathArguments::None)
                && generic_ctx.has_type_param(&segment.ident)
            {
                return true;
            }

            path_depends_on_generics(&type_path.path, generic_ctx)
        }
        Type::Ptr(ptr) => type_depends_on_generics(ptr.elem.as_ref(), generic_ctx),
        Type::Reference(reference) => {
            type_depends_on_generics(reference.elem.as_ref(), generic_ctx)
        }
        Type::Slice(slice) => type_depends_on_generics(slice.elem.as_ref(), generic_ctx),
        Type::Tuple(tuple) => tuple
            .elems
            .iter()
            .any(|elem| type_depends_on_generics(elem, generic_ctx)),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::column_name_for_ident;

    #[test]
    fn column_names_strip_raw_identifier_prefix() {
        let raw: syn::Ident = syn::parse_str("r#type").unwrap();
        let normal: syn::Ident = syn::parse_str("symbol").unwrap();

        assert_eq!(raw.to_string(), "r#type");
        assert_eq!(column_name_for_ident(&raw), "type");
        assert_eq!(column_name_for_ident(&normal), "symbol");
    }
}

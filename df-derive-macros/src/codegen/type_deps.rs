use crate::ir::StructIR;
use quote::ToTokens;
use syn::{GenericArgument, GenericParam, Ident, PathArguments, Type};

#[allow(clippy::struct_field_names)]
pub(in crate::codegen) struct GenericContext {
    type_params: Vec<Ident>,
    const_params: Vec<Ident>,
    lifetime_params: Vec<Ident>,
}

impl GenericContext {
    pub(in crate::codegen) fn new(ir: &StructIR) -> Self {
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

#[derive(PartialEq, Eq)]
struct TypeKey(String);

impl TypeKey {
    fn new(ty: &Type) -> Self {
        Self(ty.to_token_stream().to_string())
    }
}

pub(in crate::codegen) fn push_unique_type(out: &mut Vec<Type>, ty: &Type) {
    let key = TypeKey::new(ty);
    if !out.iter().any(|existing| TypeKey::new(existing) == key) {
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

pub(in crate::codegen) fn type_depends_on_generics(
    ty: &Type,
    generic_ctx: &GenericContext,
) -> bool {
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

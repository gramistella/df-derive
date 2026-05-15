use crate::attrs::LeafOverride;
use crate::type_analysis::AnalyzedBase;
use syn::Ident;

pub(super) fn decimal_generic_params_for_override(
    override_: Option<&LeafOverride>,
    base: &AnalyzedBase,
) -> Vec<Ident> {
    match (override_, base) {
        (Some(LeafOverride::Decimal { .. }), AnalyzedBase::Generic(ident)) => vec![ident.clone()],
        _ => Vec::new(),
    }
}

pub(super) fn decimal_backend_ty_for_override(
    override_: Option<&LeafOverride>,
    base: &AnalyzedBase,
) -> Option<syn::Type> {
    match (override_, base) {
        (Some(LeafOverride::Decimal { .. }), AnalyzedBase::Struct(ty)) => Some(ty.clone()),
        _ => None,
    }
}

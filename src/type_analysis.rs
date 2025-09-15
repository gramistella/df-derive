use crate::ir::{BaseType, PrimitiveTransform, Wrapper};
use syn::{GenericArgument, PathArguments, Type};

#[derive(Clone)]
pub struct AnalyzedType {
    pub base: BaseType,
    pub wrappers: Vec<Wrapper>,
    pub transform: Option<PrimitiveTransform>,
}

pub fn analyze_type(ty: &Type, as_string: bool) -> Result<AnalyzedType, syn::Error> {
    let mut wrappers: Vec<Wrapper> = Vec::new();
    let mut current_type = ty;

    // Loop to peel off wrappers in any order
    loop {
        if let Some(inner_ty) = extract_inner_type(current_type, "Option") {
            wrappers.push(Wrapper::Option);
            current_type = inner_ty;
            continue;
        }
        if let Some(inner_ty) = extract_inner_type(current_type, "Vec") {
            wrappers.push(Wrapper::Vec);
            current_type = inner_ty;
            continue;
        }
        // No more wrappers found, break the loop
        break;
    }

    // Before resolving the base type, detect some explicitly unsupported types
    if let Type::Path(type_path) = current_type
        && let Some(segment) = type_path.path.segments.last()
        && segment.ident == "HashMap"
    {
        return Err(syn::Error::new_spanned(
            current_type,
            "df-derive does not support HashMap",
        ));
    }

    let base = analyze_base_type(current_type)
        .ok_or_else(|| syn::Error::new_spanned(current_type, "Unsupported field type"))?;

    // Determine abstract transform; attribute stringification overrides
    let transform = if as_string {
        Some(PrimitiveTransform::ToString)
    } else {
        match &base {
            BaseType::DateTimeUtc => Some(PrimitiveTransform::DateTimeToMillis),
            BaseType::Decimal => Some(PrimitiveTransform::DecimalToString),
            _ => None,
        }
    };

    Ok(AnalyzedType {
        base,
        wrappers,
        transform,
    })
}

fn analyze_base_type(ty: &Type) -> Option<BaseType> {
    if is_datetime_utc(ty) {
        return Some(BaseType::DateTimeUtc);
    }
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.last()
    {
        let type_ident = &segment.ident;
        let base_type = match type_ident.to_string().as_str() {
            "String" => BaseType::String,
            "f64" => BaseType::F64,
            "f32" => BaseType::F32,
            "i8" => BaseType::I8,
            "u8" => BaseType::U8,
            "i16" => BaseType::I16,
            "u16" => BaseType::U16,
            "i64" => BaseType::I64,
            "isize" => BaseType::ISize,
            "u64" => BaseType::U64,
            "usize" => BaseType::USize,
            "u32" => BaseType::U32,
            "i32" => BaseType::I32,
            "bool" => BaseType::Bool,
            "Decimal" => BaseType::Decimal,
            _ => BaseType::Struct(type_ident.clone()),
        };
        return Some(base_type);
    }
    None
}

fn extract_inner_type<'a>(ty: &'a Type, wrapper: &str) -> Option<&'a Type> {
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.last()
        && segment.ident == wrapper
        && let PathArguments::AngleBracketed(args) = &segment.arguments
        && let Some(GenericArgument::Type(inner_ty)) = args.args.first()
    {
        return Some(inner_ty);
    }
    None
}

fn is_datetime_utc(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.last()
        && segment.ident == "DateTime"
        && let PathArguments::AngleBracketed(args) = &segment.arguments
        && let Some(GenericArgument::Type(Type::Path(inner))) = args.args.first()
        && let Some(inner_seg) = inner.path.segments.last()
    {
        return inner_seg.ident == "Utc";
    }
    false
}

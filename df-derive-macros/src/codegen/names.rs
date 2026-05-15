use syn::Ident;

pub(in crate::codegen) fn column_name_for_ident(ident: &Ident) -> String {
    let name = ident.to_string();
    name.strip_prefix("r#").unwrap_or(&name).to_owned()
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

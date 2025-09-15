use crate::ir::{BaseType, PrimitiveTransform, StructIR, Wrapper};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

// Recreate light-weight IR mirrors for internal use to avoid coupling to old FieldKind
#[derive(Clone)]
pub(super) struct PrimitiveIR {
    base_type: BaseType,
    transform: Option<PrimitiveTransform>,
}
#[derive(Clone)]
pub(super) struct NestedIR {
    type_ident: Ident,
}

// Granular traits per concern
pub trait SchemaProvider {
    fn gen_schema_entries(&self) -> TokenStream;
}
pub trait RowWiseGenerator {
    fn gen_series_creation(&self) -> TokenStream;
    fn gen_empty_series_creation(&self) -> TokenStream;
    fn gen_anyvalue_conversion(&self) -> TokenStream;
}
/// Unified column-populator trait used by both columnar and vec-anyvalues paths.
pub trait ColumnPopulator {
    fn gen_populator_inits(&self, idx: usize) -> Vec<TokenStream>;
    fn gen_populator_push(&self, it_ident: &Ident, idx: usize) -> TokenStream;
}

/// Finalization steps differ between the two consumers
pub trait VecAnyvaluesFinisher {
    fn gen_vec_values_finishers(&self, idx: usize) -> TokenStream;
}
pub trait ColumnarBuilderFinisher {
    fn gen_columnar_builders(&self, idx: usize) -> Vec<TokenStream>;
}

pub enum Strategy {
    Primitive(PrimitiveStrategy),
    Nested(NestedStructStrategy),
}

impl SchemaProvider for Strategy {
    fn gen_schema_entries(&self) -> TokenStream {
        match self {
            Self::Primitive(s) => s.gen_schema_entries(),
            Self::Nested(s) => s.gen_schema_entries(),
        }
    }
}
impl RowWiseGenerator for Strategy {
    fn gen_series_creation(&self) -> TokenStream {
        match self {
            Self::Primitive(s) => s.gen_series_creation(),
            Self::Nested(s) => s.gen_series_creation(),
        }
    }
    fn gen_empty_series_creation(&self) -> TokenStream {
        match self {
            Self::Primitive(s) => s.gen_empty_series_creation(),
            Self::Nested(s) => s.gen_empty_series_creation(),
        }
    }
    fn gen_anyvalue_conversion(&self) -> TokenStream {
        match self {
            Self::Primitive(s) => s.gen_anyvalue_conversion(),
            Self::Nested(s) => s.gen_anyvalue_conversion(),
        }
    }
}
impl ColumnPopulator for Strategy {
    fn gen_populator_inits(&self, idx: usize) -> Vec<TokenStream> {
        match self {
            Self::Primitive(s) => s.gen_populator_inits(idx),
            Self::Nested(s) => s.gen_populator_inits(idx),
        }
    }
    fn gen_populator_push(&self, it_ident: &Ident, idx: usize) -> TokenStream {
        match self {
            Self::Primitive(s) => s.gen_populator_push(it_ident, idx),
            Self::Nested(s) => s.gen_populator_push(it_ident, idx),
        }
    }
}

impl VecAnyvaluesFinisher for Strategy {
    fn gen_vec_values_finishers(&self, idx: usize) -> TokenStream {
        match self {
            Self::Primitive(s) => s.gen_vec_values_finishers(idx),
            Self::Nested(s) => s.gen_vec_values_finishers(idx),
        }
    }
}
impl ColumnarBuilderFinisher for Strategy {
    fn gen_columnar_builders(&self, idx: usize) -> Vec<TokenStream> {
        match self {
            Self::Primitive(s) => s.gen_columnar_builders(idx),
            Self::Nested(s) => s.gen_columnar_builders(idx),
        }
    }
}

pub fn build_strategies(ir: &StructIR) -> Vec<Strategy> {
    ir.fields
        .iter()
        .map(|f| match &f.base_type {
            BaseType::Struct(type_ident) => {
                // If transform indicates stringification, treat as primitive
                if matches!(f.transform, Some(PrimitiveTransform::ToString)) {
                    Strategy::Primitive(PrimitiveStrategy::new(
                        f.name.clone(),
                        f.field_index,
                        f.wrappers.clone(),
                        PrimitiveIR {
                            base_type: f.base_type.clone(),
                            transform: f.transform.clone(),
                        },
                    ))
                } else {
                    Strategy::Nested(NestedStructStrategy::new(
                        f.name.clone(),
                        f.field_index,
                        f.wrappers.clone(),
                        NestedIR {
                            type_ident: type_ident.clone(),
                        },
                    ))
                }
            }
            _ => Strategy::Primitive(PrimitiveStrategy::new(
                f.name.clone(),
                f.field_index,
                f.wrappers.clone(),
                PrimitiveIR {
                    base_type: f.base_type.clone(),
                    transform: f.transform.clone(),
                },
            )),
        })
        .collect()
}

pub struct PrimitiveStrategy {
    field_ident: Ident,
    field_index: Option<usize>,
    field_name: String,
    wrappers: Vec<Wrapper>,
    p: PrimitiveIR,
}

impl PrimitiveStrategy {
    pub fn new(
        field_ident: Ident,
        field_index: Option<usize>,
        wrappers: Vec<Wrapper>,
        p: PrimitiveIR,
    ) -> Self {
        let field_name = field_ident.to_string();
        Self {
            field_ident,
            field_index,
            field_name,
            wrappers,
            p,
        }
    }
}

impl SchemaProvider for PrimitiveStrategy {
    fn gen_schema_entries(&self) -> TokenStream {
        let mapping = crate::codegen::type_registry::compute_mapping(
            &self.p.base_type,
            self.p.transform.as_ref(),
            &self.wrappers,
        );
        let dtype = mapping.full_dtype;
        let name = &self.field_name;
        quote! { vec![(#name, #dtype)] }
    }
}

impl RowWiseGenerator for PrimitiveStrategy {
    fn gen_series_creation(&self) -> TokenStream {
        let name = &self.field_name;
        let access = self.field_index.map_or_else(
            || {
                let field_ident = &self.field_ident;
                quote! { self.#field_ident }
            },
            |index| {
                let index_lit = syn::Index::from(index);
                quote! { self.#index_lit }
            },
        );
        let p = &self.p;
        super::wrapped_codegen::generate_primitive_for_series(
            name,
            &access,
            &p.base_type,
            p.transform.as_ref(),
            &self.wrappers,
        )
    }

    fn gen_empty_series_creation(&self) -> TokenStream {
        let mapping = crate::codegen::type_registry::compute_mapping(
            &self.p.base_type,
            self.p.transform.as_ref(),
            &self.wrappers,
        );
        let dtype = mapping.full_dtype;
        let name = &self.field_name;
        quote! { vec![polars::prelude::Series::new_empty(#name.into(), &#dtype).into()] }
    }

    fn gen_anyvalue_conversion(&self) -> TokenStream {
        let p = &self.p;
        let access = self.field_index.map_or_else(
            || {
                let field_ident = &self.field_ident;
                quote! { self.#field_ident }
            },
            |index| {
                let index_lit = syn::Index::from(index);
                quote! { self.#index_lit }
            },
        );
        super::wrapped_codegen::generate_primitive_for_anyvalue(
            &format_ident!("values"),
            &access,
            &p.base_type,
            p.transform.as_ref(),
            &self.wrappers,
        )
    }
}

impl VecAnyvaluesFinisher for PrimitiveStrategy {
    fn gen_vec_values_finishers(&self, idx: usize) -> TokenStream {
        super::wrapped_codegen::primitive_finishers_for_vec_anyvalues(
            &self.wrappers,
            &self.p.base_type,
            self.p.transform.as_ref(),
            idx,
        )
    }
}

impl ColumnPopulator for PrimitiveStrategy {
    fn gen_populator_inits(&self, idx: usize) -> Vec<TokenStream> {
        super::wrapped_codegen::primitive_decls(
            &self.wrappers,
            &self.p.base_type,
            self.p.transform.as_ref(),
            idx,
        )
    }

    fn gen_populator_push(&self, it_ident: &Ident, idx: usize) -> TokenStream {
        let access = self.field_index.map_or_else(
            || {
                let field_ident = &self.field_ident;
                quote! { #it_ident.#field_ident }
            },
            |index| {
                let index_lit = syn::Index::from(index);
                quote! { #it_ident.#index_lit }
            },
        );
        super::wrapped_codegen::generate_primitive_for_columnar_push(
            &access,
            &self.p.base_type,
            self.p.transform.as_ref(),
            &self.wrappers,
            idx,
        )
    }
}

impl ColumnarBuilderFinisher for PrimitiveStrategy {
    fn gen_columnar_builders(&self, idx: usize) -> Vec<TokenStream> {
        let name = &self.field_name;
        if self.wrappers.iter().any(|w| matches!(w, Wrapper::Vec)) {
            let vals_ident = format_ident!("__df_derive_pv_vals_{}", idx);
            vec![quote! {{
                let s = polars::prelude::Series::new(#name.into(), &#vals_ident);
                columns.push(s.into());
            }}]
        } else {
            let vec_ident = format_ident!("__df_derive_buf_{}", idx);
            vec![quote! {{
                let s = polars::prelude::Series::new(#name.into(), &#vec_ident);
                columns.push(s.into());
            }}]
        }
    }
}

pub struct NestedStructStrategy {
    field_ident: Ident,
    field_index: Option<usize>,
    field_name: String,
    wrappers: Vec<Wrapper>,
    n: NestedIR,
}

impl NestedStructStrategy {
    pub fn new(
        field_ident: Ident,
        field_index: Option<usize>,
        wrappers: Vec<Wrapper>,
        n: NestedIR,
    ) -> Self {
        let field_name = field_ident.to_string();
        Self {
            field_ident,
            field_index,
            field_name,
            wrappers,
            n,
        }
    }
}

impl SchemaProvider for NestedStructStrategy {
    fn gen_schema_entries(&self) -> TokenStream {
        let name = &self.field_name;
        super::wrapped_codegen::generate_schema_entries_for_struct(
            &self.n.type_ident,
            name,
            self.wrappers.iter().any(|w| matches!(w, Wrapper::Vec)),
        )
    }
}

impl RowWiseGenerator for NestedStructStrategy {
    fn gen_series_creation(&self) -> TokenStream {
        let name = &self.field_name;
        let access = self.field_index.map_or_else(
            || {
                let field_ident = &self.field_ident;
                quote! { self.#field_ident }
            },
            |index| {
                let index_lit = syn::Index::from(index);
                quote! { self.#index_lit }
            },
        );
        super::wrapped_codegen::generate_nested_for_series(
            &self.n.type_ident,
            name,
            &access,
            &self.wrappers,
        )
    }

    fn gen_empty_series_creation(&self) -> TokenStream {
        let name = &self.field_name;
        super::wrapped_codegen::nested_empty_series_row(&self.n.type_ident, name, &self.wrappers)
    }

    fn gen_anyvalue_conversion(&self) -> TokenStream {
        let access = self.field_index.map_or_else(
            || {
                let field_ident = &self.field_ident;
                quote! { self.#field_ident }
            },
            |index| {
                let index_lit = syn::Index::from(index);
                quote! { self.#index_lit }
            },
        );
        super::wrapped_codegen::generate_nested_for_anyvalue(
            &self.n.type_ident,
            &format_ident!("values"),
            &access,
            &self.wrappers,
        )
    }
}

impl ColumnPopulator for NestedStructStrategy {
    fn gen_populator_inits(&self, idx: usize) -> Vec<TokenStream> {
        super::wrapped_codegen::nested_decls(&self.wrappers, &self.n.type_ident, idx)
    }

    fn gen_populator_push(&self, it_ident: &Ident, idx: usize) -> TokenStream {
        let access = self.field_index.map_or_else(
            || {
                let field_ident = &self.field_ident;
                quote! { #it_ident.#field_ident }
            },
            |index| {
                let index_lit = syn::Index::from(index);
                quote! { #it_ident.#index_lit }
            },
        );
        super::wrapped_codegen::generate_nested_for_columnar_push(
            &self.n.type_ident,
            &access,
            &self.wrappers,
            idx,
        )
    }
}

impl VecAnyvaluesFinisher for NestedStructStrategy {
    fn gen_vec_values_finishers(&self, idx: usize) -> TokenStream {
        super::wrapped_codegen::nested_finishers_for_vec_anyvalues(&self.wrappers, idx)
    }
}

impl ColumnarBuilderFinisher for NestedStructStrategy {
    fn gen_columnar_builders(&self, idx: usize) -> Vec<TokenStream> {
        super::wrapped_codegen::nested_columnar_builders(&self.wrappers, idx, &self.field_name)
    }
}

// Helpers moved to centralized wrapping.rs

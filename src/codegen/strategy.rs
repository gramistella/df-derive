use crate::ir::{BaseType, PrimitiveTransform, StructIR, Wrapper, has_vec};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

use super::wrapped_codegen::PopulatorIdents;

// Recreate light-weight IR mirrors for internal use to avoid coupling to old FieldKind
#[derive(Clone)]
pub(super) struct PrimitiveIR {
    base_type: BaseType,
    transform: Option<PrimitiveTransform>,
}
#[derive(Clone)]
pub(super) struct NestedIR {
    /// Fully-qualified call path for the field's base type, prebuilt to be
    /// spliced into either expression position (`#type_path::method()`) or
    /// type position (`Vec<#type_path>`). For a non-generic struct this is
    /// just the bare ident; for a generic struct used at this field it
    /// includes turbofish args, e.g. `Foo::<M>`. Generic type parameters
    /// also use this form (just the param ident) since the macro injects
    /// the trait bounds that make `T::method()` resolve.
    type_path: TokenStream,
    /// Fully-qualified path to the `Columnar` trait, prebuilt for UFCS calls
    /// (`<#type_path as #columnar_trait>::columnar_to_dataframe(&flat)`). The
    /// bulk emitters use this so they work for both generic-parameter base
    /// types (where the trait is bound on the parameter) and concrete struct
    /// base types (where the trait isn't necessarily in scope at the call
    /// site, e.g. inside the inherent `__df_derive_columnar_from_refs` body).
    columnar_trait: TokenStream,
    /// Fully-qualified path to the `ToDataFrame` trait, prebuilt for UFCS
    /// calls to `schema()` from inside bulk-emit token streams. Same scope
    /// problem as `columnar_trait`: bulk-emit tokens are inlined into bodies
    /// (e.g. inherent helpers) where `ToDataFrame` may not be in scope.
    to_df_trait: TokenStream,
    is_generic: bool,
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

/// Optional override that lets a strategy bypass the per-row decls/push/builders
/// triple in the columnar path and emit a single bulk builder instead. Used by
/// generic-leaf fields where calling `T::columnar_to_dataframe` once on a
/// collected slice is dramatically faster than building one tiny `DataFrame` per
/// item.
pub trait ColumnarBulkEmitter {
    fn gen_bulk_columnar_emit(&self, idx: usize) -> Option<Vec<TokenStream>>;
}

/// Mirror of `ColumnarBulkEmitter` for the `__df_derive_vec_to_inner_list_values`
/// helper path: when a strategy can produce its `AnyValue::List` entries in bulk
/// (without a per-row push), it returns Some.
pub trait VecAnyvaluesBulkEmitter {
    fn gen_bulk_vec_anyvalues_emit(&self, idx: usize) -> Option<Vec<TokenStream>>;
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

impl ColumnarBulkEmitter for Strategy {
    fn gen_bulk_columnar_emit(&self, idx: usize) -> Option<Vec<TokenStream>> {
        match self {
            Self::Primitive(_) => None,
            Self::Nested(s) => s.gen_bulk_columnar_emit(idx),
        }
    }
}

impl VecAnyvaluesBulkEmitter for Strategy {
    fn gen_bulk_vec_anyvalues_emit(&self, idx: usize) -> Option<Vec<TokenStream>> {
        match self {
            Self::Primitive(_) => None,
            Self::Nested(s) => s.gen_bulk_vec_anyvalues_emit(idx),
        }
    }
}

pub fn build_strategies(ir: &StructIR, config: &super::MacroConfig) -> Vec<Strategy> {
    let columnar_trait = &config.columnar_trait_path;
    let to_df_trait = &config.to_dataframe_trait_path;
    ir.fields
        .iter()
        .map(|f| match &f.base_type {
            BaseType::Struct(type_ident, type_args) => {
                if matches!(
                    f.transform,
                    Some(PrimitiveTransform::ToString | PrimitiveTransform::AsStr)
                ) {
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
                            type_path: build_type_path(type_ident, type_args.as_ref()),
                            columnar_trait: columnar_trait.clone(),
                            to_df_trait: to_df_trait.clone(),
                            is_generic: false,
                        },
                    ))
                }
            }
            BaseType::Generic(type_ident) => {
                if matches!(
                    f.transform,
                    Some(PrimitiveTransform::ToString | PrimitiveTransform::AsStr)
                ) {
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
                            type_path: quote! { #type_ident },
                            columnar_trait: columnar_trait.clone(),
                            to_df_trait: to_df_trait.clone(),
                            is_generic: true,
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

/// Build the type-as-path token stream used by `NestedIR`. For a struct
/// referenced without args (e.g. `Address`) this is just the bare ident; for
/// a generic struct referenced with args (e.g. `Foo<M>` or `Foo<M, N>`) it is
/// the turbofish form `Foo::<M, N>`, which is valid in both expression and
/// type position in modern Rust. The turbofish is necessary in expression
/// position (`Foo::<M>::schema()`) and accepted everywhere else.
pub(super) fn build_type_path(
    ident: &Ident,
    args: Option<&syn::AngleBracketedGenericArguments>,
) -> TokenStream {
    args.map_or_else(
        || quote! { #ident },
        |ab| {
            let inner = &ab.args;
            quote! { #ident::<#inner> }
        },
    )
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
        quote! { vec![(::std::string::String::from(#name), #dtype)] }
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
        if has_vec(&self.wrappers) {
            // The list builder was constructed with the correct inner dtype
            // (`outer_list_inner_dtype`), so the finished series already has
            // the schema-declared dtype — no cast needed here.
            let lb_ident = PopulatorIdents::primitive_list_builder(idx);
            vec![quote! {{
                let s = polars::prelude::IntoSeries::into_series(
                    polars::prelude::ListBuilderTrait::finish(&mut *#lb_ident),
                )
                .with_name(#name.into());
                columns.push(s.into());
            }}]
        } else {
            // Non-Vec primitive: the buffer holds the raw element type chosen
            // by `compute_mapping` (e.g. `Vec<i64>` for `DateTime<Utc>`,
            // `Vec<String>` for `Decimal`). For transforms whose schema dtype
            // differs from the buffer's natural Series dtype (`Datetime(...)`
            // and `Decimal(p, s)`), `needs_cast` returns true and we cast the
            // built Series to the schema dtype — matching what the row-wise
            // path does in `generate_primitive_for_series`. Without this cast,
            // the columnar/batch DataFrame's runtime dtype diverges from
            // `T::schema()`.
            let p = &self.p;
            let mapping = crate::codegen::type_registry::compute_mapping(
                &p.base_type,
                p.transform.as_ref(),
                &self.wrappers,
            );
            let dtype = mapping.full_dtype;
            let do_cast = crate::codegen::type_registry::needs_cast(p.transform.as_ref());
            let vec_ident = PopulatorIdents::primitive_buf(idx);
            vec![quote! {{
                let mut s = <polars::prelude::Series as polars::prelude::NamedFrom<_, _>>::new(#name.into(), &#vec_ident);
                if #do_cast { s = s.cast(&#dtype)?; }
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
            &self.n.type_path,
            name,
            has_vec(&self.wrappers),
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
            &self.n.type_path,
            name,
            &access,
            &self.wrappers,
            self.n.is_generic,
        )
    }

    fn gen_empty_series_creation(&self) -> TokenStream {
        let name = &self.field_name;
        super::wrapped_codegen::nested_empty_series_row(&self.n.type_path, name, &self.wrappers)
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
            &self.n.type_path,
            &format_ident!("values"),
            &access,
            &self.wrappers,
            self.n.is_generic,
        )
    }
}

impl ColumnPopulator for NestedStructStrategy {
    fn gen_populator_inits(&self, idx: usize) -> Vec<TokenStream> {
        super::wrapped_codegen::nested_decls(&self.wrappers, &self.n.type_path, idx)
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
            &self.n.type_path,
            &access,
            &self.wrappers,
            idx,
            self.n.is_generic,
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

impl NestedStructStrategy {
    /// Field-access expression rooted at the columnar-loop iterator
    /// (`__df_derive_it`). Hand-built here rather than reused from the per-row
    /// generator because the bulk path emits its own loop closures.
    fn it_access(&self) -> TokenStream {
        self.field_index.map_or_else(
            || {
                let id = &self.field_ident;
                quote! { __df_derive_it.#id }
            },
            |i| {
                let li = syn::Index::from(i);
                quote! { __df_derive_it.#li }
            },
        )
    }

    /// Returns `Some(emit)` when this strategy has a bulk implementation for
    /// the given context; `None` falls back to the per-row pipeline. Eligible
    /// for the depth-1 wrapper shapes the bulk helpers support: bare leaf,
    /// `Option<T>`, or `Vec<T>`.
    ///
    /// Dispatch:
    /// - Bare leaf / `Option<T>`: generic-only. Concrete nested structs keep
    ///   the per-row inherent `__df_derive_*` fast paths in
    ///   `generate_nested_for_*` — those already do typed-buffer work
    ///   per-call and the bulk overhead would be an allocation hit for
    ///   common cases.
    /// - `Vec<T>`: works for both. Generic uses the `T: Clone` bound to
    ///   build a `Vec<T>` and call `<T as Columnar>::columnar_to_dataframe`.
    ///   Concrete uses `Vec<&Inner>` and the inherent
    ///   `Inner::__df_derive_columnar_from_refs`, side-stepping any
    ///   `Inner: Clone` requirement on user struct types.
    fn try_bulk_emit(
        &self,
        idx: usize,
        ctx: super::wrapped_codegen::BulkContext<'_>,
    ) -> Option<Vec<TokenStream>> {
        let ty = &self.n.type_path;
        let columnar_trait = &self.n.columnar_trait;
        let to_df_trait = &self.n.to_df_trait;
        let access = self.it_access();
        let emit = match self.wrappers.as_slice() {
            [] if self.n.is_generic => super::wrapped_codegen::gen_bulk_generic_leaf(
                ty,
                columnar_trait,
                to_df_trait,
                idx,
                &access,
                ctx,
            ),
            [Wrapper::Option] if self.n.is_generic => {
                super::wrapped_codegen::gen_bulk_generic_option(
                    ty,
                    columnar_trait,
                    to_df_trait,
                    idx,
                    &access,
                    ctx,
                )
            }
            [Wrapper::Vec] if self.n.is_generic => super::wrapped_codegen::gen_bulk_generic_vec(
                ty,
                columnar_trait,
                to_df_trait,
                idx,
                &access,
                ctx,
            ),
            [Wrapper::Vec] => {
                super::wrapped_codegen::gen_bulk_concrete_vec(ty, to_df_trait, idx, &access, ctx)
            }
            _ => return None,
        };
        Some(vec![emit])
    }
}

impl ColumnarBulkEmitter for NestedStructStrategy {
    fn gen_bulk_columnar_emit(&self, idx: usize) -> Option<Vec<TokenStream>> {
        let parent_name = self.field_name.as_str();
        self.try_bulk_emit(
            idx,
            super::wrapped_codegen::BulkContext::Columnar { parent_name },
        )
    }
}

impl VecAnyvaluesBulkEmitter for NestedStructStrategy {
    fn gen_bulk_vec_anyvalues_emit(&self, idx: usize) -> Option<Vec<TokenStream>> {
        self.try_bulk_emit(idx, super::wrapped_codegen::BulkContext::VecAnyvalues)
    }
}

// Helpers moved to centralized wrapping.rs

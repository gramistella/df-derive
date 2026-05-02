use crate::ir::{BaseType, PrimitiveTransform, StructIR, Wrapper, has_vec};
use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use super::populator_idents::PopulatorIdents;

// Recreate light-weight IR mirrors for internal use to avoid coupling to old FieldKind
#[derive(Clone)]
pub(super) struct PrimitiveIR {
    base_type: BaseType,
    transform: Option<PrimitiveTransform>,
    /// Fully-qualified path to the `Decimal128Encode` trait, prebuilt for
    /// trait-method-call syntax inside emitted token streams. Only relevant
    /// for `BaseType::Decimal` fields with a `DecimalToInt128` transform —
    /// other base/transform combinations ignore the path. Stored on every
    /// `PrimitiveIR` regardless because cloning a `TokenStream` is cheap and
    /// it sidesteps an `Option<TokenStream>` everywhere downstream.
    decimal128_encode_trait: TokenStream,
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
    /// (`<#type_path as #columnar_trait>::columnar_from_refs(&refs)`). The
    /// bulk emitters use this so they work for both generic-parameter base
    /// types (where the trait is bound on the parameter) and concrete struct
    /// base types (where the trait isn't necessarily in scope at the call
    /// site).
    columnar_trait: TokenStream,
    /// Fully-qualified path to the `ToDataFrame` trait, prebuilt for UFCS
    /// calls to `schema()` from inside bulk-emit token streams. Same scope
    /// problem as `columnar_trait`: bulk-emit tokens are inlined into bodies
    /// where `ToDataFrame` may not be in scope.
    to_df_trait: TokenStream,
    is_generic: bool,
}

pub enum Strategy {
    Primitive(PrimitiveStrategy),
    Nested(NestedStructStrategy),
}

impl Strategy {
    pub fn gen_schema_entries(&self) -> TokenStream {
        match self {
            Self::Primitive(s) => s.gen_schema_entries(),
            Self::Nested(s) => s.gen_schema_entries(),
        }
    }

    pub fn gen_empty_series_creation(&self) -> TokenStream {
        match self {
            Self::Primitive(s) => s.gen_empty_series_creation(),
            Self::Nested(s) => s.gen_empty_series_creation(),
        }
    }

    pub fn gen_populator_inits(&self, idx: usize) -> Vec<TokenStream> {
        match self {
            Self::Primitive(s) => s.gen_populator_inits(idx),
            Self::Nested(s) => s.gen_populator_inits(idx),
        }
    }

    pub fn gen_populator_push(&self, it_ident: &Ident, idx: usize) -> TokenStream {
        match self {
            Self::Primitive(s) => s.gen_populator_push(it_ident, idx),
            Self::Nested(s) => s.gen_populator_push(it_ident, idx),
        }
    }

    pub fn gen_vec_values_finishers(&self, idx: usize) -> TokenStream {
        match self {
            Self::Primitive(s) => s.gen_vec_values_finishers(idx),
            Self::Nested(s) => s.gen_vec_values_finishers(idx),
        }
    }

    pub fn gen_columnar_builders(&self, idx: usize) -> Vec<TokenStream> {
        match self {
            Self::Primitive(s) => s.gen_columnar_builders(idx),
            Self::Nested(s) => s.gen_columnar_builders(idx),
        }
    }

    /// Per-field codegen that pushes one `AnyValue` per inner schema column
    /// onto `values_vec_ident` for a single instance bound to `it_ident`.
    /// Used by the `to_inner_values(&self)` trait override to bypass the
    /// `to_dataframe()` round-trip: the outer struct's override invokes this
    /// per field, and the nested `on_leaf` recursively calls the inner
    /// type's `to_inner_values()` instead of constructing a one-row
    /// `DataFrame`.
    pub fn gen_for_anyvalue(&self, it_ident: &Ident, values_vec_ident: &Ident) -> TokenStream {
        match self {
            Self::Primitive(s) => s.gen_for_anyvalue(it_ident, values_vec_ident),
            Self::Nested(s) => s.gen_for_anyvalue(it_ident, values_vec_ident),
        }
    }

    /// Returns `Some` when this field can bypass the per-row decls/push/builders
    /// triple in the columnar path and emit a single bulk builder. Used by
    /// generic-leaf fields where calling `T::columnar_to_dataframe` once on a
    /// collected slice is dramatically faster than building one tiny `DataFrame`
    /// per item, and by `Vec<Vec<#native>>` primitive fields where flattening
    /// straight into a `LargeListArray<LargeListArray<PrimitiveArray>>` skips
    /// the per-parent-row `Box<dyn ListBuilderTrait>::append_series` round-trip.
    ///
    /// `pa_root` is the cached `polars-arrow` crate-root token stream from
    /// the per-derive entry point; bulk emitters use it to reach
    /// `OffsetsBuffer`, `LargeListArray`, and `PrimitiveArray`. Pass it down
    /// so each per-field codegen call doesn't re-run
    /// `proc_macro_crate::crate_name`.
    pub fn gen_bulk_columnar_emit(
        &self,
        pa_root: &TokenStream,
        idx: usize,
    ) -> Option<Vec<TokenStream>> {
        match self {
            Self::Primitive(s) => s.gen_bulk_columnar_emit(pa_root, idx),
            Self::Nested(s) => s.gen_bulk_columnar_emit(pa_root, idx),
        }
    }

    /// Mirror of `gen_bulk_columnar_emit` for the
    /// `__df_derive_vec_to_inner_list_values` helper path: when a strategy can
    /// produce its `AnyValue::List` entries in bulk (without a per-row push),
    /// returns `Some`. Primitive fields return `Some` only for the
    /// `Vec<Vec<#native>>` shape over a bare numeric primitive base.
    pub fn gen_bulk_vec_anyvalues_emit(
        &self,
        pa_root: &TokenStream,
        idx: usize,
    ) -> Option<Vec<TokenStream>> {
        match self {
            Self::Primitive(s) => s.gen_bulk_vec_anyvalues_emit(pa_root, idx),
            Self::Nested(s) => s.gen_bulk_vec_anyvalues_emit(pa_root, idx),
        }
    }
}

pub fn build_strategies(ir: &StructIR, config: &super::MacroConfig) -> Vec<Strategy> {
    let columnar_trait = &config.columnar_trait_path;
    let to_df_trait = &config.to_dataframe_trait_path;
    let decimal_encode = &config.decimal128_encode_trait_path;
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
                            decimal128_encode_trait: decimal_encode.clone(),
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
                            decimal128_encode_trait: decimal_encode.clone(),
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
                    decimal128_encode_trait: decimal_encode.clone(),
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

    fn gen_schema_entries(&self) -> TokenStream {
        let mapping = crate::codegen::type_registry::compute_mapping(
            &self.p.base_type,
            self.p.transform.as_ref(),
            &self.wrappers,
        );
        let dtype = mapping.full_dtype;
        let name = &self.field_name;
        quote! { ::std::vec![(::std::string::String::from(#name), #dtype)] }
    }

    fn gen_empty_series_creation(&self) -> TokenStream {
        let mapping = crate::codegen::type_registry::compute_mapping(
            &self.p.base_type,
            self.p.transform.as_ref(),
            &self.wrappers,
        );
        let dtype = mapping.full_dtype;
        let name = &self.field_name;
        let pp = super::polars_paths::prelude();
        quote! { ::std::vec![#pp::Series::new_empty(#name.into(), &#dtype).into()] }
    }

    fn gen_vec_values_finishers(&self, idx: usize) -> TokenStream {
        super::primitive::primitive_finishers_for_vec_anyvalues(
            &self.wrappers,
            &self.p.base_type,
            self.p.transform.as_ref(),
            idx,
        )
    }

    fn gen_populator_inits(&self, idx: usize) -> Vec<TokenStream> {
        super::primitive::primitive_decls(
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
        super::primitive::generate_primitive_for_columnar_push(
            &access,
            &self.p.base_type,
            self.p.transform.as_ref(),
            &self.wrappers,
            idx,
            &self.p.decimal128_encode_trait,
        )
    }

    fn gen_for_anyvalue(&self, it_ident: &Ident, values_vec_ident: &Ident) -> TokenStream {
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
        super::primitive::generate_primitive_for_anyvalue(
            values_vec_ident,
            &access,
            &self.p.base_type,
            self.p.transform.as_ref(),
            &self.wrappers,
            &self.p.decimal128_encode_trait,
        )
    }

    /// Field-access expression rooted at the columnar/anyvalues bulk-loop
    /// iterator (`__df_derive_it`). Mirrors the helper of the same name on
    /// `NestedStructStrategy` so the bulk-emit path can build its own loop
    /// without referencing the per-row `gen_populator_push` access.
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

    fn gen_bulk_columnar_emit(
        &self,
        pa_root: &TokenStream,
        _idx: usize,
    ) -> Option<Vec<TokenStream>> {
        let access = self.it_access();
        let parent_name = Some(self.field_name.as_str());
        super::primitive::try_gen_vec_vec_string_emit(
            pa_root,
            &access,
            &self.p.base_type,
            self.p.transform.as_ref(),
            &self.wrappers,
            parent_name,
        )
        .or_else(|| {
            super::primitive::try_gen_vec_vec_option_numeric_emit(
                pa_root,
                &access,
                &self.p.base_type,
                self.p.transform.as_ref(),
                &self.wrappers,
                parent_name,
            )
        })
        .or_else(|| {
            super::primitive::try_gen_vec_vec_option_string_emit(
                pa_root,
                &access,
                &self.p.base_type,
                self.p.transform.as_ref(),
                &self.wrappers,
                parent_name,
            )
        })
        .or_else(|| {
            super::primitive::try_gen_vec_vec_bool_emit(
                pa_root,
                &access,
                &self.p.base_type,
                self.p.transform.as_ref(),
                &self.wrappers,
                parent_name,
            )
        })
        .or_else(|| {
            super::primitive::try_gen_vec_option_numeric_emit(
                pa_root,
                &access,
                &self.p.base_type,
                self.p.transform.as_ref(),
                &self.wrappers,
                parent_name,
            )
        })
        .or_else(|| {
            super::primitive::try_gen_vec_option_string_emit(
                pa_root,
                &access,
                &self.p.base_type,
                self.p.transform.as_ref(),
                &self.wrappers,
                parent_name,
            )
        })
        .or_else(|| {
            super::primitive::try_gen_vec_option_bool_emit(
                pa_root,
                &access,
                &self.p.base_type,
                self.p.transform.as_ref(),
                &self.wrappers,
                parent_name,
            )
        })
        .or_else(|| {
            super::primitive::try_gen_nested_primitive_vec_emit(
                pa_root,
                &access,
                &self.p.base_type,
                self.p.transform.as_ref(),
                &self.wrappers,
                parent_name,
            )
        })
        .or_else(|| {
            super::primitive::try_gen_vec_bool_emit(
                pa_root,
                &access,
                &self.p.base_type,
                self.p.transform.as_ref(),
                &self.wrappers,
                parent_name,
            )
        })
        .map(|emit| vec![emit])
    }

    fn gen_bulk_vec_anyvalues_emit(
        &self,
        pa_root: &TokenStream,
        _idx: usize,
    ) -> Option<Vec<TokenStream>> {
        let access = self.it_access();
        super::primitive::try_gen_vec_vec_string_emit(
            pa_root,
            &access,
            &self.p.base_type,
            self.p.transform.as_ref(),
            &self.wrappers,
            None,
        )
        .or_else(|| {
            super::primitive::try_gen_vec_vec_option_numeric_emit(
                pa_root,
                &access,
                &self.p.base_type,
                self.p.transform.as_ref(),
                &self.wrappers,
                None,
            )
        })
        .or_else(|| {
            super::primitive::try_gen_vec_vec_option_string_emit(
                pa_root,
                &access,
                &self.p.base_type,
                self.p.transform.as_ref(),
                &self.wrappers,
                None,
            )
        })
        .or_else(|| {
            super::primitive::try_gen_vec_vec_bool_emit(
                pa_root,
                &access,
                &self.p.base_type,
                self.p.transform.as_ref(),
                &self.wrappers,
                None,
            )
        })
        .or_else(|| {
            super::primitive::try_gen_vec_option_numeric_emit(
                pa_root,
                &access,
                &self.p.base_type,
                self.p.transform.as_ref(),
                &self.wrappers,
                None,
            )
        })
        .or_else(|| {
            super::primitive::try_gen_vec_option_string_emit(
                pa_root,
                &access,
                &self.p.base_type,
                self.p.transform.as_ref(),
                &self.wrappers,
                None,
            )
        })
        .or_else(|| {
            super::primitive::try_gen_vec_option_bool_emit(
                pa_root,
                &access,
                &self.p.base_type,
                self.p.transform.as_ref(),
                &self.wrappers,
                None,
            )
        })
        .or_else(|| {
            super::primitive::try_gen_nested_primitive_vec_emit(
                pa_root,
                &access,
                &self.p.base_type,
                self.p.transform.as_ref(),
                &self.wrappers,
                None,
            )
        })
        .or_else(|| {
            super::primitive::try_gen_vec_bool_emit(
                pa_root,
                &access,
                &self.p.base_type,
                self.p.transform.as_ref(),
                &self.wrappers,
                None,
            )
        })
        .map(|emit| vec![emit])
    }

    fn gen_columnar_builders(&self, idx: usize) -> Vec<TokenStream> {
        let name = &self.field_name;
        let pp = super::polars_paths::prelude();
        if has_vec(&self.wrappers) {
            // The list builder was constructed with the correct inner dtype
            // (`outer_list_inner_dtype` / `inner_logical`), so the finished
            // series already has the schema-declared dtype — no cast needed
            // here. Typed `ListPrimitiveChunkedBuilder<Native>` finishers call
            // `ListBuilderTrait::finish(&mut self)` directly; the boxed-dyn
            // path needs the `&mut *` deref through the Box.
            let lb_ident = PopulatorIdents::primitive_list_builder(idx);
            let builder_ref = if super::primitive::typed_primitive_list_info(
                &self.p.base_type,
                self.p.transform.as_ref(),
                &self.wrappers,
            )
            .is_some()
            {
                quote! { &mut #lb_ident }
            } else {
                quote! { &mut *#lb_ident }
            };
            return vec![quote! {{
                let s = #pp::IntoSeries::into_series(
                    #pp::ListBuilderTrait::finish(#builder_ref),
                )
                .with_name(#name.into());
                columns.push(s.into());
            }}];
        }
        if let Some(emit) = self.gen_columnar_direct_fast_path_finish(idx, name) {
            return vec![emit];
        }

        // Decimal: the buffer is `Vec<i128>` / `Vec<Option<i128>>` already
        // rescaled to the schema scale. Build the `DecimalChunked` directly
        // via `Int128Chunked::into_decimal_unchecked` — skips the
        // `Series::new(&Vec<i128>) → cast(Decimal)` round-trip.
        if let (BaseType::Decimal, Some(PrimitiveTransform::DecimalToInt128 { precision, scale })) =
            (&self.p.base_type, self.p.transform.as_ref())
        {
            let p = *precision as usize;
            let s = *scale as usize;
            let vec_ident = PopulatorIdents::primitive_buf(idx);
            let int128 = super::polars_paths::int128_chunked();
            let opt = crate::ir::has_option(&self.wrappers);
            return if opt {
                vec![quote! {{
                    let ca = <#int128 as #pp::NewChunkedArray<_, _>>::from_iter_options(
                        #name.into(),
                        #vec_ident.into_iter(),
                    );
                    let s = #pp::IntoSeries::into_series(
                        ca.into_decimal_unchecked(#p, #s),
                    );
                    columns.push(s.into());
                }}]
            } else {
                vec![quote! {{
                    let ca = #int128::from_vec(#name.into(), #vec_ident);
                    let s = #pp::IntoSeries::into_series(
                        ca.into_decimal_unchecked(#p, #s),
                    );
                    columns.push(s.into());
                }}]
            };
        }
        // Non-Vec primitive: the buffer holds the raw element type chosen
        // by `compute_mapping` (e.g. `Vec<i64>` for `DateTime<Utc>`). For
        // transforms whose schema dtype differs from the buffer's natural
        // Series dtype (`Datetime(...)`), `needs_cast` returns true and we
        // cast the built Series to the schema dtype — matching what the
        // row-wise path does in `generate_primitive_for_series`. Without
        // this cast, the columnar/batch DataFrame's runtime dtype diverges
        // from `T::schema()`.
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
            let mut s = <#pp::Series as #pp::NamedFrom<_, _>>::new(#name.into(), &#vec_ident);
            if #do_cast { s = s.cast(&#dtype)?; }
            columns.push(s.into());
        }}]
    }

    /// Try each direct-finish helper for the non-`Vec` primitive shapes that
    /// have a fast path: bare-`String`/`as_string`, `Option<String>`/
    /// `Option<as_string>`, bare numeric / `Option<numeric>`, and
    /// `Option<bool>`. Each helper returns `None` for shapes outside its
    /// carve-out, so the first `Some` wins. Same column dtype as the
    /// `Series::new(name, &Vec<…>)` slow path in every case, minus the
    /// second walk (and per-row validity branch where applicable). Returns
    /// `None` only when no fast path applies — caller falls through to the
    /// generic owning-buffer finisher.
    fn gen_columnar_direct_fast_path_finish(&self, idx: usize, name: &str) -> Option<TokenStream> {
        let base = &self.p.base_type;
        let transform = self.p.transform.as_ref();
        let wrappers = &self.wrappers;
        super::primitive::gen_bare_view_string_direct_columnar_finish(
            base, transform, wrappers, idx, name,
        )
        .or_else(|| {
            super::primitive::gen_option_string_direct_columnar_finish(
                base, transform, wrappers, idx, name,
            )
        })
        .or_else(|| {
            super::primitive::gen_numeric_direct_columnar_finish(
                base, transform, wrappers, idx, name,
            )
        })
        .or_else(|| {
            super::primitive::gen_option_bool_direct_columnar_finish(
                base, transform, wrappers, idx, name,
            )
        })
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

    fn gen_schema_entries(&self) -> TokenStream {
        let name = &self.field_name;
        super::nested::generate_schema_entries_for_struct(
            &self.n.type_path,
            name,
            has_vec(&self.wrappers),
        )
    }

    fn gen_empty_series_creation(&self) -> TokenStream {
        let name = &self.field_name;
        super::nested::nested_empty_series_row(&self.n.type_path, name, &self.wrappers)
    }

    fn gen_populator_inits(&self, idx: usize) -> Vec<TokenStream> {
        super::nested::nested_decls(&self.wrappers, &self.n.type_path, idx)
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
        super::nested::generate_nested_for_columnar_push(
            &self.n.type_path,
            &access,
            &self.wrappers,
            idx,
            self.n.is_generic,
        )
    }

    fn gen_for_anyvalue(&self, it_ident: &Ident, values_vec_ident: &Ident) -> TokenStream {
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
        super::nested::generate_nested_for_anyvalue(
            &self.n.type_path,
            values_vec_ident,
            &access,
            &self.wrappers,
            self.n.is_generic,
        )
    }

    fn gen_vec_values_finishers(&self, idx: usize) -> TokenStream {
        super::nested::nested_finishers_for_vec_anyvalues(&self.wrappers, idx)
    }

    fn gen_columnar_builders(&self, idx: usize) -> Vec<TokenStream> {
        super::nested::nested_columnar_builders(&self.wrappers, idx, &self.field_name)
    }

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
    /// for the depth-1, depth-2, and depth-3 wrapper shapes the bulk helpers
    /// support: bare leaf, `Option<T>`, `Vec<T>`, `Option<Vec<T>>`,
    /// `Vec<Option<T>>`, and `Option<Vec<Option<T>>>`.
    ///
    /// All bulk emitters route through the `Columnar::columnar_from_refs`
    /// trait method, which works uniformly for both generic-parameter and
    /// concrete-struct base types — no per-element clone required.
    ///
    /// Remaining nestings (`Vec<Vec<T>>`, `Option<Option<T>>`, etc.) fall
    /// through to the per-row pipeline; those paths already drive
    /// `Vec<AnyValue>` aggregation through
    /// `__df_derive_vec_to_inner_list_values`, so the added bulk machinery
    /// isn't worth the complexity.
    fn try_bulk_emit(
        &self,
        pa_root: &TokenStream,
        idx: usize,
        ctx: super::bulk::BulkContext<'_>,
    ) -> Option<Vec<TokenStream>> {
        let ty = &self.n.type_path;
        let columnar_trait = &self.n.columnar_trait;
        let to_df_trait = &self.n.to_df_trait;
        let access = self.it_access();
        let emit = match self.wrappers.as_slice() {
            [] => super::bulk::gen_bulk_leaf(ty, columnar_trait, to_df_trait, idx, &access, ctx),
            [Wrapper::Option] => {
                super::bulk::gen_bulk_option(ty, columnar_trait, to_df_trait, idx, &access, ctx)
            }
            [Wrapper::Vec] => super::bulk::gen_bulk_vec(
                pa_root,
                ty,
                columnar_trait,
                to_df_trait,
                idx,
                &access,
                ctx,
            ),
            [Wrapper::Option, Wrapper::Vec] => super::bulk::gen_bulk_option_vec(
                pa_root,
                ty,
                columnar_trait,
                to_df_trait,
                idx,
                &access,
                ctx,
            ),
            [Wrapper::Vec, Wrapper::Option] => super::bulk::gen_bulk_vec_option(
                pa_root,
                ty,
                columnar_trait,
                to_df_trait,
                idx,
                &access,
                ctx,
            ),
            [Wrapper::Option, Wrapper::Vec, Wrapper::Option] => {
                super::bulk::gen_bulk_option_vec_option(
                    pa_root,
                    ty,
                    columnar_trait,
                    to_df_trait,
                    idx,
                    &access,
                    ctx,
                )
            }
            _ => return None,
        };
        Some(vec![emit])
    }

    fn gen_bulk_columnar_emit(
        &self,
        pa_root: &TokenStream,
        idx: usize,
    ) -> Option<Vec<TokenStream>> {
        let parent_name = self.field_name.as_str();
        self.try_bulk_emit(
            pa_root,
            idx,
            super::bulk::BulkContext::Columnar { parent_name },
        )
    }

    fn gen_bulk_vec_anyvalues_emit(
        &self,
        pa_root: &TokenStream,
        idx: usize,
    ) -> Option<Vec<TokenStream>> {
        self.try_bulk_emit(pa_root, idx, super::bulk::BulkContext::VecAnyvalues)
    }
}

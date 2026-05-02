use crate::ir::{BaseType, PrimitiveTransform, StructIR, Wrapper, has_vec};
use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use super::encoder::{self, Encoder, LeafCtx};
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

/// Sealed internal trait the per-field codegen routines call against. Both
/// `PrimitiveStrategy` and `NestedStructStrategy` implement it. The public
/// `Strategy` alias is `Box<dyn StrategyVariant>`, so callers see a uniform
/// trait-object surface and `Box`'s `Deref` impl turns each method call
/// site into a single virtual dispatch — no per-method `match` ladder.
pub trait StrategyVariant {
    fn gen_schema_entries(&self) -> TokenStream;
    fn gen_empty_series_creation(&self) -> TokenStream;
    fn gen_populator_inits(&self, idx: usize) -> Vec<TokenStream>;
    fn gen_populator_push(&self, it_ident: &Ident, idx: usize) -> TokenStream;
    fn gen_vec_values_finishers(&self, idx: usize) -> TokenStream;
    fn gen_columnar_builders(&self, idx: usize) -> Vec<TokenStream>;
    /// Per-field codegen that pushes one `AnyValue` per inner schema column
    /// onto `values_vec_ident` for a single instance bound to `it_ident`.
    /// Used by the `to_inner_values(&self)` trait override to bypass the
    /// `to_dataframe()` round-trip: the outer struct's override invokes
    /// this per field, and the nested `on_leaf` recursively calls the
    /// inner type's `to_inner_values()` instead of constructing a one-row
    /// `DataFrame`.
    fn gen_for_anyvalue(&self, it_ident: &Ident, values_vec_ident: &Ident) -> TokenStream;
    /// Returns `Some` when this field can bypass the per-row
    /// decls/push/builders triple in the columnar path and emit a single
    /// bulk builder. `pa_root` is the cached `polars-arrow` crate-root
    /// token stream threaded down from the per-derive entry point so each
    /// call doesn't re-run `proc_macro_crate::crate_name`.
    fn gen_bulk_columnar_emit(&self, pa_root: &TokenStream, idx: usize)
    -> Option<Vec<TokenStream>>;
    /// Mirror of `gen_bulk_columnar_emit` for the
    /// `__df_derive_vec_to_inner_list_values` helper path: when a strategy
    /// can produce its `AnyValue::List` entries in bulk (without a per-row
    /// push), returns `Some`.
    fn gen_bulk_vec_anyvalues_emit(
        &self,
        pa_root: &TokenStream,
        idx: usize,
    ) -> Option<Vec<TokenStream>>;
}

pub type Strategy = Box<dyn StrategyVariant>;

pub fn build_strategies(ir: &StructIR, config: &super::MacroConfig) -> Vec<Strategy> {
    let columnar_trait = &config.columnar_trait_path;
    let to_df_trait = &config.to_dataframe_trait_path;
    let decimal_encode = &config.decimal128_encode_trait_path;
    let stringy = |t: &Option<PrimitiveTransform>| {
        matches!(
            t,
            Some(PrimitiveTransform::ToString | PrimitiveTransform::AsStr)
        )
    };
    let primitive = |f: &crate::ir::FieldIR| -> Strategy {
        Box::new(PrimitiveStrategy::new(
            f.name.clone(),
            f.field_index,
            f.wrappers.clone(),
            PrimitiveIR {
                base_type: f.base_type.clone(),
                transform: f.transform.clone(),
                decimal128_encode_trait: decimal_encode.clone(),
            },
        ))
    };
    let nested = |f: &crate::ir::FieldIR, type_path: TokenStream, is_generic: bool| -> Strategy {
        Box::new(NestedStructStrategy::new(
            f.name.clone(),
            f.field_index,
            f.wrappers.clone(),
            NestedIR {
                type_path,
                columnar_trait: columnar_trait.clone(),
                to_df_trait: to_df_trait.clone(),
                is_generic,
            },
        ))
    };
    ir.fields
        .iter()
        .map(|f| match &f.base_type {
            BaseType::Struct(id, args) if !stringy(&f.transform) => {
                nested(f, build_type_path(id, args.as_ref()), false)
            }
            BaseType::Generic(id) if !stringy(&f.transform) => nested(f, quote! { #id }, true),
            _ => primitive(f),
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

    /// Build the per-field encoder when this shape (`[]` or `[Option]` over a
    /// supported primitive leaf) is served by the encoder IR. Returns `None`
    /// for `Vec<...>` shapes (Step 2) and for the few leaf carve-outs the IR
    /// doesn't yet cover (notably bare `ISize`/`USize`).
    ///
    /// `access` is only meaningful for the encoder's `push` field; the `decls`
    /// and `finish_series` fields are independent of it (modulo `name`, which
    /// is baked into `finish_series`). Callers that only consume
    /// `decls`/`finish_series` may pass any well-formed expression — the
    /// `gen_populator_inits`, `gen_columnar_builders`, and
    /// `gen_vec_values_finishers` paths do this with `quote! {}`.
    ///
    /// `name` is the column name baked into the produced Series. Pass the
    /// field name for the columnar context; pass `""` for the vec-anyvalues
    /// context (matching the prior `primitive_finishers_for_vec_anyvalues`
    /// behavior — the rename happens implicitly through `AnyValue::List`).
    fn try_build_encoder(&self, access: &TokenStream, idx: usize, name: &str) -> Option<Encoder> {
        let ctx = LeafCtx {
            access,
            idx,
            name,
            decimal128_encode_trait: &self.p.decimal128_encode_trait,
        };
        encoder::try_build_encoder(
            &self.p.base_type,
            self.p.transform.as_ref(),
            &self.wrappers,
            &ctx,
        )
    }
}

impl StrategyVariant for PrimitiveStrategy {
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
        // Encoder IR covers `[]`, `[Option]`, and the `[Vec, ...]` shapes
        // added in Step 2. For the `[Vec, ...]` shapes the bulk emit lives
        // in `gen_populator_inits` (consumed by the populator decls block);
        // the finisher just wraps the encoder's `finish_series` in
        // `AnyValue::List(...)`. For `[]`/`[Option]` shapes the finisher
        // does the same with the leaf-buffer Series. The access expression
        // is required when the encoder builds its bulk decls; threading
        // `it_access()` keeps the precount loops over the real field
        // even though this method is the finisher (the access doesn't
        // appear in `finish_series` itself; it only matters because
        // `try_build_encoder` is shape-conditional).
        let access = self.it_access();
        if let Some(enc) = self.try_build_encoder(&access, idx, "") {
            let series = enc.finish_series;
            let pp = super::polars_paths::prelude();
            return quote! {
                let inner = { #series };
                out_values.push(#pp::AnyValue::List(inner));
            };
        }
        super::primitive::primitive_finishers_for_vec_anyvalues(
            &self.wrappers,
            &self.p.base_type,
            self.p.transform.as_ref(),
            idx,
        )
    }

    fn gen_populator_inits(&self, idx: usize) -> Vec<TokenStream> {
        // Encoder IR covers `[]`, `[Option]`, and the `[Vec, ...]` shapes
        // added in Step 2. For the `[Vec, ...]` shapes the precount loops
        // and the bulk per-leaf push live inside the encoder's `decls`,
        // so the access must be the real field expression — pass
        // `it_access()` even though the leaf-shape encoders ignore it for
        // `decls`. Falls through to the legacy `primitive_decls` for the
        // few shapes the encoder doesn't yet cover (notably bare
        // `ISize`/`USize`).
        let access = self.it_access();
        if let Some(enc) = self.try_build_encoder(&access, idx, &self.field_name) {
            return enc.decls;
        }
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
        if let Some(enc) = self.try_build_encoder(&access, idx, &self.field_name) {
            return enc.push;
        }
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

    fn gen_bulk_columnar_emit(
        &self,
        _pa_root: &TokenStream,
        idx: usize,
    ) -> Option<Vec<TokenStream>> {
        // For `[Vec, ...]` shapes the encoder packs the entire emit (precount,
        // buffers, fill loop, leaf array, list array stacking) into one block.
        // Routing that block through the bulk-emit channel — instead of the
        // decls/push/builders triple — keeps it AFTER the per-row iteration
        // over `items`. That ordering matches the legacy direct-fast-path
        // emitters and preserves their cache locality (the per-row buffers
        // for non-vec fields finish populating before the vec fields touch
        // memory). See bench `vec_vec_i32` / `vec_opt_datetime`: emitting in
        // decls regresses ~4% relative to placing the same work post-loop.
        if !has_vec(&self.wrappers) {
            return None;
        }
        let access = self.it_access();
        let enc = self.try_build_encoder(&access, idx, &self.field_name)?;
        let pp = super::polars_paths::prelude();
        let decls = enc.decls;
        let series = enc.finish_series;
        Some(vec![quote! {
            #(#decls)*
            {
                let s: #pp::Series = { #series };
                columns.push(s.into());
            }
        }])
    }

    fn gen_bulk_vec_anyvalues_emit(
        &self,
        _pa_root: &TokenStream,
        idx: usize,
    ) -> Option<Vec<TokenStream>> {
        // Mirror of `gen_bulk_columnar_emit` for the vec-anyvalues helper:
        // wrap the encoder's `finish_series` in `AnyValue::List(...)` and
        // emit the whole block as a bulk operation. Same ordering rationale
        // as the columnar path.
        if !has_vec(&self.wrappers) {
            return None;
        }
        let access = self.it_access();
        let enc = self.try_build_encoder(&access, idx, "")?;
        let pp = super::polars_paths::prelude();
        let decls = enc.decls;
        let series = enc.finish_series;
        Some(vec![quote! {
            #(#decls)*
            {
                let inner: #pp::Series = { #series };
                out_values.push(#pp::AnyValue::List(inner));
            }
        }])
    }

    fn gen_columnar_builders(&self, idx: usize) -> Vec<TokenStream> {
        let name = &self.field_name;
        let pp = super::polars_paths::prelude();
        // Encoder IR covers `[]`, `[Option]`, `[Vec]`, `[Vec, Option]`,
        // `[Vec, Vec]`, and `[Vec, Vec, Option]` for primitive shapes.
        // For these the encoder already has the right dtype baked in; we
        // just splice the finish expression into a `columns.push` block.
        // The access is irrelevant to the finisher itself, but consistency
        // with `gen_populator_inits` (which threads the real access
        // through) keeps `try_build_encoder` shape-checked under the same
        // arguments.
        let access = self.it_access();
        if let Some(enc) = self.try_build_encoder(&access, idx, name) {
            let series = enc.finish_series;
            return vec![quote! {{
                let s = { #series };
                columns.push(s.into());
            }}];
        }
        if has_vec(&self.wrappers) {
            // Legacy path for `[Option, Vec]`, `[Option, Vec, Option]`,
            // `[Vec, Option, Vec]`, deeper nestings, and any vec-bearing
            // shape over `ISize`/`USize`. The list builder was constructed
            // in `primitive_decls` with the correct inner dtype, so the
            // finished Series already has the schema dtype. Typed
            // `ListPrimitiveChunkedBuilder<Native>` finishers call
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
        // Non-Vec primitive (currently only ISize/USize bare and Option):
        // the buffer holds the raw element type chosen by `compute_mapping`
        // (e.g. `Vec<i64>` for `ISize`). The remaining cases need no cast
        // (transform is `None` so `needs_cast` is false) — but we keep the
        // gated `s.cast` to match the prior generic path's emission exactly.
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
    /// `Vec<Option<T>>`, `Option<Vec<Option<T>>>`, and `Vec<Vec<T>>`.
    ///
    /// All bulk emitters route through the `Columnar::columnar_from_refs`
    /// trait method, which works uniformly for both generic-parameter and
    /// concrete-struct base types — no per-element clone required.
    ///
    /// Remaining nestings (`Option<Option<T>>`, deeper triple-Vec shapes,
    /// etc.) fall through to the per-row pipeline; those paths already drive
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
            [Wrapper::Vec, Wrapper::Vec] => super::bulk::gen_bulk_vec_vec(
                pa_root,
                ty,
                columnar_trait,
                to_df_trait,
                idx,
                &access,
                ctx,
            ),
            _ => return None,
        };
        Some(vec![emit])
    }
}

impl StrategyVariant for NestedStructStrategy {
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

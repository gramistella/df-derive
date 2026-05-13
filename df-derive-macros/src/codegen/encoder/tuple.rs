//! Tuple-field encoder paths.
//!
//! A tuple-typed field flattens to one or more columns at the parent struct's
//! schema — one per primitive leaf, or one per inner schema column for nested
//! struct elements. Each element's columns are prefixed `<field>.field_<i>`,
//! mirroring the dot-notation convention nested-struct fields use. The parent
//! field's outer wrapper stack (`Option`/`Vec` layers) distributes across
//! every element column: `Vec<(A, B)>` produces parallel `List<A>` and
//! `List<B>` columns; `Option<(A, B)>` produces parallel `Option<A>` and
//! `Option<B>` columns with row validity OR'd into each.
//!
//! Per-element emission is recursive: the parent's wrappers compose with each
//! element's own wrappers, plus a per-element projection (`v.<i>`) at the
//! boundary. The composition obeys the same rules as nested structs (parent
//! Option distributes as outer-list validity on each element column's
//! outermost layer; consecutive Options collapse into one validity bit per
//! Polars's representation).

use crate::codegen::MacroConfig;
use crate::codegen::strategy::{EmitMode, FieldEmit};
use crate::ir::{
    AccessChain, AccessStep, FieldIR, LeafSpec, TupleElement, VecLayerSpec, VecLayers, WrapperShape,
};
use proc_macro2::TokenStream;
use quote::quote;

use super::idents;
use super::shape_walk::{
    LayerIdents, LayerProjection, LayerWrap, OwnPolicy, ShapePrecount, ShapeScan,
    freeze_offsets_buf, freeze_validity_bitmap, shape_assemble_list_stack, shape_offsets_decls,
    shape_validity_decls,
};
use super::{
    BaseCtx, Encoder, LeafCtx, NestedLeafCtx, build_encoder, build_nested_encoder, struct_type_path,
};

// ============================================================================
// Schema / empty-rows
// ============================================================================

/// Build the schema/empty-rows entries for a tuple-typed field. Each element
/// contributes one or more entries; nested tuples recurse, and nested
/// structs/generic parameters delegate to the existing schema helpers (which
/// iterate `T::schema()?` at runtime).
pub fn build_field_entries(
    field: &FieldIR,
    elements: &[TupleElement],
    mode: EmitMode,
    config: &MacroConfig,
) -> TokenStream {
    let parent_name = field.name.to_string();
    let outer_layers = field.wrapper_shape.vec_depth();
    build_tuple_entries(elements, &parent_name, outer_layers, mode, config)
}

fn build_tuple_entries(
    elements: &[TupleElement],
    column_prefix: &str,
    outer_layers: usize,
    mode: EmitMode,
    config: &MacroConfig,
) -> TokenStream {
    let pp = crate::codegen::external_paths::prelude();
    let mut per_elem: Vec<TokenStream> = Vec::with_capacity(elements.len());
    for (i, elem) in elements.iter().enumerate() {
        let elem_prefix = format!("{column_prefix}.field_{i}");
        per_elem.push(build_element_entries(
            elem,
            &elem_prefix,
            outer_layers,
            mode,
            config,
        ));
    }
    match mode {
        EmitMode::SchemaEntries => quote! {
            {
                let mut tuple_fields: ::std::vec::Vec<(::std::string::String, #pp::DataType)> =
                    ::std::vec::Vec::new();
                #(
                    tuple_fields.extend(#per_elem);
                )*
                tuple_fields
            }
        },
        EmitMode::EmptyRows => quote! {
            {
                let mut tuple_series: ::std::vec::Vec<#pp::Column> = ::std::vec::Vec::new();
                #(
                    tuple_series.extend(#per_elem);
                )*
                tuple_series
            }
        },
    }
}

fn build_element_entries(
    elem: &TupleElement,
    column_prefix: &str,
    outer_layers: usize,
    mode: EmitMode,
    config: &MacroConfig,
) -> TokenStream {
    let pp = crate::codegen::external_paths::prelude();
    let total_layers = outer_layers + elem.wrapper_shape.vec_depth();
    match &elem.leaf_spec {
        LeafSpec::Struct(path) => {
            let type_path = struct_type_path(path);
            element_nested_entries(&type_path, column_prefix, total_layers, mode, config)
        }
        LeafSpec::Generic(id) => {
            let type_path = quote! { #id };
            element_nested_entries(&type_path, column_prefix, total_layers, mode, config)
        }
        LeafSpec::Tuple(inner) => {
            build_tuple_entries(inner, column_prefix, total_layers, mode, config)
        }
        _ => {
            let elem_dtype = elem.leaf_spec.dtype();
            let full_dtype = crate::codegen::external_paths::wrap_list_layers_compile_time_pub(
                &pp,
                elem_dtype,
                total_layers,
            );
            match mode {
                EmitMode::SchemaEntries => quote! {
                    ::std::vec![(::std::string::String::from(#column_prefix), #full_dtype)]
                },
                EmitMode::EmptyRows => quote! {
                    ::std::vec![
                        #pp::Series::new_empty(#column_prefix.into(), &#full_dtype).into()
                    ]
                },
            }
        }
    }
}

fn element_nested_entries(
    type_path: &TokenStream,
    column_prefix: &str,
    total_layers: usize,
    mode: EmitMode,
    config: &MacroConfig,
) -> TokenStream {
    match mode {
        EmitMode::SchemaEntries => crate::codegen::nested::generate_schema_entries_for_struct(
            type_path,
            &config.to_dataframe_trait_path,
            column_prefix,
            total_layers,
        ),
        EmitMode::EmptyRows => crate::codegen::nested::nested_empty_series_row(
            type_path,
            &config.to_dataframe_trait_path,
            column_prefix,
            total_layers,
        ),
    }
}

// ============================================================================
// Columnar emission
// ============================================================================

/// Build the columnar emit pieces for a tuple-typed field. Decls and pushes
/// are empty; every element contributes a self-contained builder block (sized
/// to run after the call site's per-row loop, mirroring the nested-struct
/// path).
pub fn build_field_emit(
    field: &FieldIR,
    config: &MacroConfig,
    field_idx: usize,
    elements: &[TupleElement],
) -> FieldEmit {
    let inner_it = idents::populator_iter();
    let parent_access = field_access(field, &inner_it);
    let parent_name = field.name.to_string();

    let mut builders: Vec<TokenStream> = Vec::with_capacity(elements.len());
    for (elem_idx, elem) in elements.iter().enumerate() {
        let elem_prefix = format!("{parent_name}.field_{elem_idx}");
        builders.push(emit_element(
            &parent_access,
            &field.wrapper_shape,
            elem,
            elem_idx,
            field_idx,
            &elem_prefix,
            config,
        ));
    }

    FieldEmit {
        decls: Vec::new(),
        push: TokenStream::new(),
        builders,
    }
}

/// `<it>.<field>` rooted at the populator iter, with smart-pointer derefs
/// applied. Mirrors `strategy::it_access`.
fn field_access(field: &FieldIR, it_ident: &syn::Ident) -> TokenStream {
    let raw = field.field_index.map_or_else(
        || {
            let id = &field.name;
            quote! { #it_ident.#id }
        },
        |i| {
            let li = syn::Index::from(i);
            quote! { #it_ident.#li }
        },
    );
    let mut out = raw;
    for _ in 0..field.outer_smart_ptr_depth {
        out = quote! { (*(#out)) };
    }
    out
}

/// Emit one element column. Composes the parent's wrappers with the element's
/// own wrappers; the boundary projection (`<elem_idx>`) is applied at the
/// deepest parent binding.
#[allow(clippy::too_many_arguments)]
fn emit_element(
    parent_access: &TokenStream,
    parent_wrapper: &WrapperShape,
    elem: &TupleElement,
    elem_idx: usize,
    field_idx: usize,
    column_prefix: &str,
    config: &MacroConfig,
) -> TokenStream {
    // Recursion: tuple elements may themselves be tuples or nested structs.
    if let LeafSpec::Tuple(inner) = &elem.leaf_spec {
        return emit_inner_tuple(
            parent_access,
            parent_wrapper,
            elem,
            elem_idx,
            inner,
            field_idx,
            column_prefix,
            config,
        );
    }

    match parent_wrapper {
        // Bare `(A, B)` (or `Box<(A, B)>` etc., with smart pointers already
        // dereffed in `parent_access`). Project directly via `.<i>` and
        // route through the standard encoder with the element's own
        // wrapper shape and leaf.
        WrapperShape::Leaf {
            option_layers: 0, ..
        } => {
            let elem_li = syn::Index::from(elem_idx);
            let mut access = quote! { #parent_access.#elem_li };
            for _ in 0..elem.outer_smart_ptr_depth {
                access = quote! { (*(#access)) };
            }
            emit_via_standard_encoder(
                &access,
                &elem.wrapper_shape,
                &elem.leaf_spec,
                field_idx,
                column_prefix,
                config,
            )
        }
        // `Option<(A, B)>` / `Option<...<Option<(A, B)>>>`. Collapse parent
        // Options to `Option<&Tuple>` per row, project to `Option<&Inner>`
        // (or `Option<Inner>` for Copy primitive leaves) via `.map(...)`,
        // and route through the standard encoder with the composed wrapper.
        WrapperShape::Leaf { access, .. } => {
            let elem_li = syn::Index::from(elem_idx);
            let collapsed_parent = super::access_chain_to_option_ref(parent_access, access);
            // The standard primitive Option-leaf machinery expects an
            // access expression that yields `Option<T>` for Copy primitives
            // (numeric / Bool / NaiveDate / NaiveTime / Duration). Project
            // by value only when the whole element wrapper stack is also
            // Copy-projectable from `&Tuple`: bare leaves and Option-only
            // stacks over Copy leaves. Vecs and smart pointers must project
            // by reference so we never move out of a borrowed tuple.
            let proj_param = idents::tuple_proj_param();
            let projected = if is_copy_element_projection(elem) {
                quote! {
                    ((#collapsed_parent).map(|#proj_param| #proj_param.#elem_li))
                }
            } else {
                quote! {
                    ((#collapsed_parent).map(|#proj_param| &#proj_param.#elem_li))
                }
            };
            let composed_shape = compose_option_with_element(&elem.wrapper_shape);
            emit_via_standard_encoder(
                &projected,
                &composed_shape,
                &elem.leaf_spec,
                field_idx,
                column_prefix,
                config,
            )
        }
        // Parent has at least one Vec layer. Construct a composed emission
        // that walks the composed wrapper stack (parent + element layers)
        // with `.iter()` rebased to project at the boundary.
        WrapperShape::Vec(parent_layers) => emit_vec_parent(
            parent_access,
            parent_layers,
            elem,
            elem_idx,
            field_idx,
            column_prefix,
            config,
        ),
    }
}

fn is_copy_element_projection(elem: &TupleElement) -> bool {
    is_copy_leaf_for_projection(&elem.leaf_spec)
        && match &elem.wrapper_shape {
            WrapperShape::Leaf { access, .. } => access.is_only_options(),
            WrapperShape::Vec(_) => false,
        }
}

/// True when the element's leaf is `Copy` AND a primitive that the
/// standard Option-leaf encoder expects to receive by value (not by
/// reference). The numeric / Bool / `NaiveDate` / `NaiveTime` / Duration
/// leaves all match this; String / Decimal / `DateTime` / Binary do not
/// (their Option-arm push bodies do `match &access` and bind by ref).
const fn is_copy_leaf_for_projection(leaf: &LeafSpec) -> bool {
    matches!(
        leaf,
        LeafSpec::Numeric(_)
            | LeafSpec::Bool
            | LeafSpec::NaiveDate
            | LeafSpec::NaiveTime
            | LeafSpec::Duration { .. }
    )
}

/// Compose the parent's collapsed Option (1 outer Option) with the element's
/// own wrapper shape. For a leaf-only element, stack one extra Option layer.
/// For a Vec-bearing element, attach the parent's Option as outer-list
/// validity on the element's outermost Vec layer.
fn compose_option_with_element(elem_shape: &WrapperShape) -> WrapperShape {
    match elem_shape {
        WrapperShape::Leaf {
            option_layers,
            access,
        } => WrapperShape::Leaf {
            option_layers: 1 + option_layers,
            access: prepend_option_access(access),
        },
        WrapperShape::Vec(layers) => {
            let mut new_layers = layers.layers.clone();
            new_layers[0].option_layers_above += 1;
            new_layers[0].access = prepend_option_access(&new_layers[0].access);
            WrapperShape::Vec(VecLayers {
                layers: new_layers,
                inner_option_layers: layers.inner_option_layers,
                inner_access: layers.inner_access.clone(),
            })
        }
    }
}

fn prepend_option_access(access: &AccessChain) -> AccessChain {
    let mut steps = Vec::with_capacity(access.steps.len() + 1);
    steps.push(AccessStep::Option);
    steps.extend(access.steps.iter().copied());
    AccessChain { steps }
}

fn concat_access_chains(left: &AccessChain, right: &AccessChain) -> AccessChain {
    let mut steps = Vec::with_capacity(left.steps.len() + right.steps.len());
    steps.extend(left.steps.iter().copied());
    steps.extend(right.steps.iter().copied());
    AccessChain { steps }
}

fn prepend_parent_option_access(parent_access: &AccessChain, access: &AccessChain) -> AccessChain {
    if parent_access.option_layers() > 0 {
        prepend_option_access(access)
    } else {
        access.clone()
    }
}

/// Emit a tuple element column with a Vec-bearing parent. Composes parent +
/// element wrappers, with the projection injected at the parent/element
/// boundary. Uses the shared shape walker with tuple-specific projection
/// layer.
#[allow(clippy::too_many_arguments)]
fn emit_vec_parent(
    parent_access: &TokenStream,
    parent_layers: &VecLayers,
    elem: &TupleElement,
    elem_idx: usize,
    field_idx: usize,
    column_prefix: &str,
    config: &MacroConfig,
) -> TokenStream {
    // Compose layers: parent + element. The element's outermost
    // `option_layers_above` (when it has its own Vec) attaches to the
    // boundary — semantically, parent's `inner_option_layers` (Options
    // immediately above the leaf, which is the projected tuple) become
    // outer-Option above the element's first Vec layer. Polars folds
    // consecutive Options into one bit, so the carry is additive.
    let mut composed_layers: Vec<VecLayerSpec> = parent_layers.layers.clone();
    let carried_inner_option = parent_layers.inner_option_layers;

    let composed_inner_option = match &elem.wrapper_shape {
        WrapperShape::Vec(elem_layers) => {
            let mut e_layers = elem_layers.layers.clone();
            // Parent's inner-Option (above-tuple Options) attaches to
            // element's outermost Vec as outer validity.
            e_layers[0].option_layers_above += carried_inner_option;
            e_layers[0].access =
                prepend_parent_option_access(&parent_layers.inner_access, &e_layers[0].access);
            composed_layers.extend(e_layers);
            elem_layers.inner_option_layers
        }
        WrapperShape::Leaf { option_layers, .. } => {
            // No element Vec layers. Parent's inner-Option and element's
            // option_layers both attach to the leaf — Polars folds them.
            carried_inner_option + option_layers
        }
    };
    let composed_inner_access = match &elem.wrapper_shape {
        WrapperShape::Vec(elem_layers) => elem_layers.inner_access.clone(),
        WrapperShape::Leaf { access, .. } => {
            concat_access_chains(&parent_layers.inner_access, access)
        }
    };

    let composed_shape = VecLayers {
        layers: composed_layers,
        inner_option_layers: composed_inner_option,
        inner_access: composed_inner_access,
    };
    let projection_layer = parent_layers.depth();
    let elem_li = syn::Index::from(elem_idx);
    let projection_path = quote! { .#elem_li };
    let projection = TupleProjection {
        layer: projection_layer,
        path: &projection_path,
        parent_access: &parent_layers.inner_access,
        smart_ptr_depth: elem.outer_smart_ptr_depth,
    };

    match &elem.leaf_spec {
        LeafSpec::Struct(path) => {
            let type_path = struct_type_path(path);
            emit_vec_parent_nested(
                parent_access,
                &composed_shape,
                projection,
                &type_path,
                field_idx,
                column_prefix,
                config,
            )
        }
        LeafSpec::Generic(id) => {
            let type_path = quote! { #id };
            emit_vec_parent_nested(
                parent_access,
                &composed_shape,
                projection,
                &type_path,
                field_idx,
                column_prefix,
                config,
            )
        }
        _ => emit_vec_parent_primitive(
            parent_access,
            &composed_shape,
            projection,
            elem,
            field_idx,
            column_prefix,
            config,
        ),
    }
}

/// Emit a tuple element column for a primitive element under a Vec-bearing
/// parent. Builds the per-element-push pipeline: precount, leaf storage,
/// per-layer offsets/validity, scan with projection at the boundary, and
/// post-scan stack of `LargeListArray::new` calls.
#[allow(clippy::too_many_arguments)]
fn emit_vec_parent_primitive(
    parent_access: &TokenStream,
    composed_shape: &VecLayers,
    projection: TupleProjection<'_>,
    elem: &TupleElement,
    field_idx: usize,
    column_prefix: &str,
    config: &MacroConfig,
) -> TokenStream {
    let pp = crate::codegen::external_paths::prelude();
    let pa_root = crate::codegen::external_paths::polars_arrow_root();
    let series_local = idents::vec_field_series(field_idx);
    let named = idents::field_named_series();
    let leaf_arr = idents::leaf_arr();
    let total_leaves = idents::total_leaves();

    let layers: Vec<LayerIdents> = (0..composed_shape.depth())
        .map(|i| tuple_layer_idents(field_idx, i))
        .collect();
    let layer_counters = tuple_layer_counters(field_idx, composed_shape.depth());

    // Build leaf pieces (storage decls, per-element push, leaf-array
    // construction, post-push offsets value) for the element's primitive
    // leaf.
    let dummy_access = TokenStream::new();
    let leaf_ctx = LeafCtx {
        base: BaseCtx {
            access: &dummy_access,
            idx: field_idx,
            name: column_prefix,
        },
        decimal128_encode_trait: &config.decimal128_encode_trait_path,
    };
    // Projection placement: when the projection lands AT the deepest
    // composed layer (i.e., the element has no own wrappers), the leaf
    // binding `__df_derive_v: &Tuple` needs in-leaf projection. When the
    // projection is at an interior layer (the element has its own
    // wrappers), the iteration source at that layer projects, and the
    // leaf binding is `&Leaf` directly — no in-leaf projection needed.
    let leaf_value_override = build_leaf_projection_value_expr(
        &elem.leaf_spec,
        projection.path,
        composed_shape.depth(),
        projection.layer,
        elem.outer_smart_ptr_depth,
    );

    let leaf_pieces = build_primitive_leaf_pieces(
        &leaf_ctx,
        &elem.leaf_spec,
        composed_shape.has_inner_option(),
        &pa_root,
        leaf_value_override.as_ref(),
    );

    let precount = build_precount(
        composed_shape,
        &layers,
        &layer_counters,
        &total_leaves,
        parent_access,
        projection,
    );
    let scan = build_scan(
        composed_shape,
        &layers,
        parent_access,
        projection,
        &leaf_pieces.per_elem_push,
        &leaf_pieces.leaf_offsets_post_push,
    );
    let offsets_decls = build_offsets_decls(&layers, &layer_counters);
    let validity_decls = build_validity_decls(composed_shape, &layers, &layer_counters, &pa_root);

    let materialize = build_materialize(
        composed_shape,
        &layers,
        &leaf_arr,
        &elem.leaf_spec.dtype(),
        &leaf_pieces.leaf_arr_expr,
        &pp,
        &pa_root,
    );

    let extra_imports = leaf_pieces.extra_imports;
    let storage_decls = leaf_pieces.storage_decls;

    quote! {
        {
            let #series_local: #pp::Series = {
                #extra_imports
                #precount
                #storage_decls
                #offsets_decls
                #validity_decls
                #scan
                #materialize
            };
            let #named = #series_local.with_name(#column_prefix.into());
            columns.push(#named.into());
        }
    }
}

/// Emit a tuple element column for a nested-struct element under a
/// Vec-bearing parent. Walks the composed shape, collects `&Inner` refs at
/// the deepest binding (with projection applied), then dispatches the
/// inner type's `columnar_from_refs` and stacks `LargeListArray`s.
// Bench-sensitive generated-code builder: this intentionally keeps the
// composed-shape walk, nested dispatch, and list stacking adjacent so tuple
// Vec emission stays predictable while deeper factoring waits.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn emit_vec_parent_nested(
    parent_access: &TokenStream,
    composed_shape: &VecLayers,
    projection: TupleProjection<'_>,
    type_path: &TokenStream,
    field_idx: usize,
    column_prefix: &str,
    config: &MacroConfig,
) -> TokenStream {
    let pp = crate::codegen::external_paths::prelude();
    let pa_root = crate::codegen::external_paths::polars_arrow_root();
    let columnar_trait = &config.columnar_trait_path;
    let to_df_trait = &config.to_dataframe_trait_path;
    let total_leaves = idents::nested_total(field_idx);
    let flat = idents::nested_flat(field_idx);
    let positions = idents::nested_positions(field_idx);
    let df = idents::nested_df(field_idx);
    let take = idents::nested_take(field_idx);
    let col_name = idents::nested_col_name();
    let dtype = idents::nested_col_dtype();
    let prefixed = idents::nested_prefixed_name();
    let inner_series = idents::nested_inner_series();
    let inner_full = idents::nested_inner_full();
    let inner_chunk = idents::nested_inner_chunk();
    let inner_col = idents::nested_inner_col();
    let inner_rech = idents::nested_inner_rech();
    let named = idents::field_named_series();

    let layers: Vec<LayerIdents> = (0..composed_shape.depth())
        .map(|i| tuple_layer_idents(field_idx, i))
        .collect();
    let layer_counters = tuple_layer_counters(field_idx, composed_shape.depth());

    let precount = build_precount(
        composed_shape,
        &layers,
        &layer_counters,
        &total_leaves,
        parent_access,
        projection,
    );

    let has_inner_option = composed_shape.has_inner_option();

    // Per-element push body: collect &Inner refs (and positions on
    // inner-Option). When the projection lands at the deepest layer
    // (the typical `Vec<(Inner, i32)>` shape), the leaf binding is
    // `&Tuple` and we project `&v.<i>` to get `&Inner`. When the
    // projection is at an interior layer (the element itself has Vec
    // wrappers), the iteration source already yielded the projected
    // refs, so the binding is `&Inner` directly.
    let leaf_v = idents::leaf_value();
    let inner_v = idents::tuple_nested_inner_v();
    let projected_leaf = if projection.layer == composed_shape.depth() {
        let projection_path = projection.path;
        let mut projected = quote! { (*#leaf_v) #projection_path };
        for _ in 0..projection.smart_ptr_depth {
            projected = quote! { (*(#projected)) };
        }
        quote! { &(#projected) }
    } else {
        quote! { #leaf_v }
    };
    let per_elem_push = if has_inner_option {
        quote! {
            match #projected_leaf {
                ::std::option::Option::Some(#inner_v) => {
                    #positions.push(::std::option::Option::Some(
                        #flat.len() as #pp::IdxSize,
                    ));
                    #flat.push(#inner_v);
                }
                ::std::option::Option::None => {
                    #positions.push(::std::option::Option::None);
                }
            }
        }
    } else {
        quote! {
            #flat.push(#projected_leaf);
        }
    };
    let leaf_offsets_post_push = if has_inner_option {
        quote! { #positions.len() }
    } else {
        quote! { #flat.len() }
    };

    let scan = build_scan(
        composed_shape,
        &layers,
        parent_access,
        projection,
        &per_elem_push,
        &leaf_offsets_post_push,
    );
    let offsets_decls = build_offsets_decls(&layers, &layer_counters);
    let validity_decls = build_validity_decls(composed_shape, &layers, &layer_counters, &pa_root);

    let positions_decl = if has_inner_option {
        quote! {
            let mut #positions: ::std::vec::Vec<::std::option::Option<#pp::IdxSize>> =
                ::std::vec::Vec::with_capacity(#total_leaves);
        }
    } else {
        TokenStream::new()
    };

    // Per-column inner-Series expressions for the four dispatch branches.
    let inner_col_direct = quote! {
        #df.column(#col_name)?.as_materialized_series().clone()
    };
    let inner_col_take = quote! {{
        let #inner_full = #df.column(#col_name)?.as_materialized_series();
        #inner_full.take(&#take)?
    }};
    let inner_col_empty = quote! { #pp::Series::new_empty("".into(), #dtype) };
    let inner_col_all_absent = quote! {
        #pp::Series::new_empty("".into(), #dtype)
            .extend_constant(#pp::AnyValue::Null, #total_leaves)?
    };

    let series_direct = wrap_per_column_layers(
        composed_shape,
        &layers,
        &inner_col_direct,
        &pp,
        &inner_chunk,
        &inner_col,
        &inner_rech,
        &dtype,
    );
    let series_take = wrap_per_column_layers(
        composed_shape,
        &layers,
        &inner_col_take,
        &pp,
        &inner_chunk,
        &inner_col,
        &inner_rech,
        &dtype,
    );
    let series_empty = wrap_per_column_layers(
        composed_shape,
        &layers,
        &inner_col_empty,
        &pp,
        &inner_chunk,
        &inner_col,
        &inner_rech,
        &dtype,
    );
    let series_all_absent = wrap_per_column_layers(
        composed_shape,
        &layers,
        &inner_col_all_absent,
        &pp,
        &inner_chunk,
        &inner_col,
        &inner_rech,
        &dtype,
    );

    let consume = |series_expr: &TokenStream| -> TokenStream {
        quote! {
            for (#col_name, #dtype) in <#type_path as #to_df_trait>::schema()? {
                let #col_name: &str = #col_name.as_str();
                let #dtype: &#pp::DataType = &#dtype;
                {
                    let #prefixed = ::std::format!("{}.{}", #column_prefix, #col_name);
                    let #inner_series: #pp::Series = #series_expr;
                    let #named = #inner_series.with_name(#prefixed.as_str().into());
                    columns.push(#named.into());
                }
            }
        }
    };

    let consume_direct = consume(&series_direct);
    let consume_take = consume(&series_take);
    let consume_empty = consume(&series_empty);
    let consume_all_absent = consume(&series_all_absent);

    let df_decl = quote! {
        let #df = <#type_path as #columnar_trait>::columnar_from_refs(&#flat)?;
    };
    let take_decl = quote! {
        let #take: #pp::IdxCa =
            <#pp::IdxCa as #pp::NewChunkedArray<_, _>>::from_iter_options(
                "".into(),
                #positions.iter().copied(),
            );
    };

    // Hoist offsets/validity freezes above the dispatch — collect-then-bulk
    // path reuses them per-column.
    let mut validity_freezes: Vec<TokenStream> = Vec::new();
    for (i, layer) in layers.iter().enumerate() {
        if !composed_shape.layers[i].has_outer_validity() {
            continue;
        }
        validity_freezes.push(freeze_validity_bitmap(
            &layer.validity_bm,
            &layer.validity_mb,
            &pa_root,
        ));
    }
    let mut offsets_freezes: Vec<TokenStream> = Vec::new();
    for layer in &layers {
        offsets_freezes.push(freeze_offsets_buf(
            &layer.offsets_buf,
            &layer.offsets,
            &pa_root,
        ));
    }
    let validity_freeze = quote! { #(#validity_freezes)* };
    let offsets_freeze = quote! { #(#offsets_freezes)* };

    let dispatch = if has_inner_option {
        quote! {
            #validity_freeze
            if #total_leaves == 0 {
                #offsets_freeze
                #consume_empty
            } else if #flat.is_empty() {
                #offsets_freeze
                #consume_all_absent
            } else if #flat.len() == #total_leaves {
                #df_decl
                #offsets_freeze
                #consume_direct
            } else {
                #df_decl
                #take_decl
                #offsets_freeze
                #consume_take
            }
        }
    } else {
        quote! {
            #validity_freeze
            if #flat.is_empty() {
                #offsets_freeze
                #consume_empty
            } else {
                #df_decl
                #offsets_freeze
                #consume_direct
            }
        }
    };

    quote! {
        {
            #precount
            let mut #flat: ::std::vec::Vec<&#type_path> =
                ::std::vec::Vec::with_capacity(#total_leaves);
            #positions_decl
            #offsets_decls
            #validity_decls
            #scan
            #dispatch
        }
    }
}

/// Wrap a per-column inner Series expression in the composed shape's
/// `LargeListArray::new` layers, routed through the assemble helper. Mirrors
/// `emit::ctb_layer_wrap` but owns its layer idents.
#[allow(clippy::too_many_arguments)]
fn wrap_per_column_layers(
    shape: &VecLayers,
    layers: &[LayerIdents],
    inner_col_expr: &TokenStream,
    pp: &TokenStream,
    inner_chunk: &syn::Ident,
    inner_col: &syn::Ident,
    inner_rech: &syn::Ident,
    dtype: &syn::Ident,
) -> TokenStream {
    if shape.depth() == 0 {
        return inner_col_expr.clone();
    }
    let chunk_decl = quote! {
        let #inner_col: #pp::Series = #inner_col_expr;
        let #inner_rech = #inner_col.rechunk();
        let #inner_chunk: #pp::ArrayRef = #inner_rech.chunks()[0].clone();
    };
    let wrap_layers = tuple_layer_wraps_clone(shape, layers);
    let stack = shape_assemble_list_stack(
        quote! { #inner_chunk },
        quote! { #inner_chunk.dtype().clone() },
        &wrap_layers,
        quote! { (*#dtype).clone() },
        &idents::tuple_layer_list_arr,
    );
    quote! {{
        #chunk_decl
        #stack
    }}
}

/// Recursive emission for a tuple-as-element. The inner tuple's columns
/// live under `<column_prefix>.field_<j>` (already prefixed by the caller).
/// Composition: the OUTER tuple's element index has already been baked
/// into `column_prefix`; this function projects into the OUTER tuple's
/// element at the access level, then recurses into each inner element
/// using a fresh `emit_element` call with the projected access as the new
/// `parent_access`.
///
/// Two operationally distinct cases:
///
/// - **No outer tuple wrappers** (parent + `outer_elem` both unwrapped, the
///   common `((A, B), C)` shape): the projection is a static access path
///   `(parent_access).<outer_idx>`. We recurse directly with that as the
///   new `parent_access`.
/// - **Wrapped outer** (rare — `Vec<((A, B), C)>` or
///   `Option<((A, B), C)>`): the projection cannot be a single static
///   path because the wrappers must be walked per-row. The parser rejects
///   these shapes before codegen and points the user at hoisting the inner
///   tuple into a named struct.
#[allow(clippy::too_many_arguments)]
fn emit_inner_tuple(
    parent_access: &TokenStream,
    parent_wrapper: &WrapperShape,
    outer_elem: &TupleElement,
    outer_elem_idx: usize,
    inner_elements: &[TupleElement],
    field_idx: usize,
    column_prefix: &str,
    config: &MacroConfig,
) -> TokenStream {
    // Static projection fast path: parent has no wrappers and the outer
    // tuple element has no wrappers. The projection is a chain of `.<i>`
    // field accesses that compose statically. Recurse with the projected
    // access expression as the new parent_access.
    let parent_static = matches!(
        parent_wrapper,
        WrapperShape::Leaf {
            option_layers: 0,
            ..
        }
    );
    let outer_static = matches!(
        &outer_elem.wrapper_shape,
        WrapperShape::Leaf {
            option_layers: 0,
            ..
        }
    );
    if parent_static && outer_static {
        let outer_li = syn::Index::from(outer_elem_idx);
        let mut outer_access = quote! { #parent_access.#outer_li };
        for _ in 0..outer_elem.outer_smart_ptr_depth {
            outer_access = quote! { (*(#outer_access)) };
        }
        let inner_wrapper = WrapperShape::Leaf {
            option_layers: 0,
            access: AccessChain::empty(),
        };
        let mut blocks: Vec<TokenStream> = Vec::with_capacity(inner_elements.len());
        for (j, inner) in inner_elements.iter().enumerate() {
            let inner_prefix = format!("{column_prefix}.field_{j}");
            blocks.push(emit_element(
                &outer_access,
                &inner_wrapper,
                inner,
                j,
                field_idx,
                &inner_prefix,
                config,
            ));
        }
        return quote! { #(#blocks)* };
    }
    unreachable!(
        "df-derive: parser should reject wrapped nested tuple projection paths before codegen"
    )
}

// ============================================================================
// Standard-encoder dispatch (no parent wrapper or Option-only parent)
// ============================================================================

/// Emit one element column via the standard `build_encoder` /
/// `build_nested_encoder`, baking the resulting Leaf decls/push/series into
/// a single self-contained block when needed (the standard encoder's Leaf
/// shape is decls-before-loop + push-in-loop + series-after-loop; we
/// orchestrate that ourselves here so the caller's `columns.push(...)`
/// happens in the right order).
fn emit_via_standard_encoder(
    access: &TokenStream,
    wrapper: &WrapperShape,
    leaf: &LeafSpec,
    field_idx: usize,
    column_prefix: &str,
    config: &MacroConfig,
) -> TokenStream {
    let pp = crate::codegen::external_paths::prelude();
    let nested_ty: TokenStream;
    let nested_ctx = match leaf {
        LeafSpec::Struct(path) => {
            nested_ty = struct_type_path(path);
            Some(NestedLeafCtx {
                base: BaseCtx {
                    access,
                    idx: field_idx,
                    name: column_prefix,
                },
                ty: &nested_ty,
                columnar_trait: &config.columnar_trait_path,
                to_df_trait: &config.to_dataframe_trait_path,
            })
        }
        LeafSpec::Generic(id) => {
            nested_ty = quote! { #id };
            Some(NestedLeafCtx {
                base: BaseCtx {
                    access,
                    idx: field_idx,
                    name: column_prefix,
                },
                ty: &nested_ty,
                columnar_trait: &config.columnar_trait_path,
                to_df_trait: &config.to_dataframe_trait_path,
            })
        }
        _ => None,
    };
    let leaf_ctx = LeafCtx {
        base: BaseCtx {
            access,
            idx: field_idx,
            name: column_prefix,
        },
        decimal128_encode_trait: &config.decimal128_encode_trait_path,
    };
    let enc = nested_ctx.as_ref().map_or_else(
        || build_encoder(leaf, wrapper, &leaf_ctx),
        |nctx| build_nested_encoder(wrapper, nctx),
    );
    match enc {
        Encoder::Leaf {
            decls,
            push,
            series,
        } => {
            let it = idents::populator_iter();
            let named = idents::field_named_series();
            let series_local = idents::vec_field_series(field_idx);
            quote! {
                {
                    #(#decls)*
                    for #it in items { #push }
                    let #series_local: #pp::Series = #series;
                    let #named = #series_local.with_name(#column_prefix.into());
                    columns.push(#named.into());
                }
            }
        }
        Encoder::Multi { columnar } => columnar,
    }
}

// ============================================================================
// Shared shape-walker adapters (Vec-bearing parent + tuple-element projection)
// ============================================================================

fn tuple_layer_idents(field_idx: usize, layer: usize) -> LayerIdents {
    LayerIdents {
        offsets: idents::tuple_layer_offsets(field_idx, layer),
        offsets_buf: idents::tuple_layer_offsets_buf(field_idx, layer),
        validity_mb: idents::tuple_layer_validity_mb(field_idx, layer),
        validity_bm: idents::tuple_layer_validity_bm(field_idx, layer),
        bind: idents::tuple_layer_bind(field_idx, layer),
    }
}

fn tuple_layer_counters(field_idx: usize, depth: usize) -> Vec<syn::Ident> {
    (0..depth.saturating_sub(1))
        .map(|layer| idents::tuple_layer_total(field_idx, layer))
        .collect()
}

#[derive(Clone, Copy)]
struct TupleProjection<'a> {
    layer: usize,
    path: &'a TokenStream,
    parent_access: &'a AccessChain,
    smart_ptr_depth: usize,
}

impl<'a> TupleProjection<'a> {
    fn as_layer_projection(self, shape: &VecLayers) -> Option<LayerProjection<'a>> {
        (self.layer < shape.depth()).then_some(LayerProjection {
            layer: self.layer,
            path: self.path,
            parent_access: self.parent_access,
            smart_ptr_depth: self.smart_ptr_depth,
        })
    }
}

fn build_precount(
    shape: &VecLayers,
    layers: &[LayerIdents],
    layer_counters: &[syn::Ident],
    total: &syn::Ident,
    access: &TokenStream,
    projection: TupleProjection<'_>,
) -> TokenStream {
    ShapePrecount {
        shape,
        access,
        layers,
        outer_some_prefix: idents::TUPLE_PRE_OUTER_SOME_PREFIX,
        total_counter: total,
        layer_counters,
        projection: projection.as_layer_projection(shape),
    }
    .build()
}

fn build_scan(
    shape: &VecLayers,
    layers: &[LayerIdents],
    access: &TokenStream,
    projection: TupleProjection<'_>,
    per_elem_push: &TokenStream,
    leaf_offsets_post_push: &TokenStream,
) -> TokenStream {
    let leaf_bind = idents::leaf_value();
    let leaf_body = |vec_bind: &TokenStream| -> TokenStream {
        if shape.inner_access.is_empty() || shape.inner_access.is_single_plain_option() {
            quote! {
                for #leaf_bind in #vec_bind.iter() {
                    #per_elem_push
                }
            }
        } else {
            let raw_bind = idents::leaf_value_raw();
            let chain_ref = super::access_chain_to_ref(&quote! { #raw_bind }, &shape.inner_access);
            let resolved = chain_ref.expr;
            if chain_ref.has_option {
                quote! {
                    for #raw_bind in #vec_bind.iter() {
                        let #leaf_bind: ::std::option::Option<_> = #resolved;
                        #per_elem_push
                    }
                }
            } else {
                quote! {
                    for #raw_bind in #vec_bind.iter() {
                        let #leaf_bind = #resolved;
                        #per_elem_push
                    }
                }
            }
        }
    };
    ShapeScan {
        shape,
        access,
        layers,
        outer_some_prefix: idents::TUPLE_OUTER_SOME_PREFIX,
        leaf_body: &leaf_body,
        leaf_offsets_post_push,
        projection: projection.as_layer_projection(shape),
    }
    .build()
}

fn build_offsets_decls(layers: &[LayerIdents], layer_counters: &[syn::Ident]) -> TokenStream {
    let offsets: Vec<&syn::Ident> = layers.iter().map(|layer| &layer.offsets).collect();
    let counter_for_depth = |layer: usize| -> TokenStream {
        let counter = &layer_counters[layer];
        quote! { #counter }
    };
    shape_offsets_decls(&offsets, &counter_for_depth)
}

fn build_validity_decls(
    shape: &VecLayers,
    layers: &[LayerIdents],
    layer_counters: &[syn::Ident],
    pa_root: &TokenStream,
) -> TokenStream {
    let validity: Vec<&syn::Ident> = layers.iter().map(|layer| &layer.validity_mb).collect();
    let counter_for_depth = |layer: usize| -> TokenStream {
        let counter = &layer_counters[layer];
        quote! { #counter }
    };
    shape_validity_decls(shape, &validity, &counter_for_depth, pa_root)
}

fn tuple_layer_wraps_move<'a>(
    shape: &VecLayers,
    layers: &'a [LayerIdents],
    pa_root: &TokenStream,
) -> Vec<LayerWrap<'a>> {
    let mut out: Vec<LayerWrap<'_>> = Vec::with_capacity(shape.depth());
    for (cur, layer) in layers.iter().enumerate() {
        let mut freeze_decl = freeze_offsets_buf(&layer.offsets_buf, &layer.offsets, pa_root);
        let validity_bm = if shape.layers[cur].has_outer_validity() {
            freeze_decl.extend(freeze_validity_bitmap(
                &layer.validity_bm,
                &layer.validity_mb,
                pa_root,
            ));
            Some(&layer.validity_bm)
        } else {
            None
        };
        out.push(LayerWrap {
            offsets_buf: OwnPolicy::Move(&layer.offsets_buf),
            validity_bm,
            freeze_decl,
        });
    }
    out
}

fn tuple_layer_wraps_clone<'a>(shape: &VecLayers, layers: &'a [LayerIdents]) -> Vec<LayerWrap<'a>> {
    let mut out: Vec<LayerWrap<'_>> = Vec::with_capacity(shape.depth());
    for (cur, layer) in layers.iter().enumerate() {
        let validity_bm = shape.layers[cur]
            .has_outer_validity()
            .then_some(&layer.validity_bm);
        out.push(LayerWrap {
            offsets_buf: OwnPolicy::Clone(&layer.offsets_buf),
            validity_bm,
            freeze_decl: TokenStream::new(),
        });
    }
    out
}

fn build_materialize(
    shape: &VecLayers,
    layers: &[LayerIdents],
    leaf_arr: &syn::Ident,
    leaf_logical_dtype: &TokenStream,
    leaf_arr_expr: &TokenStream,
    pp: &TokenStream,
    pa_root: &TokenStream,
) -> TokenStream {
    let seed_arrow_dtype_id = idents::seed_arrow_dtype();
    let seed_dtype_decl = quote! {
        let #seed_arrow_dtype_id: #pa_root::datatypes::ArrowDataType =
            #pa_root::array::Array::dtype(&#leaf_arr).clone();
    };
    let wrap_layers = tuple_layer_wraps_move(shape, layers, pa_root);
    let stack = shape_assemble_list_stack(
        quote! { ::std::boxed::Box::new(#leaf_arr) as #pp::ArrayRef },
        quote! { #seed_arrow_dtype_id },
        &wrap_layers,
        leaf_logical_dtype.clone(),
        &idents::tuple_layer_list_arr,
    );
    quote! {
        #leaf_arr_expr
        #seed_dtype_decl
        #stack
    }
}

/// Build the leaf-level value expression for a tuple-element column that
/// includes the per-element projection. Returns `None` when no projection is
/// needed at the leaf level (the element has its own wrappers, so the
/// projection is applied at an interior layer's iteration source by
/// `build_scan_iter`'s interior-layer arm). Returns `Some(expr)` when the
/// projection lands at the deepest layer — the for-loop binding
/// `__df_derive_v` is `&Tuple` and the leaf reads `(*v).<i>` (or with smart-
/// pointer derefs for elements like `(Box<i32>, String)`).
fn build_leaf_projection_value_expr(
    leaf: &LeafSpec,
    projection_path: &TokenStream,
    composed_depth: usize,
    projection_layer: usize,
    elem_outer_smart_ptr_depth: usize,
) -> Option<TokenStream> {
    if projection_layer != composed_depth {
        // Interior projection: handled by `build_scan_iter`. The leaf
        // binding `__df_derive_v` already points at the leaf, no
        // projection needed in the value expression.
        return None;
    }
    let v = idents::leaf_value();
    // Projected element value expression. For Copy primitive leaves we
    // read by value via `(*v).<i>` (the .<i> field access copies through
    // the deref of the &Tuple binding). For non-Copy leaves (String,
    // Decimal) we read by reference via `&(*v).<i>` so the leaf can
    // borrow without moving out of the tuple.
    //
    // Smart-pointer derefs on the element apply on top of the
    // projection — for `(Box<i32>, String)` element 0 is `Box<i32>`, so
    // the element's outer-smart-pointer-depth is 1; the projected
    // expression is `*((*v).0)`.
    let copy_projected = {
        let mut p = quote! { (*#v) #projection_path };
        for _ in 0..elem_outer_smart_ptr_depth {
            p = quote! { (*(#p)) };
        }
        p
    };
    let ref_projected = {
        // For each element-side smart-pointer layer, dereference inside
        // the projected expression: `&Box<T>` → `&T` via `*` on the
        // inner. Without smart pointers this collapses to `&((*v).<i>)`.
        let mut inner = quote! { (*#v) #projection_path };
        for _ in 0..elem_outer_smart_ptr_depth {
            inner = quote! { (*(#inner)) };
        }
        quote! { &(#inner) }
    };
    match leaf {
        LeafSpec::Numeric(kind) => {
            let info = crate::codegen::type_registry::numeric_info_for(*kind);
            Some(crate::codegen::type_registry::numeric_stored_value(
                *kind,
                copy_projected,
                &info.native,
            ))
        }
        LeafSpec::Bool => Some(copy_projected),
        LeafSpec::DateTime(_)
        | LeafSpec::NaiveDateTime(_)
        | LeafSpec::NaiveDate
        | LeafSpec::NaiveTime
        | LeafSpec::Duration { .. }
        | LeafSpec::Decimal { .. } => {
            // Mapped-numeric leaves run a per-row `map_primitive_expr` on
            // the binding `__df_derive_v` (e.g. `(__df_derive_v).timestamp_millis()`).
            // For tuple projection we want the mapping to operate on the
            // projected element. The leaf-pieces builder accepts a
            // `leaf_value_expr` override that names the projected
            // *receiver* of the mapping; the builder applies the standard
            // `map_primitive_expr` on top.
            //
            // For DateTime / NaiveDateTime / NaiveDate / NaiveTime / Duration / Decimal we
            // pass a reference to the projected element so the mapping's
            // method calls (`timestamp_millis`, `signed_duration_since`,
            // `num_nanoseconds`, etc.) autoderef through `&Element`.
            Some(ref_projected)
        }
        LeafSpec::String => {
            // The String arm in `build_primitive_leaf_pieces` reads
            // `.as_str()` on the override expression. We return the
            // projected `&String` reference so the caller wraps with
            // `.as_str()`.
            Some(ref_projected)
        }
        LeafSpec::Binary
        | LeafSpec::AsString(_)
        | LeafSpec::AsStr(_)
        | LeafSpec::Struct(..)
        | LeafSpec::Generic(_)
        | LeafSpec::Tuple(_) => None,
    }
}

// ============================================================================
// Primitive leaf pieces (per-element-push storage + push + leaf-array build)
// ============================================================================

struct PrimitiveLeafPieces {
    storage_decls: TokenStream,
    per_elem_push: TokenStream,
    leaf_arr_expr: TokenStream,
    leaf_offsets_post_push: TokenStream,
    extra_imports: TokenStream,
}

/// Build the per-element-push pieces for a primitive leaf at the bottom of
/// a Vec-bearing tuple shape. `leaf_value_expr` is the expression that
/// reads the leaf value FROM the for-loop binding `__df_derive_v` —
/// `*__df_derive_v` for non-projected paths and `(*__df_derive_v).<i>` (or
/// equivalent) for tuple-element projections. The leaf binding must be
/// `&LeafType` (the standard for-loop iteration shape); when the per-row
/// binding is `&Tuple`, the caller computes the projection and passes the
/// resulting `LeafType`-yielding expression.
// Bench-sensitive generated-code builder: these primitive arms preserve the
// current per-leaf emission shape used by the Vec tuple benchmarks, so this
// lint is allowed until a measured split lands.
#[allow(clippy::too_many_lines)]
fn build_primitive_leaf_pieces(
    ctx: &LeafCtx<'_>,
    leaf: &LeafSpec,
    has_inner_option: bool,
    pa_root: &TokenStream,
    leaf_value_expr: Option<&TokenStream>,
) -> PrimitiveLeafPieces {
    use super::leaf::{mb_decl_filled, validity_into_option};
    let v = idents::leaf_value();
    let total_leaves_id = idents::total_leaves();
    let leaf_capacity_expr = quote! { #total_leaves_id };
    let leaf_arr = idents::leaf_arr();

    match leaf {
        LeafSpec::Numeric(kind) => {
            let info = crate::codegen::type_registry::numeric_info_for(*kind);
            // `leaf_value_expr` is the projection-aware value expression
            // when provided; otherwise compute the standard `*__df_derive_v`
            // shape (with widening for `isize`/`usize`).
            let value_expr = leaf_value_expr.cloned().unwrap_or_else(|| {
                crate::codegen::type_registry::numeric_stored_value(
                    *kind,
                    quote! { *#v },
                    &info.native,
                )
            });
            let native = &info.native;
            let flat = idents::vec_flat();
            let validity = idents::bool_validity();
            let storage = if has_inner_option {
                let validity_decl = mb_decl_filled(&validity, &leaf_capacity_expr, true);
                quote! {
                    let mut #flat: ::std::vec::Vec<#native> =
                        ::std::vec::Vec::with_capacity(#leaf_capacity_expr);
                    #validity_decl
                }
            } else {
                quote! {
                    let mut #flat: ::std::vec::Vec<#native> =
                        ::std::vec::Vec::with_capacity(#leaf_capacity_expr);
                }
            };
            let push = if has_inner_option {
                quote! {
                    match #v {
                        ::std::option::Option::Some(#v) => {
                            #flat.push({ #value_expr });
                        }
                        ::std::option::Option::None => {
                            #flat.push(<#native as ::std::default::Default>::default());
                            #validity.set(#flat.len() - 1, false);
                        }
                    }
                }
            } else {
                quote! { #flat.push(#value_expr); }
            };
            let leaf_arr_expr = if has_inner_option {
                let valid_opt = validity_into_option(&validity);
                quote! {
                    let #leaf_arr: #pa_root::array::PrimitiveArray<#native> =
                        #pa_root::array::PrimitiveArray::<#native>::new(
                            <#native as #pa_root::types::NativeType>::PRIMITIVE.into(),
                            #flat.into(),
                            #valid_opt,
                        );
                }
            } else {
                quote! {
                    let #leaf_arr: #pa_root::array::PrimitiveArray<#native> =
                        #pa_root::array::PrimitiveArray::<#native>::from_vec(#flat);
                }
            };
            PrimitiveLeafPieces {
                storage_decls: storage,
                per_elem_push: push,
                leaf_arr_expr,
                leaf_offsets_post_push: quote! { #flat.len() },
                extra_imports: TokenStream::new(),
            }
        }
        LeafSpec::String => {
            let view_buf = idents::vec_view_buf();
            let validity = idents::bool_validity();
            let leaf_idx = idents::vec_leaf_idx();
            // The String leaf reads via `.as_str()` on the per-element
            // binding. The projection override (if provided) becomes the
            // receiver of `.as_str()`.
            let str_value_some = leaf_value_expr
                .map_or_else(|| quote! { #v.as_str() }, |e| quote! { (#e).as_str() });
            let str_value_bare = str_value_some.clone();
            let storage = if has_inner_option {
                let validity_decl = mb_decl_filled(&validity, &leaf_capacity_expr, true);
                quote! {
                    let mut #view_buf: #pa_root::array::MutableBinaryViewArray<str> =
                        #pa_root::array::MutableBinaryViewArray::<str>::with_capacity(#leaf_capacity_expr);
                    #validity_decl
                    let mut #leaf_idx: usize = 0;
                }
            } else {
                quote! {
                    let mut #view_buf: #pa_root::array::MutableBinaryViewArray<str> =
                        #pa_root::array::MutableBinaryViewArray::<str>::with_capacity(#leaf_capacity_expr);
                }
            };
            let push = if has_inner_option {
                quote! {
                    match #v {
                        ::std::option::Option::Some(#v) => {
                            #view_buf.push_value_ignore_validity({ #str_value_some });
                        }
                        ::std::option::Option::None => {
                            #view_buf.push_value_ignore_validity("");
                            #validity.set(#leaf_idx, false);
                        }
                    }
                    #leaf_idx += 1;
                }
            } else {
                quote! { #view_buf.push_value_ignore_validity({ #str_value_bare }); }
            };
            let leaf_arr_expr = if has_inner_option {
                let valid_opt = validity_into_option(&validity);
                quote! {
                    let #leaf_arr: #pa_root::array::Utf8ViewArray = #view_buf
                        .freeze()
                        .with_validity(#valid_opt);
                }
            } else {
                quote! {
                    let #leaf_arr: #pa_root::array::Utf8ViewArray = #view_buf.freeze();
                }
            };
            PrimitiveLeafPieces {
                storage_decls: storage,
                per_elem_push: push,
                leaf_arr_expr,
                leaf_offsets_post_push: quote! { #view_buf.len() },
                extra_imports: TokenStream::new(),
            }
        }
        LeafSpec::Bool => {
            let values_ident = idents::bool_values();
            let validity_ident = idents::bool_validity();
            let leaf_idx = idents::vec_leaf_idx();
            let values_decl = mb_decl_filled(&values_ident, &leaf_capacity_expr, false);
            let storage = if has_inner_option {
                let validity_decl = mb_decl_filled(&validity_ident, &leaf_capacity_expr, true);
                quote! {
                    #values_decl
                    #validity_decl
                    let mut #leaf_idx: usize = 0;
                }
            } else {
                quote! {
                    #values_decl
                    let mut #leaf_idx: usize = 0;
                }
            };
            let push = if has_inner_option {
                quote! {
                    match #v {
                        ::std::option::Option::Some(true) => {
                            #values_ident.set(#leaf_idx, true);
                        }
                        ::std::option::Option::Some(false) => {}
                        ::std::option::Option::None => {
                            #validity_ident.set(#leaf_idx, false);
                        }
                    }
                    #leaf_idx += 1;
                }
            } else {
                quote! {
                    if *#v {
                        #values_ident.set(#leaf_idx, true);
                    }
                    #leaf_idx += 1;
                }
            };
            let leaf_arr_expr = if has_inner_option {
                let valid_opt = validity_into_option(&validity_ident);
                quote! {
                    let #leaf_arr: #pa_root::array::BooleanArray =
                        #pa_root::array::BooleanArray::new(
                            #pa_root::datatypes::ArrowDataType::Boolean,
                            ::std::convert::Into::<#pa_root::bitmap::Bitmap>::into(#values_ident),
                            #valid_opt,
                        );
                }
            } else {
                quote! {
                    let #leaf_arr: #pa_root::array::BooleanArray =
                        #pa_root::array::BooleanArray::new(
                            #pa_root::datatypes::ArrowDataType::Boolean,
                            ::std::convert::Into::<#pa_root::bitmap::Bitmap>::into(#values_ident),
                            ::std::option::Option::None,
                        );
                }
            };
            PrimitiveLeafPieces {
                storage_decls: storage,
                per_elem_push: push,
                leaf_arr_expr,
                leaf_offsets_post_push: quote! { #leaf_idx },
                extra_imports: TokenStream::new(),
            }
        }
        LeafSpec::DateTime(_)
        | LeafSpec::NaiveDateTime(_)
        | LeafSpec::NaiveDate
        | LeafSpec::NaiveTime
        | LeafSpec::Duration { .. }
        | LeafSpec::Decimal { .. } => {
            // The mapped-numeric path runs `map_primitive_expr` on a
            // receiver expression. For tuple projection the receiver is
            // the projected element (`&(*v).<i>`); for non-projected
            // shapes, it's the standard for-loop binding `__df_derive_v`.
            // The Some-arm rebinds `__df_derive_v` so the override is
            // only consulted at the bare arm.
            let receiver_bare = leaf_value_expr.cloned().unwrap_or_else(|| quote! { #v });
            let mapped_v = crate::codegen::type_registry::map_primitive_expr(
                &receiver_bare,
                leaf,
                ctx.decimal128_encode_trait,
            );
            let native: TokenStream = match leaf {
                LeafSpec::DateTime(_)
                | LeafSpec::NaiveDateTime(_)
                | LeafSpec::NaiveTime
                | LeafSpec::Duration { .. } => quote! { i64 },
                LeafSpec::NaiveDate => quote! { i32 },
                LeafSpec::Decimal { .. } => quote! { i128 },
                _ => unreachable!(),
            };
            let flat = idents::vec_flat();
            let validity = idents::bool_validity();
            let needs_decimal_import = matches!(leaf, LeafSpec::Decimal { .. });
            let extra_imports = if needs_decimal_import {
                let trait_path = ctx.decimal128_encode_trait;
                quote! { use #trait_path as _; }
            } else {
                TokenStream::new()
            };
            let storage = if has_inner_option {
                let validity_decl = mb_decl_filled(&validity, &leaf_capacity_expr, true);
                quote! {
                    let mut #flat: ::std::vec::Vec<#native> =
                        ::std::vec::Vec::with_capacity(#leaf_capacity_expr);
                    #validity_decl
                }
            } else {
                quote! {
                    let mut #flat: ::std::vec::Vec<#native> =
                        ::std::vec::Vec::with_capacity(#leaf_capacity_expr);
                }
            };
            let push = if has_inner_option {
                quote! {
                    match #v {
                        ::std::option::Option::Some(#v) => {
                            #flat.push({ #mapped_v });
                        }
                        ::std::option::Option::None => {
                            #flat.push(<#native as ::std::default::Default>::default());
                            #validity.set(#flat.len() - 1, false);
                        }
                    }
                }
            } else {
                quote! { #flat.push(#mapped_v); }
            };
            let leaf_arr_expr = if has_inner_option {
                let valid_opt = validity_into_option(&validity);
                quote! {
                    let #leaf_arr: #pa_root::array::PrimitiveArray<#native> =
                        #pa_root::array::PrimitiveArray::<#native>::new(
                            <#native as #pa_root::types::NativeType>::PRIMITIVE.into(),
                            #flat.into(),
                            #valid_opt,
                        );
                }
            } else {
                quote! {
                    let #leaf_arr: #pa_root::array::PrimitiveArray<#native> =
                        #pa_root::array::PrimitiveArray::<#native>::from_vec(#flat);
                }
            };
            PrimitiveLeafPieces {
                storage_decls: storage,
                per_elem_push: push,
                leaf_arr_expr,
                leaf_offsets_post_push: quote! { #flat.len() },
                extra_imports,
            }
        }
        LeafSpec::Binary => {
            let view_buf = idents::vec_view_buf();
            let validity = idents::bool_validity();
            let leaf_idx = idents::vec_leaf_idx();
            let storage = if has_inner_option {
                let validity_decl = mb_decl_filled(&validity, &leaf_capacity_expr, true);
                quote! {
                    let mut #view_buf: #pa_root::array::MutableBinaryViewArray<[u8]> =
                        #pa_root::array::MutableBinaryViewArray::<[u8]>::with_capacity(#leaf_capacity_expr);
                    #validity_decl
                    let mut #leaf_idx: usize = 0;
                }
            } else {
                quote! {
                    let mut #view_buf: #pa_root::array::MutableBinaryViewArray<[u8]> =
                        #pa_root::array::MutableBinaryViewArray::<[u8]>::with_capacity(#leaf_capacity_expr);
                }
            };
            let empty = quote! { &[][..] };
            let push = if has_inner_option {
                quote! {
                    match #v {
                        ::std::option::Option::Some(#v) => {
                            #view_buf.push_value_ignore_validity(&#v[..]);
                        }
                        ::std::option::Option::None => {
                            #view_buf.push_value_ignore_validity(#empty);
                            #validity.set(#leaf_idx, false);
                        }
                    }
                    #leaf_idx += 1;
                }
            } else {
                quote! { #view_buf.push_value_ignore_validity(&#v[..]); }
            };
            let leaf_arr_expr = if has_inner_option {
                let valid_opt = validity_into_option(&validity);
                quote! {
                    let #leaf_arr: #pa_root::array::BinaryViewArray = #view_buf
                        .freeze()
                        .with_validity(#valid_opt);
                }
            } else {
                quote! {
                    let #leaf_arr: #pa_root::array::BinaryViewArray = #view_buf.freeze();
                }
            };
            PrimitiveLeafPieces {
                storage_decls: storage,
                per_elem_push: push,
                leaf_arr_expr,
                leaf_offsets_post_push: quote! { #view_buf.len() },
                extra_imports: TokenStream::new(),
            }
        }
        LeafSpec::AsString(_) => unreachable!(
            "as_string is rejected on tuple-typed fields by parser::reject_attrs_on_tuple",
        ),
        LeafSpec::AsStr(_) => unreachable!(
            "as_str is rejected on tuple-typed fields by parser::reject_attrs_on_tuple",
        ),
        LeafSpec::Struct(..) | LeafSpec::Generic(_) | LeafSpec::Tuple(_) => unreachable!(
            "build_primitive_leaf_pieces reached with non-primitive leaf — those route \
             through emit_vec_parent_nested or emit_inner_tuple",
        ),
    }
}

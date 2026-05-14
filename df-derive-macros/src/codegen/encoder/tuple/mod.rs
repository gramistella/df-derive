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

mod entries;
mod projection;
mod standard;
mod vec_parent;

use crate::codegen::MacroConfig;
use crate::codegen::strategy::FieldEmit;
use crate::ir::{
    AccessChain, FieldIR, LeafRoute, LeafShape, NestedLeaf, PrimitiveLeaf, TupleElement, VecLayers,
    WrapperShape,
};
use proc_macro2::TokenStream;
use quote::quote;

use super::idents;
use super::shape_walk::{
    LayerIdents, freeze_offsets_buf, freeze_validity_bitmap, shape_assemble_list_stack,
};
use super::{idx_size_len_expr, struct_type_tokens};
use projection::{
    TupleProjection, compose_option_with_element, option_tuple_projection_receiver,
    project_parent_option_tuple_element,
};
use standard::{emit_via_standard_encoder, emit_via_standard_encoder_with_option_receiver};
use vec_parent::{
    TupleScanLeaf, build_offsets_decls, build_precount, build_scan, build_validity_decls,
    emit_vec_parent, tuple_layer_counters, tuple_layer_idents, tuple_layer_wraps_clone,
};

pub use entries::build_field_entries;

fn tuple_nested_type_path(nested: NestedLeaf<'_>) -> TokenStream {
    match nested {
        NestedLeaf::Struct(ty) => struct_type_tokens(ty),
        NestedLeaf::Generic(id) => quote! { #id },
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
    let parent_name = crate::codegen::helpers::column_name_for_ident(&field.name);

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

    FieldEmit::WholeColumn { builders }
}

#[derive(Clone, Copy)]
pub(super) enum TupleLeafRoute<'a> {
    Primitive(PrimitiveLeaf<'a>),
    Nested(NestedLeaf<'a>),
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
    let leaf_route = match elem.leaf_spec.route() {
        LeafRoute::Tuple(inner) => {
            return emit_inner_tuple(
                parent_access,
                elem,
                elem_idx,
                inner,
                field_idx,
                column_prefix,
                config,
            );
        }
        LeafRoute::Primitive(leaf) => TupleLeafRoute::Primitive(leaf),
        LeafRoute::Nested(nested) => TupleLeafRoute::Nested(nested),
    };

    match parent_wrapper {
        // Bare `(A, B)` (or `Box<(A, B)>` etc., with smart pointers already
        // dereffed in `parent_access`). Project directly via `.<i>` and
        // route through the standard encoder with the element's own
        // wrapper shape and leaf.
        WrapperShape::Leaf(LeafShape::Bare) => {
            let elem_li = syn::Index::from(elem_idx);
            let mut access = quote! { #parent_access.#elem_li };
            for _ in 0..elem.outer_smart_ptr_depth {
                access = quote! { (*(#access)) };
            }
            emit_via_standard_encoder(
                &access,
                &elem.wrapper_shape,
                leaf_route,
                field_idx,
                column_prefix,
                config,
            )
        }
        // `Option<(A, B)>` / `Option<...<Option<(A, B)>>>`. Collapse parent
        // Options to `Option<&Tuple>` per row, project through an element
        // reference that has already applied the element's outer smart-pointer
        // depth, then route through the standard encoder with the composed
        // wrapper.
        WrapperShape::Leaf(LeafShape::Optional { access, .. }) => {
            let collapsed_parent = super::access_chain_to_option_ref(parent_access, access);
            let projected = project_parent_option_tuple_element(&collapsed_parent, elem, elem_idx);
            let composed_shape = compose_option_with_element(&elem.wrapper_shape);
            emit_via_standard_encoder_with_option_receiver(
                &projected,
                &composed_shape,
                leaf_route,
                field_idx,
                column_prefix,
                config,
                option_tuple_projection_receiver(elem),
            )
        }
        // Parent has at least one Vec layer. Construct a composed emission
        // that walks the composed wrapper stack (parent + element layers)
        // with `.iter()` rebased to project at the boundary.
        WrapperShape::Vec(parent_layers) => emit_vec_parent(
            parent_access,
            parent_layers,
            elem,
            leaf_route,
            elem_idx,
            field_idx,
            column_prefix,
            config,
        ),
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
    leaf_projection_access: Option<&AccessChain>,
    type_path: &TokenStream,
    field_idx: usize,
    column_prefix: &str,
    config: &MacroConfig,
) -> TokenStream {
    let pp = config.external_paths.prelude();
    let pa_root = config.external_paths.polars_arrow_root();
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
    let validate_nested_frame = idents::validate_nested_frame();
    let validate_nested_column_dtype = idents::validate_nested_column_dtype();

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
    // inner-Option). The scan applies tuple projection before this body, so
    // the leaf binding is always `&Inner` or `Option<&Inner>`.
    let leaf_v = idents::leaf_value();
    let inner_v = idents::tuple_nested_inner_v();
    let projected_leaf = quote! { #leaf_v };
    let per_elem_push = if has_inner_option {
        let flat_idx = idx_size_len_expr(&flat, pp);
        quote! {
            match #projected_leaf {
                ::std::option::Option::Some(#inner_v) => {
                    #positions.push(::std::option::Option::Some(
                        #flat_idx,
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
        TupleScanLeaf {
            projection_access: leaf_projection_access,
            per_elem_push: &per_elem_push,
            offsets_post_push: &leaf_offsets_post_push,
            pp,
        },
    );
    let offsets_decls = build_offsets_decls(&layers, &layer_counters);
    let validity_decls = build_validity_decls(composed_shape, &layers, &layer_counters, pa_root);

    let positions_decl = if has_inner_option {
        quote! {
            let mut #positions: ::std::vec::Vec<::std::option::Option<#pp::IdxSize>> =
                ::std::vec::Vec::with_capacity(#total_leaves);
        }
    } else {
        TokenStream::new()
    };

    // Per-column inner-Series expressions for the four dispatch branches.
    let inner_col_direct = quote! {{
        let #inner_full = #df.column(#col_name)?.as_materialized_series();
        #validate_nested_column_dtype(#inner_full, #col_name, #dtype)?;
        #inner_full.clone()
    }};
    let inner_col_take = quote! {{
        let #inner_full = #df.column(#col_name)?.as_materialized_series();
        #validate_nested_column_dtype(#inner_full, #col_name, #dtype)?;
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
        pp,
        &inner_chunk,
        &inner_col,
        &inner_rech,
        &dtype,
        pa_root,
    );
    let series_take = wrap_per_column_layers(
        composed_shape,
        &layers,
        &inner_col_take,
        pp,
        &inner_chunk,
        &inner_col,
        &inner_rech,
        &dtype,
        pa_root,
    );
    let series_empty = wrap_per_column_layers(
        composed_shape,
        &layers,
        &inner_col_empty,
        pp,
        &inner_chunk,
        &inner_col,
        &inner_rech,
        &dtype,
        pa_root,
    );
    let series_all_absent = wrap_per_column_layers(
        composed_shape,
        &layers,
        &inner_col_all_absent,
        pp,
        &inner_chunk,
        &inner_col,
        &inner_rech,
        &dtype,
        pa_root,
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
        #validate_nested_frame(&#df, #flat.len(), ::core::any::type_name::<#type_path>())?;
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
            pa_root,
        ));
    }
    let mut offsets_freezes: Vec<TokenStream> = Vec::new();
    for layer in &layers {
        offsets_freezes.push(freeze_offsets_buf(
            &layer.offsets_buf,
            &layer.offsets,
            pa_root,
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
    pa_root: &TokenStream,
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
        pp,
        pa_root,
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
    outer_elem: &TupleElement,
    outer_elem_idx: usize,
    inner_elements: &[TupleElement],
    field_idx: usize,
    column_prefix: &str,
    config: &MacroConfig,
) -> TokenStream {
    let outer_li = syn::Index::from(outer_elem_idx);
    let mut outer_access = quote! { #parent_access.#outer_li };
    for _ in 0..outer_elem.outer_smart_ptr_depth {
        outer_access = quote! { (*(#outer_access)) };
    }
    let inner_wrapper = WrapperShape::Leaf(LeafShape::Bare);
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
    quote! { #(#blocks)* }
}

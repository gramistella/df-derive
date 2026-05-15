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
mod nested;
mod projection;
mod standard;
mod vec_parent;

use crate::codegen::MacroConfig;
use crate::codegen::strategy::FieldEmit;
use crate::ir::{
    FieldIR, LeafRoute, LeafShape, NestedLeaf, PrimitiveLeaf, TupleElement, WrapperShape,
};
use proc_macro2::TokenStream;
use quote::quote;

use super::idents;
use super::struct_type_tokens;
use projection::{
    compose_option_with_element, option_tuple_projection_receiver,
    project_parent_option_tuple_element,
};
use standard::{emit_via_standard_encoder, emit_via_standard_encoder_with_option_receiver};
use vec_parent::emit_vec_parent;

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
    let parent_name = crate::codegen::names::column_name_for_ident(&field.name);

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

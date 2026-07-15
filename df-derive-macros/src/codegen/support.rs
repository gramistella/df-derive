use super::{MacroConfig, encoder};
use crate::ir::{FieldPlan, StructIR, TerminalLeafRoute, TupleNode, TupleNodeKind, WrapperShape};
use proc_macro2::TokenStream;
use quote::quote;

fn primitive_vec_helper_needs(ir: &StructIR) -> encoder::PrimitiveVecHelperNeeds {
    fn include(
        needs: &mut encoder::PrimitiveVecHelperNeeds,
        leaf: crate::ir::PrimitiveLeaf<'_>,
        wrapper: &WrapperShape,
    ) {
        let WrapperShape::Vec(shape) = wrapper else {
            return;
        };
        let column_needs = encoder::primitive_vec_helper_needs(leaf, shape);
        needs.push_reserved |= column_needs.push_reserved;
        needs.set_prepared_bitmap |= column_needs.set_prepared_bitmap;
    }

    fn visit_tuple(node: &TupleNode, needs: &mut encoder::PrimitiveVecHelperNeeds) {
        match node.kind() {
            TupleNodeKind::Leaf(common) => {
                if let TerminalLeafRoute::Primitive(leaf) = common.leaf_spec().route() {
                    include(needs, leaf, node.wrapper_shape());
                }
            }
            TupleNodeKind::Tuple(elements) => {
                for child in elements.iter() {
                    visit_tuple(child, needs);
                }
            }
        }
    }

    let mut needs = encoder::PrimitiveVecHelperNeeds::default();
    for field in &ir.fields {
        match field {
            FieldPlan::Column(column) => {
                if let TerminalLeafRoute::Primitive(leaf) = column.leaf_spec().route() {
                    include(&mut needs, leaf, column.wrapper_shape());
                }
            }
            FieldPlan::Tuple(tuple) => {
                for node in tuple.elements().iter() {
                    visit_tuple(node, &mut needs);
                }
            }
        }
    }
    needs
}

pub(in crate::codegen) fn generate_support(ir: &StructIR, config: &MacroConfig) -> TokenStream {
    let pa_root = config.external_paths.polars_arrow_root();
    let ident_scope = encoder::idents::GeneratedIdentScope::new(&ir.generics);
    let push_reserved = encoder::idents::push_reserved(ident_scope);
    let set_prepared_bitmap = encoder::idents::set_prepared_bitmap(ident_scope);

    let helper_needs = primitive_vec_helper_needs(ir);
    let push_reserved_helper = helper_needs.push_reserved.then(|| {
        quote! {
            #[inline(always)]
            #[allow(clippy::inline_always)]
            unsafe fn #push_reserved<T>(
                destination: &mut ::std::vec::Vec<T>,
                value: T,
            ) {
                let initialized = destination.len();
                debug_assert!(initialized < destination.capacity());
                // SAFETY: the caller promises that `initialized` names one
                // reserved, uninitialized slot. The value is written before
                // the initialized length advances.
                unsafe {
                    ::core::ptr::write(
                        destination.as_mut_ptr().add(initialized),
                        value,
                    );
                    destination.set_len(initialized + 1);
                }
            }
        }
    });

    let set_prepared_bitmap_helper = helper_needs.set_prepared_bitmap.then(|| {
        quote! {
            #[inline(always)]
            #[allow(clippy::inline_always)]
            unsafe fn #set_prepared_bitmap(
                bitmap: &mut #pa_root::bitmap::MutableBitmap,
                index: usize,
                value: bool,
            ) {
                debug_assert!(index < bitmap.len());
                // SAFETY: the caller promises that `index` is inside the
                // bitmap range prepared by its list schedule.
                unsafe {
                    bitmap.set_unchecked(index, value);
                }
            }
        }
    });

    quote! {
        #push_reserved_helper
        #set_prepared_bitmap_helper
    }
}

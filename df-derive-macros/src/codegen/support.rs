use super::{MacroConfig, encoder};
use crate::ir::{FieldPlan, StructIR, TerminalLeafRoute, TupleNode, TupleNodeKind, WrapperShape};
use proc_macro2::TokenStream;
use quote::quote;

fn needs_list_assembly(ir: &StructIR) -> bool {
    ir.fields.iter().any(crate::ir::FieldPlan::has_vec_shape)
}

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

#[allow(clippy::too_many_lines)]
pub(in crate::codegen) fn generate_support(ir: &StructIR, config: &MacroConfig) -> TokenStream {
    let pp = config.external_paths.prelude();
    let pa_root = config.external_paths.polars_arrow_root();
    let assemble_helper = encoder::idents::assemble_helper();
    let list_assembly = encoder::idents::list_assembly();
    let ident_scope = encoder::idents::GeneratedIdentScope::new(&ir.generics);
    let push_reserved = encoder::idents::push_reserved(ident_scope);
    let set_prepared_bitmap = encoder::idents::set_prepared_bitmap(ident_scope);

    let list_assembly_helpers = if needs_list_assembly(ir) {
        quote! {
            struct #list_assembly {
                list_arr: #pp::LargeListArray,
                logical_dtype: #pp::DataType,
            }

            impl #list_assembly {
                #[inline(always)]
                #[allow(clippy::inline_always)]
                fn new(
                    list_arr: #pp::LargeListArray,
                    inner_logical_dtype: #pp::DataType,
                ) -> Self {
                    Self {
                        list_arr,
                        logical_dtype: #pp::DataType::List(
                            ::std::boxed::Box::new(inner_logical_dtype),
                        ),
                    }
                }

                #[inline(always)]
                #[allow(clippy::inline_always)]
                fn into_series(self) -> #pp::PolarsResult<#pp::Series> {
                    let expected_arrow_dtype: #pa_root::datatypes::ArrowDataType =
                        self.logical_dtype
                            .to_physical()
                            .to_arrow(#pp::CompatLevel::newest());
                    let actual_arrow_dtype = #pa_root::array::Array::dtype(&self.list_arr);
                    if actual_arrow_dtype != &expected_arrow_dtype {
                        return ::std::result::Result::Err(#pp::polars_err!(
                            ComputeError:
                            "df-derive: list assembly dtype mismatch: actual Arrow dtype {:?}, logical dtype {:?}",
                            actual_arrow_dtype,
                            self.logical_dtype,
                        ));
                    }
                    let Self {
                        list_arr,
                        logical_dtype,
                    } = self;
                    // SAFETY: `Self::new` is the generated list assembly
                    // boundary. Every caller reaches it through
                    // `encoder::shape_walk::shape_assemble_list_stack`,
                    // which builds `list_arr` from the leaf/nested physical
                    // Arrow dtype and the same logical dtype that schema
                    // generation emits. The release-mode check above
                    // compares the final Arrow list dtype against
                    // `logical_dtype.to_physical()`, covering logical
                    // wrappers such as Date, Datetime, Duration, Time,
                    // Decimal, and nested List envelopes. This matters for
                    // manual `ColumnarSpec` implementations: a bad declared
                    // dtype can no longer violate the unchecked
                    // constructor's dtype invariant.
                    unsafe {
                        ::std::result::Result::Ok(#pp::Series::from_chunks_and_dtype_unchecked(
                            "".into(),
                            ::std::vec![
                                ::std::boxed::Box::new(list_arr) as #pp::ArrayRef,
                            ],
                            &logical_dtype,
                        ))
                    }
                }
            }

            #[inline(always)]
            #[allow(non_snake_case, clippy::inline_always)]
            fn #assemble_helper(
                list_arr: #pp::LargeListArray,
                inner_logical_dtype: #pp::DataType,
            ) -> #pp::PolarsResult<#pp::Series> {
                #list_assembly::new(list_arr, inner_logical_dtype).into_series()
            }
        }
    } else {
        TokenStream::new()
    };

    let helper_needs = primitive_vec_helper_needs(ir);
    let push_reserved_helper = helper_needs.push_reserved.then(|| {
        quote! {
            #[inline(always)]
            #[allow(clippy::inline_always)]
            fn #push_reserved<T>(
                destination: &mut ::std::vec::Vec<T>,
                value: T,
            ) {
                // Every generated caller first reserves its observed segment
                // or allocates the exact deferred leaf count. The assertion
                // diagnoses future codegen violations without a release branch.
                debug_assert!(destination.len() < destination.capacity());
                let initialized = destination.len();
                // SAFETY: the codegen invariant documented above proves
                // `initialized` names a spare slot. The value is written
                // before the initialized length is advanced.
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
            fn #set_prepared_bitmap(
                bitmap: &mut #pa_root::bitmap::MutableBitmap,
                index: usize,
                value: bool,
            ) {
                // Generated primitive-list schedules size this bitmap before
                // the element loop and advance the index once per value.
                debug_assert!(index < bitmap.len());
                // SAFETY: the const-scoped helper is unnameable by user code;
                // all generated call sites establish the bound above.
                unsafe {
                    bitmap.set_unchecked(index, value);
                }
            }
        }
    });

    quote! {
        #list_assembly_helpers
        #push_reserved_helper
        #set_prepared_bitmap_helper
    }
}

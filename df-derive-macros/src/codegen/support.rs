use super::{MacroConfig, encoder};
use crate::ir::StructIR;
use proc_macro2::TokenStream;
use quote::quote;

fn needs_list_assembly(ir: &StructIR) -> bool {
    ir.fields.iter().any(crate::ir::FieldPlan::has_vec_shape)
}

#[allow(clippy::too_many_lines)]
pub(in crate::codegen) fn generate_support(ir: &StructIR, config: &MacroConfig) -> TokenStream {
    let pp = config.external_paths.prelude();
    let pa_root = config.external_paths.polars_arrow_root();
    let assemble_helper = encoder::idents::assemble_helper();
    let list_assembly = encoder::idents::list_assembly();

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

    quote! {
        #list_assembly_helpers
    }
}

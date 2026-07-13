use super::{MacroConfig, encoder};
use crate::ir::{StructIR, TerminalLeafRoute};
use proc_macro2::TokenStream;
use quote::quote;

fn needs_list_assembly(ir: &StructIR) -> bool {
    ir.columns.iter().any(|column| column.vec_depth() > 0)
}

fn needs_nested_validation(ir: &StructIR) -> bool {
    ir.columns
        .iter()
        .any(|column| matches!(column.leaf_spec().route(), TerminalLeafRoute::Nested(_)))
}

pub(in crate::codegen) fn needs_unique_name_validation(ir: &StructIR) -> bool {
    ir.columns.iter().any(|column| {
        column
            .nested_name_policy()
            .requires_unique_name_validation()
    })
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
                    // safe manual `ToDataFrame` / `Columnar` implementations:
                    // a bad schema can no longer violate the unchecked
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

    let nested_validation_helpers = if needs_nested_validation(ir) {
        let validate_nested_frame = encoder::idents::validate_nested_frame();

        quote! {
            #[inline(always)]
            #[allow(non_snake_case, clippy::inline_always)]
            fn #validate_nested_frame(
                df: &#pp::DataFrame,
                expected_schema: &#pp::SchemaRef,
                expected_height: usize,
                type_name: &str,
            ) -> #pp::PolarsResult<()> {
                let actual_height = df.height();
                if actual_height != expected_height {
                    return ::std::result::Result::Err(#pp::polars_err!(
                        ComputeError:
                        "df-derive: nested Columnar::encode for {} returned height {}, expected {}",
                        type_name,
                        actual_height,
                        expected_height,
                    ));
                }
                let actual_columns = df.columns();
                if actual_columns.len() != expected_schema.len() {
                    return ::std::result::Result::Err(#pp::polars_err!(
                        ComputeError:
                        "df-derive: nested Columnar::encode for {} returned schema width {}, expected {}",
                        type_name,
                        actual_columns.len(),
                        expected_schema.len(),
                    ));
                }
                for (index, (actual_column, (expected_name, expected_dtype))) in
                    actual_columns.iter().zip(expected_schema.iter()).enumerate()
                {
                    let actual_name = actual_column.name();
                    let actual_dtype = actual_column.dtype();
                    if actual_name != expected_name {
                        return ::std::result::Result::Err(#pp::polars_err!(
                            ComputeError:
                            "df-derive: nested Columnar::encode for {} returned column `{}` at index {}, expected `{}`",
                            type_name,
                            actual_name,
                            index,
                            expected_name,
                        ));
                    }
                    if actual_dtype != expected_dtype {
                        return ::std::result::Result::Err(#pp::polars_err!(
                            ComputeError:
                            "df-derive: nested Columnar::encode for {} returned dtype {:?} for column `{}` at index {}, expected {:?}",
                            type_name,
                            actual_dtype,
                            actual_name,
                            index,
                            expected_dtype,
                        ));
                    }
                }
                ::std::result::Result::Ok(())
            }
        }
    } else {
        TokenStream::new()
    };

    let unique_name_validation_helper = if needs_unique_name_validation(ir) {
        let validate_unique_column_names = encoder::idents::validate_unique_column_names();

        quote! {
            #[inline(always)]
            #[allow(non_snake_case, clippy::inline_always)]
            fn #validate_unique_column_names<'a, I>(
                names: I,
                type_name: &str,
            ) -> #pp::PolarsResult<()>
            where
                I: ::core::iter::IntoIterator<Item = &'a str>,
            {
                let mut seen: ::std::collections::BTreeSet<&'a str> =
                    ::std::collections::BTreeSet::new();
                for name in names {
                    if !seen.insert(name) {
                        return ::std::result::Result::Err(#pp::polars_err!(
                            ComputeError:
                            "df-derive: duplicate column `{}` while building flattened DataFrame output for {}",
                            name,
                            type_name,
                        ));
                    }
                }
                ::std::result::Result::Ok(())
            }
        }
    } else {
        TokenStream::new()
    };

    quote! {
        #list_assembly_helpers

        #nested_validation_helpers

        #unique_name_validation_helper
    }
}

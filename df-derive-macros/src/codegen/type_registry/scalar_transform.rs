use proc_macro2::TokenStream;
use quote::quote;

use crate::ir::{DateTimeUnit, DurationSource};

use crate::codegen::external_paths::ExternalPaths;

use super::logical_dtype::LogicalPrimitive;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::codegen) enum ScalarTransform {
    DateTime(DateTimeUnit),
    NaiveDateTime(DateTimeUnit),
    NaiveDate,
    NaiveTime,
    Duration {
        unit: DateTimeUnit,
        source: DurationSource,
    },
    Decimal {
        precision: u8,
        scale: u8,
    },
}

impl ScalarTransform {
    const fn logical(self) -> LogicalPrimitive {
        match self {
            Self::DateTime(unit) => LogicalPrimitive::DateTime(unit),
            Self::NaiveDateTime(unit) => LogicalPrimitive::NaiveDateTime(unit),
            Self::NaiveDate => LogicalPrimitive::NaiveDate,
            Self::NaiveTime => LogicalPrimitive::NaiveTime,
            Self::Duration { unit, source: _ } => LogicalPrimitive::Duration(unit),
            Self::Decimal { precision, scale } => LogicalPrimitive::Decimal { precision, scale },
        }
    }

    pub(in crate::codegen) fn dtype(self, paths: &ExternalPaths) -> TokenStream {
        self.logical().dtype(paths)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::codegen) enum PrimitiveExprReceiver {
    Place,
    Ref,
    RefRef,
}

impl PrimitiveExprReceiver {
    fn decimal_receiver(self, var: &TokenStream) -> TokenStream {
        match self {
            Self::Place => quote! { &(#var) },
            Self::Ref => quote! { #var },
            Self::RefRef => quote! { *(#var) },
        }
    }
}

#[allow(clippy::too_many_lines)]
pub(in crate::codegen) fn map_primitive_expr(
    var: &TokenStream,
    receiver: PrimitiveExprReceiver,
    leaf: ScalarTransform,
    decimal128_encode_trait: &syn::Path,
    paths: &ExternalPaths,
) -> TokenStream {
    match leaf {
        ScalarTransform::DateTime(unit) => match unit {
            DateTimeUnit::Milliseconds => quote! { (#var).timestamp_millis() },
            DateTimeUnit::Microseconds => quote! { (#var).timestamp_micros() },
            DateTimeUnit::Nanoseconds => {
                let pp = paths.prelude();
                quote! {
                    (#var).timestamp_nanos_opt().ok_or_else(|| #pp::polars_err!(
                        ComputeError: "df-derive: DateTime<Tz> value is out of range for nanosecond timestamps (chrono supports approximately 1677..2262)"
                    ))?
                }
            }
        },
        ScalarTransform::NaiveDateTime(unit) => match unit {
            DateTimeUnit::Milliseconds => quote! { (#var).and_utc().timestamp_millis() },
            DateTimeUnit::Microseconds => quote! { (#var).and_utc().timestamp_micros() },
            DateTimeUnit::Nanoseconds => {
                let pp = paths.prelude();
                quote! {
                    (#var).and_utc().timestamp_nanos_opt().ok_or_else(|| #pp::polars_err!(
                        ComputeError: "df-derive: NaiveDateTime value is out of range for nanosecond timestamps (chrono supports approximately 1677..2262)"
                    ))?
                }
            }
        },
        ScalarTransform::NaiveDate => {
            let chrono = crate::codegen::external_paths::chrono_root();
            quote! {
                {
                    use #chrono::Datelike as _;
                    (#var).num_days_from_ce() - 719_163
                }
            }
        }
        ScalarTransform::NaiveTime => {
            let chrono = crate::codegen::external_paths::chrono_root();
            quote! {
                {
                    use #chrono::Timelike as _;
                    ((#var).num_seconds_from_midnight() as i64) * 1_000_000_000
                        + ((#var).nanosecond() as i64)
                }
            }
        }
        ScalarTransform::Duration { unit, source } => {
            let pp = paths.prelude();
            match source {
                DurationSource::Std => match unit {
                    DateTimeUnit::Nanoseconds => quote! {
                        <i64 as ::core::convert::TryFrom<u128>>::try_from((#var).as_nanos()).map_err(|_| #pp::polars_err!(
                            ComputeError: "df-derive: std::time::Duration value out of i64 ns range"
                        ))?
                    },
                    DateTimeUnit::Microseconds => quote! {
                        <i64 as ::core::convert::TryFrom<u128>>::try_from((#var).as_micros()).map_err(|_| #pp::polars_err!(
                            ComputeError: "df-derive: std::time::Duration value out of i64 us range"
                        ))?
                    },
                    DateTimeUnit::Milliseconds => quote! {
                        <i64 as ::core::convert::TryFrom<u128>>::try_from((#var).as_millis()).map_err(|_| #pp::polars_err!(
                            ComputeError: "df-derive: std::time::Duration value out of i64 ms range"
                        ))?
                    },
                },
                DurationSource::Chrono => match unit {
                    DateTimeUnit::Nanoseconds => quote! {
                        (#var).num_nanoseconds().ok_or_else(|| #pp::polars_err!(
                            ComputeError: "df-derive: chrono::Duration value out of i64 ns range"
                        ))?
                    },
                    DateTimeUnit::Microseconds => quote! {
                        (#var).num_microseconds().ok_or_else(|| #pp::polars_err!(
                            ComputeError: "df-derive: chrono::Duration value out of i64 us range"
                        ))?
                    },
                    DateTimeUnit::Milliseconds => quote! { (#var).num_milliseconds() },
                },
            }
        }
        ScalarTransform::Decimal { precision, scale } => {
            // Decimal backends own rescaling through `Decimal128Encode`; the
            // result must match Polars decimal mantissa semantics.
            let target = u32::from(scale);
            let precision = u32::from(precision);
            let scale = u32::from(scale);
            let pp = paths.prelude();
            let decimal_receiver = receiver.decimal_receiver(var);
            quote! {{
                match #decimal128_encode_trait::try_to_i128_mantissa(#decimal_receiver, #target) {
                    ::std::option::Option::Some(__df_m) => {
                        if __df_m.unsigned_abs() >= 10u128.pow(#precision) {
                            return ::std::result::Result::Err(
                                #pp::polars_err!(ComputeError:
                                    "df-derive: decimal mantissa {} exceeds declared precision {} for Decimal({}, {})",
                                    __df_m,
                                    #precision,
                                    #precision,
                                    #scale,
                                )
                            );
                        }
                        __df_m
                    }
                    ::std::option::Option::None => return ::std::result::Result::Err(
                        #pp::polars_err!(ComputeError:
                            "df-derive: decimal mantissa rescale to scale {} failed (overflow or precision loss)",
                            #target
                        )
                    ),
                }
            }}
        }
    }
}

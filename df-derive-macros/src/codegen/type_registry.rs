use crate::ir::{
    DateTimeUnit, DurationSource, NumericKind, PrimitiveLeaf, StorageNumericKind, WrapperShape,
};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use super::external_paths::ExternalPaths;

pub(super) struct NumericInfo {
    pub native: TokenStream,
    pub dtype: TokenStream,
    pub chunked: TokenStream,
    pub widen_from: Option<TokenStream>,
}

pub(super) fn numeric_info_for(kind: NumericKind, paths: &ExternalPaths) -> NumericInfo {
    let pp = paths.prelude();
    let info = |native: TokenStream, variant: &str, widen_from: Option<TokenStream>| {
        let chunked_ident = format_ident!("{}Chunked", variant);
        let dtype_ident = format_ident!("{}", variant);
        NumericInfo {
            native,
            dtype: quote! { #pp::DataType::#dtype_ident },
            chunked: quote! { #pp::#chunked_ident },
            widen_from,
        }
    };
    match kind.storage_kind() {
        StorageNumericKind::I8 => info(quote! { i8 }, "Int8", None),
        StorageNumericKind::I16 => info(quote! { i16 }, "Int16", None),
        StorageNumericKind::I32 => info(quote! { i32 }, "Int32", None),
        StorageNumericKind::I64 => info(quote! { i64 }, "Int64", None),
        StorageNumericKind::I128 => info(quote! { i128 }, "Int128", None),
        StorageNumericKind::U8 => info(quote! { u8 }, "UInt8", None),
        StorageNumericKind::U16 => info(quote! { u16 }, "UInt16", None),
        StorageNumericKind::U32 => info(quote! { u32 }, "UInt32", None),
        StorageNumericKind::U64 => info(quote! { u64 }, "UInt64", None),
        StorageNumericKind::U128 => info(quote! { u128 }, "UInt128", None),
        StorageNumericKind::F32 => info(quote! { f32 }, "Float32", None),
        StorageNumericKind::F64 => info(quote! { f64 }, "Float64", None),
        StorageNumericKind::ISize => info(quote! { i64 }, "Int64", Some(quote! { isize })),
        StorageNumericKind::USize => info(quote! { u64 }, "UInt64", Some(quote! { usize })),
    }
}

pub(super) fn numeric_stored_value(
    kind: NumericKind,
    source_value: TokenStream,
    native: &TokenStream,
) -> TokenStream {
    let value = if kind.is_nonzero() {
        quote! { (#source_value).get() }
    } else {
        source_value
    };
    if kind.is_widened() {
        quote! { (#value as #native) }
    } else {
        value
    }
}

pub(super) fn time_unit_tokens(unit: DateTimeUnit, paths: &ExternalPaths) -> TokenStream {
    let pp = paths.prelude();
    match unit {
        DateTimeUnit::Milliseconds => quote! { #pp::TimeUnit::Milliseconds },
        DateTimeUnit::Microseconds => quote! { #pp::TimeUnit::Microseconds },
        DateTimeUnit::Nanoseconds => quote! { #pp::TimeUnit::Nanoseconds },
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::codegen) enum LogicalPrimitive {
    Numeric(NumericKind),
    String,
    Bool,
    Binary,
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

impl LogicalPrimitive {
    pub(in crate::codegen) fn dtype(self, paths: &ExternalPaths) -> TokenStream {
        let pp = paths.prelude();
        let dt = quote! { #pp::DataType };
        match self {
            Self::Numeric(kind) => numeric_info_for(kind, paths).dtype,
            Self::String => quote! { #dt::String },
            Self::Bool => quote! { #dt::Boolean },
            Self::Binary => quote! { #dt::Binary },
            Self::DateTime(unit) | Self::NaiveDateTime(unit) => {
                let unit = time_unit_tokens(unit, paths);
                quote! { #dt::Datetime(#unit, ::std::option::Option::None) }
            }
            Self::NaiveDate => quote! { #dt::Date },
            Self::NaiveTime => quote! { #dt::Time },
            Self::Duration { unit, source } => {
                let _ = source;
                let unit = time_unit_tokens(unit, paths);
                quote! { #dt::Duration(#unit) }
            }
            Self::Decimal { precision, scale } => {
                let p = precision as usize;
                let s = scale as usize;
                quote! { #dt::Decimal(#p, #s) }
            }
        }
    }
}

impl PrimitiveLeaf<'_> {
    pub(super) const fn logical(&self) -> LogicalPrimitive {
        match *self {
            Self::Numeric(kind) => LogicalPrimitive::Numeric(kind),
            Self::String | Self::AsString | Self::AsStr(_) => LogicalPrimitive::String,
            Self::Bool => LogicalPrimitive::Bool,
            Self::Binary => LogicalPrimitive::Binary,
            Self::DateTime(unit) => LogicalPrimitive::DateTime(unit),
            Self::NaiveDateTime(unit) => LogicalPrimitive::NaiveDateTime(unit),
            Self::NaiveDate => LogicalPrimitive::NaiveDate,
            Self::NaiveTime => LogicalPrimitive::NaiveTime,
            Self::Duration { unit, source } => LogicalPrimitive::Duration { unit, source },
            Self::Decimal { precision, scale } => LogicalPrimitive::Decimal { precision, scale },
        }
    }

    pub(super) fn dtype(&self, paths: &ExternalPaths) -> TokenStream {
        self.logical().dtype(paths)
    }
}

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
            Self::Duration { unit, source } => LogicalPrimitive::Duration { unit, source },
            Self::Decimal { precision, scale } => LogicalPrimitive::Decimal { precision, scale },
        }
    }

    pub(in crate::codegen) fn dtype(self, paths: &ExternalPaths) -> TokenStream {
        self.logical().dtype(paths)
    }
}

pub fn full_dtype(
    leaf: PrimitiveLeaf<'_>,
    wrapper: &WrapperShape,
    paths: &ExternalPaths,
) -> TokenStream {
    let pp = paths.prelude();
    let elem_dtype = leaf.dtype(paths);
    super::external_paths::wrap_list_layers_compile_time(pp, elem_dtype, wrapper.vec_depth())
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
pub fn map_primitive_expr(
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
            let chrono = super::external_paths::chrono_root();
            quote! {
                {
                    use #chrono::Datelike as _;
                    (#var).num_days_from_ce() - 719_163
                }
            }
        }
        ScalarTransform::NaiveTime => {
            let chrono = super::external_paths::chrono_root();
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
                    // `num_milliseconds()` is infallible (returns i64
                    // directly) — no `?` needed.
                    DateTimeUnit::Milliseconds => quote! { (#var).num_milliseconds() },
                },
            }
        }
        ScalarTransform::Decimal { precision, scale } => {
            // Dispatch the rescale through the user-controlled
            // `Decimal128Encode` trait so different decimal backends
            // (`rust_decimal::Decimal`, `bigdecimal::BigDecimal`, …) can
            // own the conversion. Each implementer must use round-half-
            // to-even (banker's rounding) on scale-down so observable
            // bytes match polars's `str_to_dec128`. A `None` return
            // surfaces as a polars `ComputeError`.
            //
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

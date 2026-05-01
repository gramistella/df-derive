use crate::ir::{BaseType, DateTimeUnit, PrimitiveTransform, Wrapper, has_vec, vec_count};
use crate::type_analysis::{
    DEFAULT_DATETIME_UNIT, DEFAULT_DECIMAL_PRECISION, DEFAULT_DECIMAL_SCALE,
};
use proc_macro2::TokenStream;
use quote::quote;

/// Pull the chosen `Datetime` time unit out of a `DateTimeToInt(_)` transform.
/// Falls back to the crate default for unrelated transforms — relevant code
/// paths only consult this when the base type is `DateTime<Utc>`, so the
/// fallback is just defensive.
const fn datetime_unit(transform: Option<&PrimitiveTransform>) -> DateTimeUnit {
    match transform {
        Some(PrimitiveTransform::DateTimeToInt(unit)) => *unit,
        _ => DEFAULT_DATETIME_UNIT,
    }
}

/// Tokens for `polars::prelude::TimeUnit::<variant>`.
fn time_unit_tokens(unit: DateTimeUnit) -> TokenStream {
    let pp = super::polars_paths::prelude();
    match unit {
        DateTimeUnit::Milliseconds => quote! { #pp::TimeUnit::Milliseconds },
        DateTimeUnit::Microseconds => quote! { #pp::TimeUnit::Microseconds },
        DateTimeUnit::Nanoseconds => quote! { #pp::TimeUnit::Nanoseconds },
    }
}

/// Pull the chosen `Decimal(precision, scale)` out of a `DecimalToInt128 {..}`
/// transform. Falls back to the crate defaults for unrelated transforms — same
/// defensive rationale as `datetime_unit`.
const fn decimal_precision_scale(transform: Option<&PrimitiveTransform>) -> (u8, u8) {
    match transform {
        Some(PrimitiveTransform::DecimalToInt128 { precision, scale }) => (*precision, *scale),
        _ => (DEFAULT_DECIMAL_PRECISION, DEFAULT_DECIMAL_SCALE),
    }
}

pub struct TypeMapping {
    pub rust_element_type: TokenStream,
    pub element_dtype: TokenStream,
    pub full_dtype: TokenStream,
}

pub fn compute_mapping(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> TypeMapping {
    let (rust_elem, elem_dtype) = base_and_transform_to_rust_and_dtype(base, transform);
    let full_dtype = wrap_dtype(&elem_dtype, wrappers);
    TypeMapping {
        rust_element_type: rust_elem,
        element_dtype: elem_dtype,
        full_dtype,
    }
}

fn base_and_transform_to_rust_and_dtype(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
) -> (TokenStream, TokenStream) {
    let pp = super::polars_paths::prelude();
    let dt = quote! { #pp::DataType };

    // Attribute stringification (`as_string`) and borrowing (`as_str`) both
    // materialize as a String dtype. The borrowing path emits `Vec<&str>`
    // buffers directly and bypasses this fallback element type, but keeping
    // them aligned means a stray code path that doesn't yet handle `AsStr`
    // degrades to allocating, not panicking.
    if transform
        .is_some_and(|t| matches!(*t, PrimitiveTransform::ToString | PrimitiveTransform::AsStr))
    {
        return (quote! { ::std::string::String }, quote! { #dt::String });
    }

    match base {
        BaseType::String => (quote! { ::std::string::String }, quote! { #dt::String }),
        BaseType::F64 => (quote! { f64 }, quote! { #dt::Float64 }),
        BaseType::F32 => (quote! { f32 }, quote! { #dt::Float32 }),
        BaseType::I8 => (quote! { i8 }, quote! { #dt::Int8 }),
        BaseType::U8 => (quote! { u8 }, quote! { #dt::UInt8 }),
        BaseType::I16 => (quote! { i16 }, quote! { #dt::Int16 }),
        BaseType::U16 => (quote! { u16 }, quote! { #dt::UInt16 }),
        BaseType::I64 | BaseType::ISize => (quote! { i64 }, quote! { #dt::Int64 }),
        BaseType::U64 | BaseType::USize => (quote! { u64 }, quote! { #dt::UInt64 }),
        BaseType::U32 => (quote! { u32 }, quote! { #dt::UInt32 }),
        BaseType::I32 => (quote! { i32 }, quote! { #dt::Int32 }),
        BaseType::Bool => (quote! { bool }, quote! { #dt::Boolean }),
        BaseType::DateTimeUtc => {
            // we materialize as i64 then cast to Datetime dtype later
            let unit = time_unit_tokens(datetime_unit(transform));
            (
                quote! { i64 },
                quote! { #dt::Datetime(#unit, ::std::option::Option::None) },
            )
        }
        BaseType::Decimal => {
            // We materialize as raw `i128` mantissa values (rescaled to the
            // schema scale by `map_primitive_expr`), so the column finisher
            // can build an `Int128Chunked` and call `into_decimal_unchecked`
            // — no per-row string allocation or parse.
            let (precision, scale) = decimal_precision_scale(transform);
            let p = precision as usize;
            let s = scale as usize;
            (quote! { i128 }, quote! { #dt::Decimal(#p, #s) })
        }
        BaseType::Struct(..) | BaseType::Generic(_) => (quote! { () }, quote! { #dt::Null }),
    }
}

fn wrap_dtype(element_dtype: &TokenStream, wrappers: &[Wrapper]) -> TokenStream {
    if has_vec(wrappers) {
        let pp = super::polars_paths::prelude();
        quote! { #pp::DataType::List(::std::boxed::Box::new(#element_dtype)) }
    } else {
        quote! { #element_dtype }
    }
}

/// Dtype of one element of the *outermost* list in `wrappers`, used to
/// construct the per-field `ListBuilder` for nested-Vec shapes. The inner
/// Series fed to that builder has this dtype:
///
/// - `Vec<T>` → element dtype (the list contains `T` directly).
/// - `Vec<Vec<T>>` → `List<element>` (each element is itself a list of `T`).
/// - `Vec<Vec<Vec<T>>>` → `List<List<element>>`, etc.
/// - `Option<Vec<T>>` / `Vec<Option<T>>` → element dtype (`Option` doesn't
///   add a list layer; nullability is carried by the values, not a wrapper).
///
/// The macro's `schema()` reporting wraps `element` exactly once for any
/// `Vec`-containing field (a known limitation), so the outer Series's
/// runtime dtype can be deeper than the reported schema dtype. The list
/// builder needs the runtime dtype of its appended Series, not the schema's
/// flattened version, or strict-typed builders like
/// `ListPrimitiveChunkedBuilder` reject the append with a `SchemaMismatch`.
pub fn outer_list_inner_dtype(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> TokenStream {
    let (_, element_dtype) = base_and_transform_to_rust_and_dtype(base, transform);
    let extra_layers = vec_count(wrappers).saturating_sub(1);
    let mut dt = element_dtype;
    if extra_layers > 0 {
        let pp = super::polars_paths::prelude();
        for _ in 0..extra_layers {
            dt = quote! { #pp::DataType::List(::std::boxed::Box::new(#dt)) };
        }
    }
    dt
}

pub fn needs_cast(transform: Option<&PrimitiveTransform>) -> bool {
    transform.is_some_and(|t| match t {
        PrimitiveTransform::DateTimeToInt(_) | PrimitiveTransform::DecimalToInt128 { .. } => true,
        PrimitiveTransform::ToString | PrimitiveTransform::AsStr => false,
    })
}

/// True for transforms whose value-mapping expression returns via `?` (i.e.
/// the expression has type `PolarsResult<_>` and short-circuits on error).
/// Two transforms are fallible:
///
/// - `DateTime<Utc> → i64` at nanosecond precision: `timestamp_nanos_opt`
///   returns `None` for dates outside approximately [1677, 2262].
/// - `Decimal → i128` rescale on scale-up: an i128 mantissa multiplied by
///   `10^diff` for `diff` up to 38 can overflow for sufficiently large
///   inputs. We surface the overflow as a `PolarsResult` error rather than
///   panicking, matching the behavior of the historical `to_string + parse`
///   path which raised a polars `ComputeError` on overflow.
///
/// Callers that splice the mapped expression into a closure (e.g.
/// `.map(|e| { #mapped }).collect()`) need to switch to a `try_collect`-style
/// pattern when this returns true.
pub const fn is_fallible_conversion(transform: Option<&PrimitiveTransform>) -> bool {
    matches!(
        transform,
        Some(
            PrimitiveTransform::DateTimeToInt(DateTimeUnit::Nanoseconds)
                | PrimitiveTransform::DecimalToInt128 { .. }
        )
    )
}

/// Build the `AnyValue::<Variant>(…)` constructor expression for one already-
/// mapped primitive value (the result of `map_primitive_expr`). Used by the
/// `to_inner_values(&self)` per-row path to materialize each leaf without
/// going through a 1-element Series + `get(0)?.into_static()` round-trip —
/// at scale that's ~N-fields × N-rows throwaway Series allocations.
pub fn anyvalue_static_expr(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    mapped_var: &TokenStream,
) -> TokenStream {
    let pp = super::polars_paths::prelude();
    if let Some(t) = transform {
        match t {
            PrimitiveTransform::DateTimeToInt(unit) => {
                let unit_ts = time_unit_tokens(*unit);
                return quote! {
                    #pp::AnyValue::Datetime(#mapped_var, #unit_ts, ::std::option::Option::None)
                };
            }
            PrimitiveTransform::DecimalToInt128 { precision, scale } => {
                let p = *precision as usize;
                let s = *scale as usize;
                return quote! { #pp::AnyValue::Decimal(#mapped_var, #p, #s) };
            }
            PrimitiveTransform::ToString | PrimitiveTransform::AsStr => {
                return quote! { #pp::AnyValue::StringOwned((#mapped_var).into()) };
            }
        }
    }
    match base {
        BaseType::F64 => quote! { #pp::AnyValue::Float64(#mapped_var) },
        BaseType::F32 => quote! { #pp::AnyValue::Float32(#mapped_var) },
        BaseType::I64 | BaseType::ISize => quote! { #pp::AnyValue::Int64(#mapped_var) },
        BaseType::U64 | BaseType::USize => quote! { #pp::AnyValue::UInt64(#mapped_var) },
        BaseType::I32 => quote! { #pp::AnyValue::Int32(#mapped_var) },
        BaseType::U32 => quote! { #pp::AnyValue::UInt32(#mapped_var) },
        BaseType::I16 => quote! { #pp::AnyValue::Int16(#mapped_var) },
        BaseType::U16 => quote! { #pp::AnyValue::UInt16(#mapped_var) },
        BaseType::I8 => quote! { #pp::AnyValue::Int8(#mapped_var) },
        BaseType::U8 => quote! { #pp::AnyValue::UInt8(#mapped_var) },
        BaseType::Bool => quote! { #pp::AnyValue::Boolean(#mapped_var) },
        BaseType::String => quote! { #pp::AnyValue::StringOwned((#mapped_var).into()) },
        BaseType::DateTimeUtc | BaseType::Decimal => unreachable!(
            "df-derive: DateTime/Decimal base reached anyvalue_static_expr without its mandatory transform (codegen invariant)"
        ),
        BaseType::Struct(..) | BaseType::Generic(_) => unreachable!(
            "df-derive: nested struct/generic leaves do not flow through the primitive AnyValue helper"
        ),
    }
}

pub fn map_primitive_expr(
    var: &TokenStream,
    transform: Option<&PrimitiveTransform>,
) -> TokenStream {
    transform.map_or_else(
        || quote! { (#var).clone() },
        |t| match t {
            PrimitiveTransform::DateTimeToInt(unit) => match unit {
                DateTimeUnit::Milliseconds => quote! { (#var).timestamp_millis() },
                DateTimeUnit::Microseconds => quote! { (#var).timestamp_micros() },
                DateTimeUnit::Nanoseconds => {
                    let pp = super::polars_paths::prelude();
                    quote! {
                        (#var).timestamp_nanos_opt().ok_or_else(|| #pp::polars_err!(
                            ComputeError: "df-derive: DateTime<Utc> value is out of range for nanosecond timestamps (chrono supports approximately 1677..2262)"
                        ))?
                    }
                }
            },
            PrimitiveTransform::ToString => {
                quote! { (#var).to_string() }
            }
            PrimitiveTransform::DecimalToInt128 { scale, .. } => {
                // Rescale the rust_decimal mantissa to the schema scale,
                // matching the historical `to_string + parse` round-trip
                // through polars's `str_to_dec128`. That path uses
                // round-half-to-even (banker's rounding) on scale-down, so
                // we replicate it here directly on the i128 mantissa rather
                // than going through `rust_decimal::Decimal::rescale`, which
                // uses half-away-from-zero. Scale-up can overflow for large
                // inputs, so this expression is treated as fallible by
                // `is_fallible_conversion`: the overflow arm `return`s an
                // `Err(...)` from the innermost enclosing function (a
                // `try_collect` closure for `Vec<Decimal>` shapes, or
                // `columnar_from_refs` / `to_inner_values` for the leaf
                // populator paths).
                //
                // Bounds: `rust_decimal::Decimal::scale()` is capped at
                // `MAX_SCALE = 28`, polars caps decimal scale at 38, so the
                // scale-up `diff` is at most 38 and the scale-down `diff`
                // is at most 28. `10i128.pow(diff)` therefore fits in i128
                // for either direction (max `10^38 < 2^127`).
                let target = u32::from(*scale);
                let pp = super::polars_paths::prelude();
                quote! {{
                    let __df_d = (#var);
                    let __df_d_scale = __df_d.scale();
                    let __df_d_m: i128 = __df_d.mantissa();
                    let __df_target: u32 = #target;
                    if __df_d_scale == __df_target {
                        __df_d_m
                    } else if __df_d_scale < __df_target {
                        let __df_diff = __df_target - __df_d_scale;
                        let __df_pow = 10i128.pow(__df_diff);
                        match __df_d_m.checked_mul(__df_pow) {
                            ::std::option::Option::Some(__df_v) => __df_v,
                            ::std::option::Option::None => return ::std::result::Result::Err(#pp::polars_err!(
                                ComputeError: "df-derive: decimal mantissa overflow rescaling from scale {} to scale {}",
                                __df_d_scale, __df_target
                            )),
                        }
                    } else {
                        // Scale-down with round-half-to-even on the unsigned
                        // magnitude, then re-apply sign — matches polars's
                        // `div_128_pow10` semantics.
                        let __df_diff = __df_d_scale - __df_target;
                        let __df_pow = 10i128.pow(__df_diff) as u128;
                        let __df_neg = __df_d_m < 0;
                        let __df_abs = __df_d_m.unsigned_abs();
                        let __df_q = (__df_abs / __df_pow) as i128;
                        let __df_r = __df_abs % __df_pow;
                        let __df_half = __df_pow / 2;
                        let __df_rounded = if __df_r > __df_half {
                            __df_q + 1
                        } else if __df_r < __df_half {
                            __df_q
                        } else {
                            __df_q + (__df_q & 1)
                        };
                        if __df_neg { -__df_rounded } else { __df_rounded }
                    }
                }}
            }
            PrimitiveTransform::AsStr => {
                // Allocating fallback for codegen sites that can't use a
                // `Vec<&str>` columnar buffer — in practice this is only
                // hit at the leaf of stacked-Option shapes (e.g.
                // `Option<Option<T>>`), where the buffer type
                // `Vec<Option<String>>` flattens both layers anyway. All
                // `Vec<…>` shapes route around this arm via
                // `generate_inner_series_from_vec`, which builds a
                // `Vec<&str>` directly. Emitting valid Rust here keeps the
                // per-field `AsRef<str>` const-fn assert as the canonical
                // user-visible error rather than a proc-macro internal
                // panic.
                quote! { <_ as ::core::convert::AsRef<str>>::as_ref(&(#var)).to_string() }
            }
        },
    )
}

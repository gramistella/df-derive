use crate::ir::{BaseType, DateTimeUnit, PrimitiveTransform, Wrapper, vec_count};
use crate::type_analysis::{
    DEFAULT_DATETIME_UNIT, DEFAULT_DECIMAL_PRECISION, DEFAULT_DECIMAL_SCALE,
};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

/// Token bundle for one numeric-shaped primitive base. Every fast-path
/// emitter consumes some subset of these — collected here so the 12-arm
/// match per metadata kind doesn't have to be repeated at every call site.
///
/// `widen_from` carries the SOURCE Rust type (`isize`/`usize`) when widening
/// is needed; `native` is the storage type (`i64`/`u64` in that case). For
/// fixed-width bases (`i8/i16/.../f64`) `widen_from` is `None` and `native`
/// matches the source type. Polars supports only fixed-width 8/16/32/64
/// lanes, so the encoder widens `ISize`/`USize` reads to `i64`/`u64` at the
/// leaf push site and stores into a `Vec<i64>` / `Vec<u64>` whose downstream
/// chunked-array build matches the schema dtype directly.
pub(super) struct NumericInfo {
    /// Native Rust storage type token, e.g. `i8`, `f64`. For `ISize`/`USize`
    /// this is the widened storage type (`i64`/`u64`); for fixed-width bases
    /// it equals the source type.
    pub native: TokenStream,
    /// `#pp::DataType::<Variant>` for the leaf, e.g. `#pp::DataType::Int8`.
    pub dtype: TokenStream,
    /// `#pp::<Variant>Chunked` alias, e.g. `#pp::Int8Chunked`.
    pub chunked: TokenStream,
    /// `Some(source_type_tokens)` when the leaf push site must widen reads
    /// (`isize`/`usize` → `i64`/`u64` via an `as` cast). `None` for
    /// fixed-width bases where storage matches the source.
    pub widen_from: Option<TokenStream>,
}

/// Returns `Some(NumericInfo)` for every numeric-shaped base
/// (`i8/i16/i32/i64/u8/u16/u32/u64/f32/f64/isize/usize`). `None` for
/// everything else (`Bool`, `String`, `DateTime`, `Decimal`, `Struct`,
/// `Generic`). `ISize`/`USize` carry `widen_from = Some(isize|usize)` and
/// `native = i64|u64`; the leaf push site reads `(*v) as i64` / `(*v) as
/// u64` to match the storage type.
pub(super) fn numeric_info(base: &BaseType) -> Option<NumericInfo> {
    let pp = super::polars_paths::prelude();
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
    Some(match base {
        BaseType::I8 => info(quote! { i8 }, "Int8", None),
        BaseType::I16 => info(quote! { i16 }, "Int16", None),
        BaseType::I32 => info(quote! { i32 }, "Int32", None),
        BaseType::I64 => info(quote! { i64 }, "Int64", None),
        BaseType::U8 => info(quote! { u8 }, "UInt8", None),
        BaseType::U16 => info(quote! { u16 }, "UInt16", None),
        BaseType::U32 => info(quote! { u32 }, "UInt32", None),
        BaseType::U64 => info(quote! { u64 }, "UInt64", None),
        BaseType::F32 => info(quote! { f32 }, "Float32", None),
        BaseType::F64 => info(quote! { f64 }, "Float64", None),
        BaseType::ISize => info(quote! { i64 }, "Int64", Some(quote! { isize })),
        BaseType::USize => info(quote! { u64 }, "UInt64", Some(quote! { usize })),
        BaseType::Bool
        | BaseType::String
        | BaseType::DateTimeUtc
        | BaseType::Decimal
        | BaseType::Struct(..)
        | BaseType::Generic(_) => return None,
    })
}

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

pub fn compute_full_dtype(
    base: &BaseType,
    transform: Option<&PrimitiveTransform>,
    wrappers: &[Wrapper],
) -> TokenStream {
    let (_, elem_dtype) = base_and_transform_to_rust_and_dtype(base, transform);
    wrap_dtype(&elem_dtype, wrappers)
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
    let layers = vec_count(wrappers);
    if layers == 0 {
        return quote! { #element_dtype };
    }
    let pp = super::polars_paths::prelude();
    let mut dt = element_dtype.clone();
    for _ in 0..layers {
        dt = quote! { #pp::DataType::List(::std::boxed::Box::new(#dt)) };
    }
    dt
}

pub fn map_primitive_expr(
    var: &TokenStream,
    transform: Option<&PrimitiveTransform>,
    decimal128_encode_trait: &TokenStream,
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
                // Dispatch the rescale through the user-controlled
                // `Decimal128Encode` trait so different decimal backends
                // (`rust_decimal::Decimal`, `bigdecimal::BigDecimal`, …) can
                // own the conversion. Each implementer must use round-half-
                // to-even (banker's rounding) on scale-down so observable
                // bytes match polars's `str_to_dec128`. A `None` return
                // surfaces as a polars `ComputeError`, matching the
                // historical scale-up overflow path.
                //
                // The trait is imported anonymously (`use ... as _;`) inside
                // the emitted block so we can call the method via dot syntax
                // (`receiver.try_to_i128_mantissa(...)`). Dot syntax triggers
                // method resolution, which auto-derefs through `&Decimal` /
                // `&&Decimal` etc. — necessary because callers pass receivers
                // of both shapes (place expressions like `self.field` of
                // type `Decimal`, and iterator items of type `&Decimal`).
                // Function-call form on a trait path does NOT auto-deref,
                // which would force the codegen to know the bare field type.
                // The anonymous-import idiom doesn't introduce any name into
                // user scope. The implementer's body is small enough that
                // the compiler inlines through the trait dispatch (confirmed
                // by bench 13 not regressing).
                let target = u32::from(*scale);
                let pp = super::polars_paths::prelude();
                quote! {{
                    use #decimal128_encode_trait as _;
                    match (#var).try_to_i128_mantissa(#target) {
                        ::std::option::Option::Some(__df_m) => __df_m,
                        ::std::option::Option::None => return ::std::result::Result::Err(
                            #pp::polars_err!(ComputeError:
                                "df-derive: decimal mantissa rescale to scale {} failed (overflow or precision loss)",
                                #target
                            )
                        ),
                    }
                }}
            }
            PrimitiveTransform::AsStr => {
                // Allocating fallback for codegen sites that can't use a
                // `Vec<&str>` columnar buffer. The encoder IR routes every
                // `as_str` shape through borrowing buffers built directly by
                // the leaf encoders, so this arm is currently unreachable on
                // parser-validated input. Emitting valid Rust here keeps the
                // per-field `AsRef<str>` const-fn assert as the canonical
                // user-visible error rather than a proc-macro internal
                // panic, should a future caller route through this path.
                quote! { <_ as ::core::convert::AsRef<str>>::as_ref(&(#var)).to_string() }
            }
        },
    )
}

use crate::ir::{DateTimeUnit, LeafSpec, NumericKind, WrapperShape};
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

/// Token bundle for the chosen numeric kind. Every numeric-shaped primitive
/// goes through this. `ISize`/`USize` carry `widen_from = Some(isize|usize)`
/// and `native = i64|u64`; the leaf push site reads `(*v) as i64` / `(*v) as
/// u64` to match the storage type.
///
/// Total over `NumericKind` (no `Option` return) — the parser has already
/// classified the leaf as numeric, so the encoder consumer knows the result
/// is non-empty. The 12-arm match here is the one place that translates the
/// parser-tagged kind into the polars-prelude token shape.
pub(super) fn numeric_info_for(kind: NumericKind) -> NumericInfo {
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
    match kind {
        NumericKind::I8 => info(quote! { i8 }, "Int8", None),
        NumericKind::I16 => info(quote! { i16 }, "Int16", None),
        NumericKind::I32 => info(quote! { i32 }, "Int32", None),
        NumericKind::I64 => info(quote! { i64 }, "Int64", None),
        NumericKind::U8 => info(quote! { u8 }, "UInt8", None),
        NumericKind::U16 => info(quote! { u16 }, "UInt16", None),
        NumericKind::U32 => info(quote! { u32 }, "UInt32", None),
        NumericKind::U64 => info(quote! { u64 }, "UInt64", None),
        NumericKind::F32 => info(quote! { f32 }, "Float32", None),
        NumericKind::F64 => info(quote! { f64 }, "Float64", None),
        NumericKind::ISize => info(quote! { i64 }, "Int64", Some(quote! { isize })),
        NumericKind::USize => info(quote! { u64 }, "UInt64", Some(quote! { usize })),
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

impl LeafSpec {
    /// Compile-time element-level dtype for this leaf, BEFORE the wrapper
    /// stack adds `List<>` envelopes. The encoder also calls this directly
    /// when it needs the leaf's logical dtype for the cast / typed-null path.
    ///
    /// `AsStr` shares the `String` arm because attribute stringification
    /// (`as_string`) and borrowing (`as_str`) both materialize as `String`.
    /// The borrowing path emits `Vec<&str>` buffers directly and bypasses
    /// this fallback element type, but keeping them aligned means a stray
    /// code path that doesn't yet handle `AsStr` degrades to allocating,
    /// not panicking.
    pub(super) fn dtype(&self) -> TokenStream {
        let pp = super::polars_paths::prelude();
        let dt = quote! { #pp::DataType };
        match self {
            Self::Numeric(kind) => numeric_info_for(*kind).dtype,
            Self::String | Self::AsString | Self::AsStr(_) => quote! { #dt::String },
            Self::Bool => quote! { #dt::Boolean },
            Self::DateTime(unit) => {
                let unit = time_unit_tokens(*unit);
                quote! { #dt::Datetime(#unit, ::std::option::Option::None) }
            }
            Self::Decimal { precision, scale } => {
                let p = *precision as usize;
                let s = *scale as usize;
                quote! { #dt::Decimal(#p, #s) }
            }
            Self::Struct(..) | Self::Generic(_) => quote! { #dt::Null },
        }
    }
}

/// Full-field dtype: leaf dtype wrapped in `List<>` envelopes for each
/// `Vec` layer in the wrapper stack. Consumers that want the leaf-only
/// dtype (e.g. the encoder's per-leaf logical-dtype payload) call
/// [`LeafSpec::dtype`] directly.
pub fn full_dtype(leaf: &LeafSpec, wrapper: &WrapperShape) -> TokenStream {
    let pp = super::polars_paths::prelude();
    let elem_dtype = leaf.dtype();
    super::polars_paths::wrap_list_layers_compile_time(&pp, elem_dtype, wrapper.vec_depth())
}

/// Map a per-row primitive value through any leaf-injected transform
/// (`DateTime` → epoch i64, `Decimal` → i128 mantissa, `AsString` → owned
/// `String`, `AsStr` → `&str` borrow). The bare `LeafSpec::*` arms
/// (`Numeric` / `String` / `Bool` / `Struct` / `Generic`) clone the value
/// directly — no transform applies. Returns the mapped per-row expression.
pub fn map_primitive_expr(
    var: &TokenStream,
    leaf: &LeafSpec,
    decimal128_encode_trait: &TokenStream,
) -> TokenStream {
    match leaf {
        LeafSpec::DateTime(unit) => match unit {
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
        LeafSpec::AsString => {
            quote! { (#var).to_string() }
        }
        LeafSpec::Decimal { scale, .. } => {
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
        LeafSpec::AsStr(_) => {
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
        LeafSpec::Numeric(_)
        | LeafSpec::String
        | LeafSpec::Bool
        | LeafSpec::Struct(..)
        | LeafSpec::Generic(_) => quote! { (#var).clone() },
    }
}

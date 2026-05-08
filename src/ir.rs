use syn::Ident;

/// Top-level IR for a struct targeted by the derive.
#[derive(Clone)]
pub struct StructIR {
    /// The identifier of the struct being derived
    pub name: Ident,
    /// The generics declared on the struct (empty when no generics are used)
    pub generics: syn::Generics,
    /// The fields of the struct in declaration order
    pub fields: Vec<FieldIR>,
}

/// IR for a single field of a struct
#[derive(Clone)]
pub struct FieldIR {
    /// Field name as declared on the struct
    pub name: Ident,
    /// Field index for tuple structs (None for named fields)
    pub field_index: Option<usize>,
    /// Per-leaf semantic shape — encodes the (base, transform) pair the
    /// encoder consumes. The parser's legality matrix collapses every valid
    /// `(base, override)` combination into one of these variants; the
    /// encoder destructures without `unreachable!` arms.
    pub leaf_spec: LeafSpec,
    /// Per-wrapper semantic shape — `Option`-only stacks (`Leaf { ... }`) or
    /// any shape with at least one `Vec` (`Vec(VecLayers)`). Consecutive
    /// `Option`s are collapsed into counts (Polars folds them into a single
    /// validity bit; the count is preserved so the encoder can decide
    /// whether a multi-Option `as_ref().and_then(...)` collapse is needed).
    pub wrapper_shape: WrapperShape,
    /// The original `syn::Type` of the field. Preserved so codegen can splice
    /// it into trait-bound asserts (e.g. `T: AsRef<str>`) with the user's
    /// source span, putting compiler errors at the field declaration rather
    /// than deep in macro expansion.
    pub field_ty: syn::Type,
}

/// Datetime time unit chosen via `#[df_derive(time_unit = "ms"|"us"|"ns")]`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DateTimeUnit {
    Milliseconds,
    Microseconds,
    Nanoseconds,
}

/// Numeric primitive kind. Carries the static information the encoder needs
/// (variant tag for chunked-array / dtype dispatch, plus widening info for
/// `isize`/`usize`) without binding to any token-stream representation.
/// Polars only supports fixed-width 8/16/32/64 lanes, so the platform-sized
/// `ISize`/`USize` variants widen to `i64`/`u64` at the leaf push site.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NumericKind {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    /// Widens to `i64`. The leaf push site reads `(*v) as i64`.
    ISize,
    /// Widens to `u64`. The leaf push site reads `(*v) as u64`.
    USize,
}

impl NumericKind {
    /// True for `ISize`/`USize` (platform-sized integers widened at the
    /// codegen boundary).
    pub const fn is_widened(self) -> bool {
        matches!(self, Self::ISize | Self::USize)
    }
}

/// Bases that can be paired with the `as_str` borrow path: `String`,
/// concrete user-defined struct types, and generic type parameters. The
/// parser's legality matrix accepts only this set with `as_str`; every
/// other base is rejected at parse time.
#[derive(Clone)]
pub enum StringyBase {
    String,
    /// Concrete user-defined struct, with optional angle-bracketed generic
    /// arguments at the field's use site (e.g. `<M>` in `Vec<Foo<M>>`).
    Struct(Ident, Option<syn::AngleBracketedGenericArguments>),
    /// Generic type parameter declared on the enclosing struct.
    Generic(Ident),
}

impl StringyBase {
    /// True when the stringy base is bare `String`. The bare-`String` path
    /// has tighter token shapes (deref-coercion through `&String`) than the
    /// UFCS path used for `Struct`/`Generic`.
    pub const fn is_string(&self) -> bool {
        matches!(self, Self::String)
    }
}

/// Per-leaf semantic shape — the encoder's vocabulary for the unwrapped
/// element type after the parser has folded `(base, override)` through its
/// legality matrix. Each variant corresponds to exactly one leaf builder in
/// the encoder; misclassification at parse time is structurally impossible
/// because every parser-accepted `(base, override)` pair maps to exactly one
/// variant by construction.
#[derive(Clone)]
pub enum LeafSpec {
    /// Numeric primitive (`i8`/`i16`/`i32`/`i64`/`u8`/`u16`/`u32`/`u64`/
    /// `f32`/`f64`) and the platform-sized widened variants (`isize`/`usize`,
    /// widened to `i64`/`u64` at the codegen boundary).
    Numeric(NumericKind),
    /// Bare `String` field, no transform.
    String,
    /// Bare `bool` field, no transform.
    Bool,
    /// `chrono::DateTime<Utc>` paired with the parser-injected
    /// `DateTimeToInt(unit)` semantics. The chosen unit determines both the
    /// chrono call used to derive the i64 mantissa and the
    /// `DataType::Datetime(...)` schema dtype.
    DateTime(DateTimeUnit),
    /// `rust_decimal::Decimal` with the chosen `Decimal(precision, scale)`
    /// dtype. `1 <= precision <= 38`; `scale <= precision`.
    Decimal { precision: u8, scale: u8 },
    /// `#[df_derive(as_string)]` — convert any `Display` value to `String`
    /// at codegen time. Materializes as `DataType::String`.
    AsString,
    /// `#[df_derive(as_str)]` — borrow `&str` via `<T as AsRef<str>>::as_ref`
    /// for the duration of the columnar populator pass. The `StringyBase`
    /// distinguishes the bare-`String` deref-coercion path from the UFCS
    /// path used for non-`String` `AsRef<str>` types.
    AsStr(StringyBase),
    /// `#[df_derive(as_binary)]` — route a `Vec<u8>` shape through a Polars
    /// `Binary` column instead of the default `List(UInt8)`. The parser
    /// strips the innermost `Vec` from the wrapper stack and replaces the
    /// `Numeric(U8)` leaf with this variant; the encoder then materializes
    /// each row's bytes via `MutableBinaryViewArray::<[u8]>`.
    Binary,
    /// Concrete user-defined struct type, no stringy transform.
    Struct(Ident, Option<syn::AngleBracketedGenericArguments>),
    /// Generic type parameter declared on the enclosing struct.
    Generic(Ident),
}

impl LeafSpec {
    /// `Copy` test for the multi-Option per-row materializer. Numeric leaves
    /// (including `ISize`/`USize`) and `Bool` are `Copy`; `String`,
    /// `Binary`, `DateTime`, `Decimal`, `AsString`, and the `AsStr` borrow
    /// path are not. The `AsStr` path takes its own branch in the multi-
    /// Option wrapper before reaching this helper, so its `false` answer
    /// here is only consulted on dead arms.
    pub const fn is_copy(&self) -> bool {
        matches!(self, Self::Numeric(_) | Self::Bool)
    }
}

/// One `Vec` layer in a normalized wrapper stack. Outermost layer first.
/// `option_layers_above` is the count of consecutive `Option` wrappers
/// immediately above this `Vec`. Zero means no list-level validity. `>0`
/// means a `MutableBitmap` rides under this `LargeListArray` and the
/// per-row access expression must walk `option_layers_above` `Option`s
/// before entering the `Vec`. Polars only carries one validity bit per list
/// level, so all consecutive `Option`s collapse to one bit (`Some(None)`
/// and `None` are indistinguishable in the column).
#[derive(Clone, Copy, Debug)]
pub struct VecLayerSpec {
    pub option_layers_above: usize,
}

impl VecLayerSpec {
    pub const fn has_outer_validity(self) -> bool {
        self.option_layers_above > 0
    }
}

/// Normalized form of a `Vec`-bearing wrapper stack. `layers[0]` is the
/// outermost `Vec`. `inner_option_layers` is the count of consecutive
/// `Option` wrappers immediately surrounding the leaf (between the
/// innermost `Vec` and the leaf type). `>0` means a per-element validity
/// bit is stored at the leaf and the per-element access expression must
/// walk `inner_option_layers` `Option`s before reaching the leaf value.
#[derive(Clone, Debug)]
pub struct VecLayers {
    pub layers: Vec<VecLayerSpec>,
    pub inner_option_layers: usize,
}

impl VecLayers {
    pub const fn depth(&self) -> usize {
        self.layers.len()
    }

    pub fn any_outer_validity(&self) -> bool {
        self.layers.iter().any(|l| l.has_outer_validity())
    }

    pub const fn has_inner_option(&self) -> bool {
        self.inner_option_layers > 0
    }
}

/// Per-wrapper semantic shape — either a leaf-only stack (no `Vec`) with
/// `option_layers` counting consecutive `Option`s above the leaf, or one or
/// more `Vec` layers with optional `Option`s collapsed into list-level
/// validity counts at each layer.
///
/// The parser builds this directly from the raw wrapper sequence; the
/// encoder destructures without re-normalizing. Polars folds consecutive
/// `Option`s into a single validity bit, so `[Option, Option]` collapses to
/// `option_layers = 2` (with the encoder using `as_ref().and_then(...)` to
/// flatten to a single `Option<&T>` before pushing). The raw count is
/// preserved so the encoder can choose between a no-op direct match
/// (`option_layers == 1`) and the multi-Option collapse (`option_layers >= 2`).
#[derive(Clone)]
pub enum WrapperShape {
    /// No `Vec` wrappers. `option_layers` is the count of `Option`s above
    /// the leaf. `0` is a bare leaf; `1` is a single `Option<T>`; `>= 2` is
    /// `Option<...<Option<T>>>` collapsed into a single validity bit per
    /// Polars's representation.
    Leaf { option_layers: usize },
    /// One or more `Vec` wrappers, optionally with `Option`s collapsed into
    /// per-layer outer validity and inner-element validity.
    Vec(VecLayers),
}

impl WrapperShape {
    /// Number of `Vec<…>` layers in the wrapper stack.
    pub const fn vec_depth(&self) -> usize {
        match self {
            Self::Leaf { .. } => 0,
            Self::Vec(v) => v.depth(),
        }
    }
}

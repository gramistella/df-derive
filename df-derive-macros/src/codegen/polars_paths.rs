// Centralized polars path resolution for generated code.
//
// Every other codegen file produces token streams that reference items in
// `polars`. Funneling those references through this module gives us a single
// fix point if polars reshuffles its modules (`prelude::Foo` → `dtype::Foo`
// has happened across major versions), and lets `proc_macro_crate::crate_name`
// honour `[package]` renames in the downstream `Cargo.toml` so a user pinning
// polars under a different name still gets working generated code.

use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

/// Resolve a crate by name to a `::<resolved-name>` path token, or splice
/// the caller's `fallback` when the crate is not found or refers to the
/// expanding crate itself. Used by [`root`] and [`polars_arrow_root`] —
/// neither of those macros realistically expands inside the polars or
/// polars-arrow crate, and a missing direct dep is best surfaced as
/// `unresolved import ::<name>` at the call site rather than `unresolved
/// name <name>`. The `mod::resolve_default_dataframe_mod` site is bespoke
/// because its `Itself` arm maps to `crate::dataframe` (a substantive
/// path), not to the fallback — a different cascade shape that this
/// helper cannot model cleanly.
pub(super) fn resolve_or_fallback(name: &str, fallback: TokenStream) -> TokenStream {
    match crate_name(name) {
        Ok(FoundCrate::Name(resolved)) => {
            let ident = format_ident!("{}", resolved);
            quote! { ::#ident }
        }
        Ok(FoundCrate::Itself) | Err(_) => fallback,
    }
}

/// Token tree for the user-visible `polars` crate root.
///
/// `Itself` and `Err` collapse to `::polars` because the macro doesn't
/// expand inside the polars crate itself in any realistic scenario, and a
/// missing direct dep is better surfaced as `unresolved import ::polars`
/// than `unresolved name polars` — same eventual error, more specific span.
fn root() -> TokenStream {
    resolve_or_fallback("polars", quote! { ::polars })
}

/// `polars::prelude` — namespace for ~all polars items the macro emits.
pub fn prelude() -> TokenStream {
    let root = root();
    quote! { #root::prelude }
}

/// `polars::prelude::Int128Chunked` — used by the Decimal columnar finisher
/// to bypass the `Series::new(&Vec<i128>) + cast(Decimal)` round-trip and
/// build the `DecimalChunked` directly via `into_decimal_unchecked`.
pub fn int128_chunked() -> TokenStream {
    let pp = prelude();
    quote! { #pp::Int128Chunked }
}

/// Token tree for the user-visible `polars-arrow` crate root.
///
/// `polars-arrow` is a hard requirement for downstream crates that use
/// `#[derive(ToDataFrame)]`: the macro emits `OffsetsBuffer` and
/// `LargeListArray` paths to construct list arrays directly, which
/// achieves 7-10× speedups on `Vec<Struct>` columns. `polars` 0.53
/// already compiles `polars-arrow` transitively but doesn't re-export it
/// under any public path, so a user pinning `polars` must declare
/// `polars-arrow` as a direct dep too.
///
/// `Itself` and `Err` collapse to `::polars_arrow` for the same reason as
/// `root()` — the macro never expands inside `polars-arrow`, and a
/// missing direct dep is best surfaced as `unresolved import
/// ::polars_arrow` at the call site.
pub fn polars_arrow_root() -> TokenStream {
    resolve_or_fallback("polars-arrow", quote! { ::polars_arrow })
}

/// Wrap an inner `DataType` token expression in `layers` `List<>` envelopes
/// at compile time. Returns `inner` unchanged when `layers == 0`. Used by
/// every site that needs to project a leaf logical dtype to its
/// `Vec<…<Vec<leaf>>>` form, where the wrap count is statically known from
/// the wrapper stack: the type-registry full-dtype computation and the
/// encoder's final-assemble per-leaf logical dtype.
pub(super) fn wrap_list_layers_compile_time(
    pp: &TokenStream,
    inner: TokenStream,
    layers: usize,
) -> TokenStream {
    let mut dt = inner;
    for _ in 0..layers {
        dt = quote! { #pp::DataType::List(::std::boxed::Box::new(#dt)) };
    }
    dt
}

/// Same as [`wrap_list_layers_compile_time`] but visible to the encoder
/// submodule. Sub-encoders (the tuple emitter under `encoder/tuple.rs`)
/// need the same wrap behavior; bumping visibility avoids re-exporting the
/// helper across the codegen boundary.
pub(in crate::codegen) fn wrap_list_layers_compile_time_pub(
    pp: &TokenStream,
    inner: TokenStream,
    layers: usize,
) -> TokenStream {
    wrap_list_layers_compile_time(pp, inner, layers)
}

/// Emit a runtime `for _ in 0..layers` loop that wraps a runtime `DataType`
/// variable in `layers` `List<>` envelopes. Returns an empty token stream
/// when `layers == 0` so the caller does not emit `for _ in 0..0`, which
/// trips `clippy::reversed_empty_ranges` inside the user's expanded code.
/// Used by the nested schema/empty-frame helpers, where the wrap count is
/// compile-time known but the inner dtype comes from a runtime
/// `T::schema()?` iteration.
pub(super) fn wrap_list_layers_runtime(
    pp: &TokenStream,
    var: &syn::Ident,
    layers: usize,
) -> TokenStream {
    if layers == 0 {
        TokenStream::new()
    } else {
        quote! {
            for _ in 0..#layers {
                #var = #pp::DataType::List(
                    ::std::boxed::Box::new(#var),
                );
            }
        }
    }
}

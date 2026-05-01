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

/// Token tree for the user-visible `polars` crate root.
///
/// `Itself` and `Err` collapse to `::polars` because the macro doesn't
/// expand inside the polars crate itself in any realistic scenario, and a
/// missing direct dep is better surfaced as `unresolved import ::polars`
/// than `unresolved name polars` — same eventual error, more specific span.
fn root() -> TokenStream {
    match crate_name("polars") {
        Ok(FoundCrate::Name(name)) => {
            let ident = format_ident!("{}", name);
            quote! { ::#ident }
        }
        Ok(FoundCrate::Itself) | Err(_) => quote! { ::polars },
    }
}

/// `polars::prelude` — namespace for ~all polars items the macro emits.
pub fn prelude() -> TokenStream {
    let root = root();
    quote! { #root::prelude }
}

/// `polars::chunked_array::builder` — exists for `get_list_builder`, the
/// only path the macro emits outside `prelude`.
pub fn chunked_array_builder() -> TokenStream {
    let root = root();
    quote! { #root::chunked_array::builder }
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
    match crate_name("polars-arrow") {
        Ok(FoundCrate::Name(name)) => {
            let ident = format_ident!("{}", name);
            quote! { ::#ident }
        }
        Ok(FoundCrate::Itself) | Err(_) => quote! { ::polars_arrow },
    }
}

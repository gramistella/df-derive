//! Shared shape-walker for the `[Vec, ...]` push/scan body.
//!
//! Both the flat-vec path (`vec::build_vec_push_loops`) and the nested-struct
//! path (`nested::build_nested_scan_body`) walk the same `VecShape` recursively.
//! They differ only at the deepest layer (push to a typed buffer vs. push a
//! ref into a `Vec<&T>` plus optional position scatter) and in a few naming
//! choices (outer-Some bind name prefix, leaf-binding ident, the per-layer
//! offsets-push value at the innermost layer).
//!
//! [`ShapeScan`] captures all of those decision points behind one struct so
//! the recursion logic lives in exactly one place. The two callers wire up
//! their own `ScanLayerIdents` (mapping their per-layer identifier bundles
//! into the shared three-ident shape: offsets, validity, bind), choose the
//! `outer_some_prefix`, and supply the deepest-layer body via `leaf_body`.
//!
//! [`ShapePrecount`] is the parallel walker for the precount loop both paths
//! run before the scan to size the offsets/validity/leaf-storage buffers up
//! front. The two paths differ only in: (a) the leaf-counter increment (the
//! flat-vec path tallies `__df_derive_total_leaves`; the nested path tallies
//! the per-field `__df_derive_gen_total_<idx>`), (b) the outer-Some bind name
//! prefix, and (c) the per-layer counter idents. The precount walker emits
//! the full `let mut #total: usize = 0;` decl, the per-layer counter decls,
//! and the `for #it in items { ... }` ring; callers splice the result
//! verbatim. Critically, when `option_layers >= 2` on an outer-Vec layer the
//! walker calls [`collapse_options_to_ref`] so multi-Option-over-Vec shapes
//! don't try to `if let Some(...) = &Option<Option<Vec<...>>>` (which doesn't
//! type-check).
//!
//! Two related decl helpers ([`shape_offsets_decls`] and
//! [`shape_validity_decls`]) generalize the per-layer offsets-vec and
//! validity-bitmap allocations: layer 0 is sized by `items.len()` (or
//! `items.len() + 1` for offsets to account for the leading `0`); deeper
//! layers use a per-depth counter ident the caller supplies via a closure.
//! The flat-vec path passes `__df_derive_total_layer_{i-1}`; the nested path
//! passes `__df_derive_n_total_layer_{i-1}`.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use super::idents;
use super::{VecShape, collapse_options_to_ref};

/// Per-layer identifiers the shared scan walker needs. Both `VecLayerIdents`
/// (flat-vec path) and `NestedLayerIdents` (nested-struct path) project into
/// this shape.
pub(super) struct ScanLayerIdents<'a> {
    pub offsets: &'a syn::Ident,
    pub validity: &'a syn::Ident,
    pub bind: &'a syn::Ident,
}

/// Inputs to the shared shape-walker for the per-row push/scan body.
///
/// `outer_some_prefix` is the ident-prefix used for the inner-Some bind in
/// `match Some(<bind>) ...` arms. The flat-vec path uses
/// `__df_derive_some_` and the nested path uses `__df_derive_n_some_` —
/// the `n_` infix isolates the two paths' bind names so they cannot collide
/// inside one generated function (today they cannot appear together; the
/// divergence is a safety margin against future emitter combinations).
///
/// `leaf_body` produces the deepest-layer for-loop body. The walker hands it
/// the inner Vec binding (already Option-unwrapped where applicable) and the
/// caller emits whatever per-element work the leaf needs (typed-buffer push,
/// flat-ref push + optional position scatter, etc).
///
/// `leaf_offsets_post_push` is the token-stream-valued expression cast to
/// `i64` and pushed onto the innermost-layer offsets vec after each outer
/// row's leaf iteration. The flat-vec path passes its typed buffer's `len()`;
/// the nested path passes `flat.len()` (or `positions.len()` under
/// `has_inner_option`).
pub(super) struct ShapeScan<'a> {
    pub shape: &'a VecShape,
    pub access: &'a TokenStream,
    pub layers: &'a [ScanLayerIdents<'a>],
    pub outer_some_prefix: &'a str,
    pub leaf_body: &'a dyn Fn(&TokenStream) -> TokenStream,
    pub leaf_offsets_post_push: &'a TokenStream,
}

impl ShapeScan<'_> {
    /// Build the per-row scan body. Wraps the layer-0 walk in the
    /// `for __df_derive_it in items { ... }` ring shared by both paths.
    pub(super) fn build(&self) -> TokenStream {
        let layer0_iter_src = {
            let access = self.access;
            quote! { (&(#access)) }
        };
        let body = self.build_layer(0, &layer0_iter_src);
        let it = idents::populator_iter();
        quote! {
            for #it in items {
                #body
            }
        }
    }

    /// Emit the body for the iteration at layer `cur`. At the deepest layer,
    /// hand off to `leaf_body`; otherwise recurse into the next layer.
    fn build_iter(&self, cur: usize, vec_bind: &TokenStream) -> TokenStream {
        let depth = self.shape.depth();
        if cur + 1 == depth {
            (self.leaf_body)(vec_bind)
        } else {
            let inner_bind = self.layers[cur + 1].bind;
            let inner_layer_body = self.build_layer(cur + 1, &quote! { #inner_bind });
            quote! {
                for #inner_bind in #vec_bind.iter() {
                    #inner_layer_body
                }
            }
        }
    }

    /// Emit the layer-`cur` body: the outer-Option match (when present), the
    /// inner iteration, and the per-layer offsets push.
    fn build_layer(&self, cur: usize, bind: &TokenStream) -> TokenStream {
        let depth = self.shape.depth();
        let layer = &self.layers[cur];
        let offsets = layer.offsets;
        let offsets_post_push = if cur + 1 == depth {
            self.leaf_offsets_post_push.clone()
        } else {
            let inner_offsets = self.layers[cur + 1].offsets;
            quote! { (#inner_offsets.len() - 1) }
        };
        let opt_layers = self.shape.layers[cur].option_layers;
        let inner_iter = if opt_layers > 0 {
            let validity = layer.validity;
            let inner_vec_bind = format_ident!("{}{}", self.outer_some_prefix, cur);
            let inner_iter = self.build_iter(cur, &quote! { #inner_vec_bind });
            // The bind here holds `&Option<...<Option<Vec<...>>>>` with
            // `opt_layers` of nesting. Collapse to `Option<&Vec<...>>` and
            // match: Some(v) pushes validity=true and iterates v; None
            // pushes validity=false and skips. Polars folds every nested
            // None into the same null. For `opt_layers == 1`, default
            // binding modes match `&Option<Vec<...>>` directly without an
            // explicit `.as_ref()` call, which LLVM doesn't always
            // eliminate — so we keep the bind unchanged in that case.
            let collapsed = if opt_layers == 1 {
                bind.clone()
            } else {
                collapse_options_to_ref(bind, opt_layers)
            };
            quote! {
                match #collapsed {
                    ::std::option::Option::Some(#inner_vec_bind) => {
                        #validity.push(true);
                        #inner_iter
                    }
                    ::std::option::Option::None => {
                        #validity.push(false);
                    }
                }
            }
        } else {
            self.build_iter(cur, bind)
        };
        quote! {
            #inner_iter
            #offsets.push(#offsets_post_push as i64);
        }
    }
}

/// Inputs to the shared precount walker. Builds the precount loop both the
/// flat-vec and nested-struct paths run before the scan to size the
/// offsets/validity/leaf-storage buffers up front.
///
/// `total_counter` is the leaf-element accumulator ident (the flat-vec path
/// passes `__df_derive_total_leaves`; the nested path passes
/// `__df_derive_gen_total_<idx>`). `layer_counters` are the per-depth
/// counter idents (`depth - 1` of them; layer `i` counts the child-lists
/// produced by layer `i+1`). `layers` reuses the `ScanLayerIdents` shape
/// from the scan walker; only the `bind` field is consumed (for the
/// inter-layer `for #inner_bind in ...` loops).
///
/// `outer_some_prefix` is the ident-prefix for the `if let Some(<bind>)`
/// arm at layers with `option_layers > 0`. The flat-vec path uses
/// `__df_derive_some_` (matching its scan walker); the nested path uses
/// `__df_derive_n_pre_some_` (distinct from its scan walker's
/// `__df_derive_n_some_` so the two loops can coexist without name shadowing
/// inside the same generated block).
///
/// The walker calls [`collapse_options_to_ref`] when `option_layers >= 2`
/// so multi-Option-over-Vec shapes type-check (default binding modes can't
/// match `&Option<Option<Vec<...>>>` against `Some(v)` directly without
/// stripping the outer `&` first).
///
/// `build` emits the entire precount block: the `let mut <total>: usize = 0;`
/// decl, the per-layer counter decls, and the `for <it> in items { ... }`
/// ring with the recursion body. Callers splice the result verbatim.
pub(super) struct ShapePrecount<'a> {
    pub shape: &'a VecShape,
    pub access: &'a TokenStream,
    pub layers: &'a [ScanLayerIdents<'a>],
    pub outer_some_prefix: &'a str,
    pub total_counter: &'a syn::Ident,
    pub layer_counters: &'a [syn::Ident],
}

impl ShapePrecount<'_> {
    /// Build the full precount block (total + per-layer counter decls plus
    /// the `for <it> in items { ... }` ring).
    pub(super) fn build(&self) -> TokenStream {
        let layer0_iter_src = {
            let access = self.access;
            quote! { (&(#access)) }
        };
        let body = self.build_layer(0, &layer0_iter_src);
        let total = self.total_counter;
        let counter_decls = self
            .layer_counters
            .iter()
            .map(|c| quote! { let mut #c: usize = 0; });
        let it = idents::populator_iter();
        quote! {
            let mut #total: usize = 0;
            #(#counter_decls)*
            for #it in items {
                #body
            }
        }
    }

    fn build_iter(&self, cur: usize, vec_bind: &TokenStream) -> TokenStream {
        let depth = self.shape.depth();
        let total = self.total_counter;
        if cur + 1 == depth {
            quote! { #total += #vec_bind.len(); }
        } else {
            let inner_bind = self.layers[cur + 1].bind;
            let counter = &self.layer_counters[cur];
            let inner_layer_body = self.build_layer(cur + 1, &quote! { #inner_bind });
            quote! {
                for #inner_bind in #vec_bind.iter() {
                    #inner_layer_body
                    #counter += 1;
                }
            }
        }
    }

    fn build_layer(&self, cur: usize, bind: &TokenStream) -> TokenStream {
        let opt_layers = self.shape.layers[cur].option_layers;
        if opt_layers > 0 {
            let inner_vec_bind = format_ident!("{}{}", self.outer_some_prefix, cur);
            let inner = self.build_iter(cur, &quote! { #inner_vec_bind });
            let collapsed = if opt_layers == 1 {
                bind.clone()
            } else {
                collapse_options_to_ref(bind, opt_layers)
            };
            quote! {
                if let ::std::option::Option::Some(#inner_vec_bind) = #collapsed {
                    #inner
                }
            }
        } else {
            self.build_iter(cur, bind)
        }
    }
}

/// Per-layer offsets-vec declarations. Layer 0 is sized `items.len() + 1`;
/// deeper layers' capacity comes from the caller-supplied counter expression
/// (the precount loop's per-layer counter ident, plus `1`).
pub(super) fn shape_offsets_decls(
    layers: &[&syn::Ident],
    layer_counter: &dyn Fn(usize) -> TokenStream,
) -> TokenStream {
    let mut out: Vec<TokenStream> = Vec::with_capacity(layers.len());
    for (i, offsets) in layers.iter().enumerate() {
        let cap = if i == 0 {
            quote! { items.len() + 1 }
        } else {
            let counter = layer_counter(i - 1);
            quote! { #counter + 1 }
        };
        out.push(quote! {
            let mut #offsets: ::std::vec::Vec<i64> =
                ::std::vec::Vec::with_capacity(#cap);
            #offsets.push(0);
        });
    }
    quote! { #(#out)* }
}

/// Per-layer outer-`Option` validity-bitmap declarations. Allocated push-
/// based (no pre-fill) — `Some` arms push `true`, `None` arms push `false`.
/// Skips layers without an outer Option.
pub(super) fn shape_validity_decls(
    shape: &VecShape,
    validity_per_layer: &[&syn::Ident],
    layer_counter: &dyn Fn(usize) -> TokenStream,
    pa_root: &TokenStream,
) -> TokenStream {
    let mut out: Vec<TokenStream> = Vec::new();
    for (i, validity) in validity_per_layer.iter().enumerate() {
        if !shape.layers[i].has_outer_validity() {
            continue;
        }
        let cap = if i == 0 {
            quote! { items.len() }
        } else {
            let counter = layer_counter(i - 1);
            quote! { #counter }
        };
        out.push(quote! {
            let mut #validity: #pa_root::bitmap::MutableBitmap =
                #pa_root::bitmap::MutableBitmap::with_capacity(#cap);
        });
    }
    quote! { #(#out)* }
}

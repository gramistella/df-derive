//! Shared shape-walker for the `[Vec, ...]` push/scan body.
//!
//! The unified emitter (`emit::vec_emit_general`) drives both per-element-push
//! and collect-then-bulk paths through these primitives, parameterized by
//! `LeafKind`. The two paths walk the same `VecLayers` recursively and differ
//! only at the deepest layer (push to a typed buffer vs. push a ref into a
//! `Vec<&T>` plus optional position scatter) and in a few naming choices
//! (outer-Some bind name prefix, leaf-binding ident, the per-layer
//! offsets-push value at the innermost layer).
//!
//! [`ShapeScan`] captures all of those decision points behind one struct so
//! the recursion logic lives in exactly one place. The two callers share a
//! unified [`LayerIdents`] bundle (5 fields: offsets / `offsets_buf` /
//! `validity_mb` / `validity_bm` / bind); the walker reads `offsets`,
//! `validity_mb`, and `bind` directly off it. Each caller chooses the
//! `outer_some_prefix` and supplies the deepest-layer body via `leaf_body`.
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
//!
//! [`shape_assemble_list_stack`] is the shared per-layer wrap stack that
//! both paths emit after the scan completes. Each caller supplies a
//! `[LayerWrap]` slice describing every layer's offsets-buf ownership via
//! [`OwnPolicy`]: `OwnPolicy::Move` for single-use sites (the flat-vec path,
//! where each frozen offsets buffer feeds exactly one `LargeListArray::new`)
//! and `OwnPolicy::Clone` when the buffer rides across multiple downstream
//! arms (the nested encoder's four-arm dispatch reuses the same offsets
//! buffer per arm). The helper emits the reversed `LargeListArray::new`
//! chain plus the routed `__df_derive_assemble_list_series_unchecked` call
//! at the outermost layer. Per-layer `OffsetsBuffer::try_from(...)?` freeze
//! decls stay interleaved with each wrap rather than hoisted upfront —
//! moving them to the top produced a reproducible regression on
//! `vec_vec_opt_string`.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::ir::{AccessChain, VecLayers};

use super::idents;
use super::{access_chain_to_option_ref, access_chain_to_ref};

/// Optional projection injected at an inter-layer transition. Tuple fields
/// use this for shapes like `Vec<(Vec<A>, B)>`: the parent tuple's `Vec`
/// layer is walked first, then the next layer receives `&tuple.0`.
/// When the tuple element has no own `Vec` layer, projection happens at the
/// leaf instead and this stays `None`.
pub(super) struct LayerProjection<'a> {
    pub layer: usize,
    pub path: &'a TokenStream,
    /// Transparent wrappers between the parent Vec item and the tuple
    /// itself. Projection resolves this once, then the target layer receives
    /// either `&Element` or `Option<&Element>`.
    pub parent_access: &'a AccessChain,
    /// Smart pointers wrapped around the projected element before its own
    /// Vec/Option layers.
    pub smart_ptr_depth: usize,
}

fn projection_base_to_ref(item_bind: &syn::Ident, parent_access: &AccessChain) -> TokenStream {
    if parent_access.is_empty() {
        return quote! { #item_bind };
    }
    if parent_access.option_layers() > 0 {
        return access_chain_to_option_ref(&quote! { #item_bind }, parent_access);
    }
    access_chain_to_ref(&quote! { #item_bind }, parent_access).expr
}

fn projected_layer_bind(
    item_bind: &syn::Ident,
    projection: &LayerProjection<'_>,
    bind_prefix: &str,
    cur: usize,
) -> TokenStream {
    let path = projection.path;
    let project_from = |tuple_ref: &TokenStream| -> TokenStream {
        let mut projected = quote! { (*(#tuple_ref)) #path };
        for _ in 0..projection.smart_ptr_depth {
            projected = quote! { (*(#projected)) };
        }
        quote! { &(#projected) }
    };

    let tuple_ref = projection_base_to_ref(item_bind, projection.parent_access);
    if projection.parent_access.option_layers() == 0 {
        return project_from(&tuple_ref);
    }

    let param = format_ident!("{bind_prefix}proj_{cur}");
    let projected = project_from(&quote! { #param });
    quote! { (#tuple_ref).map(|#param| #projected) }
}

/// Unified per-layer identifier bundle shared by the flat-vec path and the
/// nested-struct path. Both paths construct these via the shared
/// `emit::layer_idents` factory, which dispatches on `field_idx:
/// Option<usize>` between the per-layer (per-element-push) and per-(field,
/// layer) (collect-then-bulk) ident sets. Once built, every consumer (scan
/// walker, precount walker, offsets-decl helper, validity-decl helper,
/// final-assemble) reads from the same shape.
pub(super) struct LayerIdents {
    /// Mutable `Vec<i64>` offsets accumulator for this layer.
    pub offsets: syn::Ident,
    /// Frozen `OffsetsBuffer<i64>` for this layer (post `try_from`).
    pub offsets_buf: syn::Ident,
    /// Mutable `MutableBitmap` for this layer's outer-Option validity.
    pub validity_mb: syn::Ident,
    /// Frozen `Bitmap` for this layer's outer-Option validity (post `From`).
    pub validity_bm: syn::Ident,
    /// Per-layer iteration binding. Layer 0 binds the field access; deeper
    /// layers bind the previous layer's iterator output.
    pub bind: syn::Ident,
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
    pub shape: &'a VecLayers,
    pub access: &'a TokenStream,
    pub layers: &'a [LayerIdents],
    pub outer_some_prefix: &'a str,
    pub leaf_body: &'a dyn Fn(&TokenStream) -> TokenStream,
    pub leaf_offsets_post_push: &'a TokenStream,
    pub projection: Option<LayerProjection<'a>>,
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
            let inner_bind = &self.layers[cur + 1].bind;
            let inner_layer_body = self.build_layer(cur + 1, &quote! { #inner_bind });
            self.projection
                .as_ref()
                .filter(|p| cur + 1 == p.layer)
                .map_or_else(
                    || {
                        quote! {
                            for #inner_bind in #vec_bind.iter() {
                                #inner_layer_body
                            }
                        }
                    },
                    |projection| {
                        let item_bind =
                            format_ident!("{}proj_item_{}", self.outer_some_prefix, cur);
                        let projected = projected_layer_bind(
                            &item_bind,
                            projection,
                            self.outer_some_prefix,
                            cur,
                        );
                        quote! {
                            for #item_bind in #vec_bind.iter() {
                                let #inner_bind = #projected;
                                #inner_layer_body
                            }
                        }
                    },
                )
        }
    }

    /// Emit the layer-`cur` body: the outer-Option match (when present), the
    /// inner iteration, and the per-layer offsets push.
    fn build_layer(&self, cur: usize, bind: &TokenStream) -> TokenStream {
        let depth = self.shape.depth();
        let layer = &self.layers[cur];
        let offsets = &layer.offsets;
        let offsets_post_push = if cur + 1 == depth {
            self.leaf_offsets_post_push.clone()
        } else {
            let inner_offsets = &self.layers[cur + 1].offsets;
            quote! { (#inner_offsets.len() - 1) }
        };
        let layer_access = access_chain_to_ref(bind, &self.shape.layers[cur].access);
        let inner_iter = if layer_access.has_option {
            let validity = &layer.validity_mb;
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
            let collapsed = layer_access.expr;
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
            self.build_iter(cur, &layer_access.expr)
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
/// produced by layer `i+1`). `layers` reuses the unified [`LayerIdents`]
/// shape from the scan walker; only the `bind` field is consumed (for the
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
    pub shape: &'a VecLayers,
    pub access: &'a TokenStream,
    pub layers: &'a [LayerIdents],
    pub outer_some_prefix: &'a str,
    pub total_counter: &'a syn::Ident,
    pub layer_counters: &'a [syn::Ident],
    pub projection: Option<LayerProjection<'a>>,
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
            let inner_bind = &self.layers[cur + 1].bind;
            let counter = &self.layer_counters[cur];
            let inner_layer_body = self.build_layer(cur + 1, &quote! { #inner_bind });
            self.projection
                .as_ref()
                .filter(|p| cur + 1 == p.layer)
                .map_or_else(
                    || {
                        quote! {
                            for #inner_bind in #vec_bind.iter() {
                                #inner_layer_body
                                #counter += 1;
                            }
                        }
                    },
                    |projection| {
                        let item_bind =
                            format_ident!("{}proj_item_{}", self.outer_some_prefix, cur);
                        let projected = projected_layer_bind(
                            &item_bind,
                            projection,
                            self.outer_some_prefix,
                            cur,
                        );
                        quote! {
                            for #item_bind in #vec_bind.iter() {
                                let #inner_bind = #projected;
                                #inner_layer_body
                                #counter += 1;
                            }
                        }
                    },
                )
        }
    }

    fn build_layer(&self, cur: usize, bind: &TokenStream) -> TokenStream {
        let layer_access = access_chain_to_ref(bind, &self.shape.layers[cur].access);
        if layer_access.has_option {
            let inner_vec_bind = format_ident!("{}{}", self.outer_some_prefix, cur);
            let inner = self.build_iter(cur, &quote! { #inner_vec_bind });
            let collapsed = layer_access.expr;
            quote! {
                if let ::std::option::Option::Some(#inner_vec_bind) = #collapsed {
                    #inner
                }
            }
        } else {
            self.build_iter(cur, &layer_access.expr)
        }
    }
}

/// Per-layer inputs to the shared layer-wrap stack.
///
/// `offsets_buf` is the per-layer offsets-buffer expression — an `OwnPolicy`
/// the helper splices verbatim into the `LargeListArray::new` argument
/// position. The flat-vec path passes `OwnPolicy::Move` (its frozen
/// buffer is single-use); the nested-struct path passes `OwnPolicy::Clone`
/// (its buffer is shared across the four-arm dispatch's `for col in
/// schema { ... }` iterations and so cannot be moved out).
///
/// `validity_bm` is the optional outer-Option validity-bitmap source —
/// `None` when the layer has no outer Option (the wrap passes
/// `Option::None` for its validity argument). When present it is always
/// cloned (the same frozen `Bitmap` rides under every list-array layer's
/// `LargeListArray::new` in the same arm and across multiple arms in the
/// nested four-way dispatch).
pub(super) enum OwnPolicy<'a> {
    /// Move the named local into the wrap argument. The local is bound
    /// inside the helper's emission scope and used exactly once.
    Move(&'a syn::Ident),
    /// Clone the named local (`Clone::clone(&#ident)`). Used when the same
    /// frozen buffer is read across multiple `LargeListArray::new` sites
    /// (e.g. across the nested four-arm dispatch's `for col in schema`
    /// iterations).
    Clone(&'a syn::Ident),
}

impl OwnPolicy<'_> {
    fn splice(&self) -> TokenStream {
        match self {
            Self::Move(id) => quote! { #id },
            Self::Clone(id) => quote! { ::std::clone::Clone::clone(&#id) },
        }
    }
}

pub(super) struct LayerWrap<'a> {
    pub offsets_buf: OwnPolicy<'a>,
    pub validity_bm: Option<&'a syn::Ident>,
    /// Optional per-layer freeze decl emitted immediately before this
    /// layer's `LargeListArray::new` call. The flat-vec path uses this to
    /// interleave each layer's `OffsetsBuffer::try_from(...)?` with its
    /// wrap (matching the pre-refactor emission shape — hoisting the
    /// freezes out of the wrap loop reproducibly regresses depth-N
    /// benches by 4-12% even though it is semantically identical). The
    /// nested-struct path leaves this empty because its freeze happens
    /// once, above the four-arm dispatch.
    pub freeze_decl: TokenStream,
}

/// Freeze a per-layer `Vec<i64>` into an `OffsetsBuffer<i64>` (single
/// statement). Centralized for all list-stack emitters so tuple fields and
/// nested/primitive paths keep the same generated token shape.
pub(super) fn freeze_offsets_buf(
    buf: &syn::Ident,
    offsets: &syn::Ident,
    pa_root: &TokenStream,
) -> TokenStream {
    quote! {
        let #buf: #pa_root::offset::OffsetsBuffer<i64> =
            #pa_root::offset::OffsetsBuffer::try_from(#offsets)?;
    }
}

/// Freeze a per-layer `MutableBitmap` into a `Bitmap` (single statement).
/// Shared by every list-stack emitter that carries outer-list validity.
pub(super) fn freeze_validity_bitmap(
    bm: &syn::Ident,
    mb: &syn::Ident,
    pa_root: &TokenStream,
) -> TokenStream {
    quote! {
        let #bm: #pa_root::bitmap::Bitmap =
            <#pa_root::bitmap::Bitmap as ::core::convert::From<
                #pa_root::bitmap::MutableBitmap,
            >>::from(#mb);
    }
}

/// Stack `layers.len()` `LargeListArray::new` calls (innermost-first,
/// outermost-last) and route the outermost through
/// `__df_derive_assemble_list_series_unchecked`.
///
/// `seed` is an expression evaluating to an `ArrayRef` (boxed leaf array
/// for the flat-vec path, `chunks()[0].clone()` for the nested path). It
/// is moved verbatim into the innermost `LargeListArray::new` call's
/// values argument. `seed_dtype` is the arrow dtype of that seed array
/// as a token expression — captured BEFORE the seed is boxed/moved so
/// the flat-vec path can keep its static `Array::dtype(&typed_leaf_arr)`
/// call (a virtual dispatch through `Box<dyn Array>::dtype()` does not
/// inline and reproducibly regresses several depth-N benches by 5-12%).
///
/// `leaf_logical_dtype` is the per-leaf logical dtype (e.g. `DataType::String`
/// for the flat-vec path, `(*__df_derive_dtype).clone()` for the nested
/// path). The helper wraps it in `(layers.len() - 1)` extra `List<>`
/// envelopes so the schema dtype matches the runtime list nesting (the
/// `__df_derive_assemble_list_series_unchecked` helper adds the outermost
/// `List<>` itself).
///
/// `arr_id_for_layer(cur)` produces the per-layer `LargeListArray` local
/// ident — the flat-vec path uses [`idents::vec_layer_list_arr`] and the
/// nested path uses [`idents::nested_layer_list_arr`].
pub(super) fn shape_assemble_list_stack(
    seed: TokenStream,
    seed_dtype: TokenStream,
    layers: &[LayerWrap<'_>],
    leaf_logical_dtype: TokenStream,
    pp: &TokenStream,
    pa_root: &TokenStream,
    arr_id_for_layer: &dyn Fn(usize) -> syn::Ident,
) -> TokenStream {
    let depth = layers.len();

    let mut block: Vec<TokenStream> = Vec::with_capacity(depth * 2);
    let mut prev_payload = seed;
    let mut prev_dtype = seed_dtype;
    for cur in (0..depth).rev() {
        let layer = &layers[cur];
        let freeze = &layer.freeze_decl;
        let buf_splice = layer.offsets_buf.splice();
        let arr_id = arr_id_for_layer(cur);
        let validity_expr = layer.validity_bm.map_or_else(
            || quote! { ::std::option::Option::None },
            |bm| quote! { ::std::option::Option::Some(::std::clone::Clone::clone(&#bm)) },
        );
        block.push(quote! {
            #freeze
            let #arr_id: #pp::LargeListArray = #pp::LargeListArray::new(
                #pp::LargeListArray::default_datatype(#prev_dtype),
                #buf_splice,
                #prev_payload,
                #validity_expr,
            );
        });
        // Subsequent wraps box the previous `LargeListArray` into an
        // `ArrayRef` and read its dtype via UFCS so the `Array` trait
        // method resolves regardless of whether the trait is in scope at
        // the user call site.
        prev_payload = quote! { ::std::boxed::Box::new(#arr_id) as #pp::ArrayRef };
        prev_dtype = quote! { #pa_root::array::Array::dtype(&#arr_id).clone() };
    }

    let helper_logical = crate::codegen::external_paths::wrap_list_layers_compile_time(
        pp,
        leaf_logical_dtype,
        depth.saturating_sub(1),
    );
    let outer = arr_id_for_layer(0);
    let assemble_helper = idents::assemble_helper();
    quote! {
        #(#block)*
        #assemble_helper(
            #outer,
            #helper_logical,
        )
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
    shape: &VecLayers,
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

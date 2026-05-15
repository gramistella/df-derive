//! Shared shape walkers for `Vec`-layer precount, scan, and list assembly.
//!
//! Dtype/array compatibility is owned here: leaf encoders may create Arrow
//! arrays and logical Polars dtypes, but `shape_assemble_list_stack` is the
//! only boundary that pairs them into list Series construction.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::ir::{AccessChain, VecLayers};

use super::idents;
use super::{access_chain_to_option_ref, access_chain_to_ref, list_offset_i64_expr};

/// Optional tuple projection injected at an inter-layer transition.
#[derive(Clone, Copy)]
pub(super) struct LayerProjection<'a> {
    pub layer: usize,
    pub path: &'a TokenStream,
    /// Transparent wrappers between the parent Vec item and the tuple itself.
    pub parent_access: &'a AccessChain,
    /// Smart pointers wrapped around the projected element before its own layers.
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

pub(super) struct LayerIdents {
    pub offsets: syn::Ident,
    pub offsets_buf: syn::Ident,
    pub validity_mb: syn::Ident,
    pub validity_bm: syn::Ident,
    pub bind: syn::Ident,
}

impl From<idents::LayerIds> for LayerIdents {
    fn from(ids: idents::LayerIds) -> Self {
        Self {
            offsets: ids.offsets,
            offsets_buf: ids.offsets_buf,
            validity_mb: ids.validity_mb,
            validity_bm: ids.validity_bm,
            bind: ids.bind,
        }
    }
}

pub(super) struct ShapeScan<'shape, 'body> {
    pub shape: &'shape VecLayers,
    pub access: &'shape TokenStream,
    pub layers: &'shape [LayerIdents],
    pub outer_some_prefix: &'shape str,
    pub leaf_body: &'body dyn Fn(&TokenStream) -> TokenStream,
    pub leaf_offsets_post_push: &'body TokenStream,
    pub pp: &'shape TokenStream,
    pub projection: Option<LayerProjection<'shape>>,
}

impl ShapeScan<'_, '_> {
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
            // Polars folds every nested None at this Vec boundary into one null bit.
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
        let offset_ident = idents::list_offset();
        let offset = list_offset_i64_expr(&offsets_post_push, self.pp);
        quote! {
            #inner_iter
            let #offset_ident: i64 = #offset;
            #offsets.push(#offset_ident);
        }
    }
}

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

pub(super) struct ShapeEmitter<'a> {
    pub shape: &'a VecLayers,
    pub access: &'a TokenStream,
    pub layers: &'a [LayerIdents],
    pub outer_some_prefix: &'a str,
    pub precount_outer_some_prefix: &'a str,
    pub total_counter: &'a syn::Ident,
    pub layer_counters: &'a [syn::Ident],
    pub pp: &'a TokenStream,
    pub pa_root: &'a TokenStream,
    pub projection: Option<LayerProjection<'a>>,
}

impl<'a> ShapeEmitter<'a> {
    pub(super) fn precount(&self) -> TokenStream {
        ShapePrecount {
            shape: self.shape,
            access: self.access,
            layers: self.layers,
            outer_some_prefix: self.precount_outer_some_prefix,
            total_counter: self.total_counter,
            layer_counters: self.layer_counters,
            projection: self.projection,
        }
        .build()
    }

    pub(super) fn scan<'body>(
        &self,
        leaf_body: &'body dyn Fn(&TokenStream) -> TokenStream,
        leaf_offsets_post_push: &'body TokenStream,
    ) -> TokenStream {
        ShapeScan {
            shape: self.shape,
            access: self.access,
            layers: self.layers,
            outer_some_prefix: self.outer_some_prefix,
            leaf_body,
            leaf_offsets_post_push,
            pp: self.pp,
            projection: self.projection,
        }
        .build()
    }

    pub(super) fn offsets_decls(
        &self,
        counter_for_depth: &dyn Fn(usize) -> TokenStream,
    ) -> TokenStream {
        let offsets: Vec<&syn::Ident> = self.layers.iter().map(|layer| &layer.offsets).collect();
        shape_offsets_decls(&offsets, counter_for_depth)
    }

    pub(super) fn validity_decls(
        &self,
        counter_for_depth: &dyn Fn(usize) -> TokenStream,
    ) -> TokenStream {
        let validity: Vec<&syn::Ident> =
            self.layers.iter().map(|layer| &layer.validity_mb).collect();
        shape_validity_decls(self.shape, &validity, counter_for_depth, self.pa_root)
    }

    pub(super) fn layer_wraps_move(&self) -> Vec<LayerWrap<'a>> {
        shape_layer_wraps_move(self.shape, self.layers, self.pa_root)
    }
}
pub(super) enum OwnPolicy<'a> {
    Move(&'a syn::Ident),
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
    pub freeze_decl: TokenStream,
}

pub(super) fn shape_freeze_validity_bitmaps(
    shape: &VecLayers,
    layers: &[LayerIdents],
    pa_root: &TokenStream,
) -> TokenStream {
    let mut freezes: Vec<TokenStream> = Vec::new();
    for (idx, layer) in layers.iter().enumerate() {
        if shape.layers[idx].has_outer_validity() {
            freezes.push(freeze_validity_bitmap(
                &layer.validity_bm,
                &layer.validity_mb,
                pa_root,
            ));
        }
    }
    quote! { #(#freezes)* }
}

pub(super) fn shape_freeze_offsets_buffers(
    layers: &[LayerIdents],
    pa_root: &TokenStream,
) -> TokenStream {
    let freezes = layers
        .iter()
        .map(|layer| freeze_offsets_buf(&layer.offsets_buf, &layer.offsets, pa_root));
    quote! { #(#freezes)* }
}

pub(super) fn shape_layer_wraps_move<'a>(
    shape: &VecLayers,
    layers: &'a [LayerIdents],
    pa_root: &TokenStream,
) -> Vec<LayerWrap<'a>> {
    let mut out: Vec<LayerWrap<'_>> = Vec::with_capacity(shape.depth());
    for (cur, layer) in layers.iter().enumerate() {
        let mut freeze_decl = freeze_offsets_buf(&layer.offsets_buf, &layer.offsets, pa_root);
        let validity_bm = if shape.layers[cur].has_outer_validity() {
            freeze_decl.extend(freeze_validity_bitmap(
                &layer.validity_bm,
                &layer.validity_mb,
                pa_root,
            ));
            Some(&layer.validity_bm)
        } else {
            None
        };
        out.push(LayerWrap {
            offsets_buf: OwnPolicy::Move(&layer.offsets_buf),
            validity_bm,
            freeze_decl,
        });
    }
    out
}

pub(super) fn shape_layer_wraps_clone<'a>(
    shape: &VecLayers,
    layers: &'a [LayerIdents],
) -> Vec<LayerWrap<'a>> {
    let mut out: Vec<LayerWrap<'_>> = Vec::with_capacity(shape.depth());
    for (cur, layer) in layers.iter().enumerate() {
        let validity_bm = shape.layers[cur]
            .has_outer_validity()
            .then_some(&layer.validity_bm);
        out.push(LayerWrap {
            offsets_buf: OwnPolicy::Clone(&layer.offsets_buf),
            validity_bm,
            freeze_decl: TokenStream::new(),
        });
    }
    out
}

pub(super) fn freeze_offsets_buf(
    buf: &syn::Ident,
    offsets: &syn::Ident,
    pa_root: &TokenStream,
) -> TokenStream {
    quote! {
        let #buf: #pa_root::offset::OffsetsBuffer<i64> =
            <#pa_root::offset::OffsetsBuffer<i64> as ::core::convert::TryFrom<::std::vec::Vec<i64>>>::try_from(#offsets)?;
    }
}

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
    debug_assert!(
        depth > 0,
        "shape_assemble_list_stack requires at least one Vec layer"
    );

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
        )?
    }
}

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

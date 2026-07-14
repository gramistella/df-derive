//! Shared shape walker for one-pass `Vec` scanning and list assembly.
//!
//! Dtype/array compatibility is owned here: leaf encoders may create Arrow
//! arrays and logical Polars dtypes, but `shape_assemble_list_stack` is the
//! only boundary that pairs them into list Series construction.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::ir::{NonEmpty, VecLayers};

use super::idents::{self, LayerIdents};
use super::{access_chain_to_ref, list_offset_i64_expr};

pub(super) struct ShapeScan<'shape, 'body> {
    pub shape: &'shape VecLayers,
    pub access: &'shape TokenStream,
    pub layers: &'shape [LayerIdents],
    pub outer_some_prefix: &'shape str,
    pub leaf_body: &'body dyn Fn(&TokenStream) -> TokenStream,
    pub leaf_offsets_post_push: &'body TokenStream,
    pub pp: &'shape TokenStream,
}

impl ShapeScan<'_, '_> {
    pub(super) fn build(&self) -> TokenStream {
        let layer0_iter_src = {
            let access = self.access;
            quote! { (&(#access)) }
        };
        self.build_layer(0, &layer0_iter_src)
    }

    fn build_iter(&self, cur: usize, vec_bind: &TokenStream) -> TokenStream {
        let depth = self.shape.depth();
        if cur + 1 == depth {
            (self.leaf_body)(vec_bind)
        } else {
            let inner_bind = &self.layers[cur + 1].bind;
            let inner_layer_body = self.build_layer(cur + 1, &quote! { #inner_bind });
            quote! {
                for #inner_bind in #vec_bind.iter() {
                    #inner_layer_body
                }
            }
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

struct ShapeCount<'shape, 'counts> {
    shape: &'shape VecLayers,
    access: &'shape TokenStream,
    layers: &'shape [LayerIdents],
    outer_some_prefix: &'shape str,
    counts: &'counts syn::Ident,
    pp: &'shape TokenStream,
}

impl ShapeCount<'_, '_> {
    fn build(&self) -> TokenStream {
        let access = self.access;
        self.build_layer(0, &quote! { (&(#access)) })
    }

    fn build_iter(&self, cur: usize, vec_bind: &TokenStream) -> TokenStream {
        if cur + 1 == self.shape.depth() {
            let counts = self.counts;
            let leaves = self.shape.depth();
            let pp = self.pp;
            return quote! {
                #counts[#leaves] = #counts[#leaves]
                    .checked_add(#vec_bind.len())
                    .ok_or_else(|| #pp::polars_err!(
                        ComputeError:
                        "df-derive: flattened list element count exceeds usize range",
                    ))?;
            };
        }

        let inner_bind = &self.layers[cur + 1].bind;
        let inner_layer_body = self.build_layer(cur + 1, &quote! { #inner_bind });
        quote! {
            for #inner_bind in #vec_bind.iter() {
                #inner_layer_body
            }
        }
    }

    fn build_layer(&self, cur: usize, bind: &TokenStream) -> TokenStream {
        let layer_access = access_chain_to_ref(bind, &self.shape.layers[cur].access);
        let inner_iter = if layer_access.has_option {
            let inner_vec_bind = format_ident!("{}{}", self.outer_some_prefix, cur);
            let inner_iter = self.build_iter(cur, &quote! { #inner_vec_bind });
            let collapsed = layer_access.expr;
            quote! {
                if let ::std::option::Option::Some(#inner_vec_bind) = #collapsed {
                    #inner_iter
                }
            }
        } else {
            self.build_iter(cur, &layer_access.expr)
        };
        let counts = self.counts;
        let child = cur + 1;
        let offset = list_offset_i64_expr(&quote! { #counts[#child] }, self.pp);
        let pp = self.pp;
        quote! {
            #inner_iter
            let _: i64 = #offset;
            #counts[#cur] = #counts[#cur]
                .checked_add(1)
                .ok_or_else(|| #pp::polars_err!(
                    ComputeError:
                    "df-derive: list layer element count exceeds usize range",
                ))?;
        }
    }
}

pub(super) struct ShapeEmitter<'a> {
    pub row_capacity: &'a syn::Ident,
    pub shape: &'a VecLayers,
    pub access: &'a TokenStream,
    pub layers: &'a [LayerIdents],
    pub outer_some_prefix: &'a str,
    pub pp: &'a TokenStream,
    pub pa_root: &'a TokenStream,
}

#[derive(Clone, Copy)]
pub(super) struct ShapeEmitterParts<'a> {
    pub row_capacity: &'a syn::Ident,
    pub shape: &'a VecLayers,
    pub access: &'a TokenStream,
    pub layers: &'a [LayerIdents],
    pub pp: &'a TokenStream,
    pub pa_root: &'a TokenStream,
}

impl<'a> ShapeEmitter<'a> {
    pub(super) const fn vec(parts: ShapeEmitterParts<'a>) -> Self {
        Self {
            row_capacity: parts.row_capacity,
            shape: parts.shape,
            access: parts.access,
            layers: parts.layers,
            outer_some_prefix: idents::VEC_OUTER_SOME_PREFIX,
            pp: parts.pp,
            pa_root: parts.pa_root,
        }
    }

    pub(super) const fn nested(parts: ShapeEmitterParts<'a>) -> Self {
        Self {
            row_capacity: parts.row_capacity,
            shape: parts.shape,
            access: parts.access,
            layers: parts.layers,
            outer_some_prefix: idents::NESTED_OUTER_SOME_PREFIX,
            pp: parts.pp,
            pa_root: parts.pa_root,
        }
    }

    /// Builds the work for one already-bound outer row. The caller owns the
    /// only iteration over the input iterator.
    pub(super) fn row_push<'body>(
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
        }
        .build()
    }

    /// Counts one already-bound row without retaining per-element references.
    /// Deferred encoders use these cardinalities to allocate the whole list
    /// shape exactly before replaying through [`Self::row_push`].
    pub(super) fn row_count(&self, counts: &syn::Ident) -> TokenStream {
        ShapeCount {
            shape: self.shape,
            access: self.access,
            layers: self.layers,
            outer_some_prefix: self.outer_some_prefix,
            counts,
            pp: self.pp,
        }
        .build()
    }

    pub(super) fn offsets_decls(&self) -> TokenStream {
        shape_offsets_decls(self)
    }

    pub(super) fn validity_decls(&self) -> TokenStream {
        shape_validity_decls(self)
    }

    pub(super) fn exact_offsets_decls(&self, counts: &syn::Ident) -> TokenStream {
        shape_exact_offsets_decls(self, counts)
    }

    pub(super) fn exact_validity_decls(&self, counts: &syn::Ident) -> TokenStream {
        shape_exact_validity_decls(self, counts)
    }

    pub(super) fn layer_wraps_move(&self) -> NonEmpty<LayerWrap<'a>> {
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
) -> NonEmpty<LayerWrap<'a>> {
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
    NonEmpty::from_vec(out).expect("VecLayers always has at least one layer")
}

pub(super) fn shape_layer_wraps_clone<'a>(
    shape: &VecLayers,
    layers: &'a [LayerIdents],
) -> NonEmpty<LayerWrap<'a>> {
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
    NonEmpty::from_vec(out).expect("VecLayers always has at least one layer")
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
    layers: &NonEmpty<LayerWrap<'_>>,
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
        )?
    }
}

fn shape_offsets_decls(emitter: &ShapeEmitter<'_>) -> TokenStream {
    let mut out: Vec<TokenStream> = Vec::with_capacity(emitter.layers.len());
    for (idx, layer) in emitter.layers.iter().enumerate() {
        let offsets = &layer.offsets;
        let row_capacity = emitter.row_capacity;
        let decl = if idx == 0 {
            quote! { ::std::vec::Vec::with_capacity(#row_capacity.saturating_add(1)) }
        } else {
            quote! { ::std::vec::Vec::new() }
        };
        out.push(quote! {
            let mut #offsets: ::std::vec::Vec<i64> = #decl;
            #offsets.push(0);
        });
    }
    quote! { #(#out)* }
}

fn shape_exact_offsets_decls(emitter: &ShapeEmitter<'_>, counts: &syn::Ident) -> TokenStream {
    let declarations = emitter.layers.iter().enumerate().map(|(idx, layer)| {
        let offsets = &layer.offsets;
        quote! {
            let mut #offsets: ::std::vec::Vec<i64> = ::std::vec::Vec::with_capacity(
                #counts[#idx].saturating_add(1),
            );
            #offsets.push(0);
        }
    });
    quote! { #(#declarations)* }
}

fn shape_validity_decls(emitter: &ShapeEmitter<'_>) -> TokenStream {
    let mut out: Vec<TokenStream> = Vec::new();
    for (i, layer) in emitter.layers.iter().enumerate() {
        if !emitter.shape.layers[i].has_outer_validity() {
            continue;
        }
        let validity = &layer.validity_mb;
        let pa_root = emitter.pa_root;
        let row_capacity = emitter.row_capacity;
        let capacity = if i == 0 {
            quote! { #row_capacity }
        } else {
            quote! { 0usize }
        };
        out.push(quote! {
            let mut #validity: #pa_root::bitmap::MutableBitmap =
                #pa_root::bitmap::MutableBitmap::with_capacity(#capacity);
        });
    }
    quote! { #(#out)* }
}

fn shape_exact_validity_decls(emitter: &ShapeEmitter<'_>, counts: &syn::Ident) -> TokenStream {
    let declarations = emitter
        .layers
        .iter()
        .enumerate()
        .filter(|(idx, _)| emitter.shape.layers[*idx].has_outer_validity())
        .map(|(idx, layer)| {
            let validity = &layer.validity_mb;
            let pa_root = emitter.pa_root;
            quote! {
                let mut #validity: #pa_root::bitmap::MutableBitmap =
                    #pa_root::bitmap::MutableBitmap::with_capacity(#counts[#idx]);
            }
        });
    quote! { #(#declarations)* }
}

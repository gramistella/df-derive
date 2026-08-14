//! Shared shape walker for one-pass `Vec` scanning and list assembly.
//!
//! Dtype/array compatibility is owned here: leaf encoders may create Arrow
//! arrays and logical Polars dtypes, but `shape_assemble_list_stack` is the
//! only boundary that pairs them through Polars' checked Series constructor.

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
    pub group_capture: Option<GroupCapture<'body>>,
    pub pp: &'shape TokenStream,
}

#[derive(Clone, Copy)]
pub(super) struct GroupCapture<'a> {
    pub leaf_groups: &'a syn::Ident,
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
        if cur + 2 == depth
            && let Some(group_capture) = self.group_capture
        {
            let leaf_groups = group_capture.leaf_groups;
            let pp = self.pp;
            return quote! {
                #leaf_groups.capture_group(#vec_bind).ok_or_else(|| {
                    #pp::polars_err!(
                        ComputeError:
                        "df-derive: flattened list element count exceeds supported offset range",
                    )
                })?;
            };
        }
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
        let offsets_post_push = if cur + 2 == depth
            && let Some(group_capture) = self.group_capture
        {
            let leaf_groups = group_capture.leaf_groups;
            quote! { #leaf_groups.segment_count() }
        } else if cur + 1 == depth {
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
            group_capture: None,
            pp: self.pp,
        }
        .build()
    }

    /// Captures one reference per penultimate list group while constructing
    /// the list shape during the source pass.
    pub(super) fn row_push_grouped(&self, leaf_groups: &syn::Ident) -> TokenStream {
        debug_assert!(self.shape.depth() >= 2);
        debug_assert!(self.shape.layers[self.shape.depth() - 1].access.is_empty());
        let leaf_body = |_: &TokenStream| TokenStream::new();
        let leaf_offsets_post_push = TokenStream::new();
        ShapeScan {
            shape: self.shape,
            access: self.access,
            layers: self.layers,
            outer_some_prefix: self.outer_some_prefix,
            leaf_body: &leaf_body,
            leaf_offsets_post_push: &leaf_offsets_post_push,
            group_capture: Some(GroupCapture { leaf_groups }),
            pp: self.pp,
        }
        .build()
    }

    pub(super) fn offsets_decls(&self) -> TokenStream {
        shape_offsets_decls(self)
    }

    pub(super) fn grouped_outer_offsets_decls(&self) -> TokenStream {
        shape_offsets_decls_for(self, &self.layers[..self.layers.len() - 1])
    }

    pub(super) fn validity_decls(&self) -> TokenStream {
        shape_validity_decls(self)
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

#[derive(Clone)]
pub(super) enum ListAssemblyTarget {
    SchemaSlot(syn::Ident),
    Intermediate,
}

impl ListAssemblyTarget {
    pub(super) fn schema_slot(output_slot: &syn::Ident) -> Self {
        Self::SchemaSlot(output_slot.clone())
    }

    pub(super) const fn intermediate() -> Self {
        Self::Intermediate
    }
}

impl From<&super::ctx::MaterializationTarget> for ListAssemblyTarget {
    fn from(target: &super::ctx::MaterializationTarget) -> Self {
        target
            .schema_slot_ident()
            .map_or(Self::Intermediate, |output_slot| {
                Self::schema_slot(output_slot)
            })
    }
}

pub(super) struct ListAssemblySeed {
    pub payload: TokenStream,
    pub arrow_dtype: TokenStream,
    pub logical_dtype: TokenStream,
}

pub(super) struct ListAssembly<'a, 'layer> {
    pub seed: ListAssemblySeed,
    pub layers: &'a NonEmpty<LayerWrap<'layer>>,
    pub target: ListAssemblyTarget,
    pub pp: &'a TokenStream,
    pub pa_root: &'a TokenStream,
    pub arr_id_for_layer: &'a dyn Fn(usize) -> syn::Ident,
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

pub(super) fn shape_assemble_list_stack(assembly: ListAssembly<'_, '_>) -> TokenStream {
    let ListAssembly {
        seed:
            ListAssemblySeed {
                payload: seed,
                arrow_dtype: seed_dtype,
                logical_dtype: leaf_logical_dtype,
            },
        layers,
        target,
        pp,
        pa_root,
        arr_id_for_layer,
    } = assembly;
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
    let (output_name, output_dtype) = match target {
        ListAssemblyTarget::SchemaSlot(output_slot) => (
            quote! { #output_slot.name().clone() },
            quote! { #output_slot.dtype() },
        ),
        ListAssemblyTarget::Intermediate => (
            quote! { "".into() },
            quote! { &#pp::DataType::List(::std::boxed::Box::new(#helper_logical)) },
        ),
    };
    let outer = arr_id_for_layer(0);
    quote! {
        #(#block)*
        #pp::Series::from_chunk_and_dtype(
            #output_name,
            ::std::boxed::Box::new(#outer) as #pp::ArrayRef,
            #output_dtype,
        )?
    }
}

fn shape_offsets_decls(emitter: &ShapeEmitter<'_>) -> TokenStream {
    shape_offsets_decls_for(emitter, emitter.layers)
}

fn shape_offsets_decls_for(emitter: &ShapeEmitter<'_>, layers: &[LayerIdents]) -> TokenStream {
    let mut out: Vec<TokenStream> = Vec::with_capacity(layers.len());
    for layer in layers {
        let offsets = &layer.offsets;
        let row_capacity = emitter.row_capacity;
        let decl = quote! {
            ::std::vec::Vec::with_capacity(#row_capacity.saturating_add(1))
        };
        out.push(quote! {
            let mut #offsets: ::std::vec::Vec<i64> = #decl;
            #offsets.push(0);
        });
    }
    quote! { #(#out)* }
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
        out.push(quote! {
            let mut #validity: #pa_root::bitmap::MutableBitmap =
                #pa_root::bitmap::MutableBitmap::with_capacity(#row_capacity);
        });
    }
    quote! { #(#out)* }
}

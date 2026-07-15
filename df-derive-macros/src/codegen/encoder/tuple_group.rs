//! Hierarchical tuple-field encoding.
//!
//! Tuple nodes keep their wrappers until this phase. Every Vec-bearing tuple
//! node owns one offsets/validity stack and one innermost tuple-slot counter;
//! all descendants execute inside that shared walk and clone the frozen list
//! buffers when their final columns are assembled.

use proc_macro2::TokenStream;
use quote::quote;

use crate::codegen::MacroConfig;
use crate::codegen::planner::{
    STATIC_TUPLE_REPLAY_LANE_WIDTH, StaticTuplePlan, tuple_group_has_replay_access,
};
use crate::ir::{
    AccessChain, ColumnCommon, LeafShape, NestedLeaf, NestedNamePolicy, NonEmpty,
    TerminalLeafRoute, TupleField, TupleNode, TupleNodeKind, TupleProjectionStep, VecLayerSpec,
    VecLayers, WrapperShape,
};

use super::idents::{self, GeneratedIdentScope, LayerIdents};
use super::nested_columns::SharedListPrefix;
use super::nested_leaf::{build_nested_encoder, build_nested_encoder_with_prefix};
use super::shape_walk::{
    ShapeEmitter, ShapeEmitterParts, shape_assemble_list_stack, shape_freeze_offsets_buffers,
    shape_freeze_validity_bitmaps, shape_layer_wraps_clone,
};
use super::{
    BaseCtx, LeafCardinality, LeafCtx, NestedLeafCtx, access_chain_to_option_ref,
    access_chain_to_ref, build_encoder_with_option_receiver, struct_type_tokens,
};
use crate::codegen::encode_plan::{
    EmitOp, EncodePlan, FinishGroup, InitOp, PostScanOp, ScanOp, SeriesNameState, SeriesPlan,
};

pub(in crate::codegen) struct TupleFieldEmit {
    pub plan: EncodePlan,
    pub terminal_count: usize,
    pub group_count: usize,
}

#[derive(Clone, Copy)]
pub(in crate::codegen) struct TupleFieldEmitParams<'a> {
    pub config: &'a MacroConfig,
    pub ident_scope: GeneratedIdentScope<'a>,
    pub terminal_start: usize,
    pub group_start: usize,
    pub row: &'a syn::Ident,
    pub replay: &'a TokenStream,
    pub static_tuple_plan: StaticTuplePlan,
    pub row_capacity: &'a syn::Ident,
    pub sink: &'a syn::Ident,
}

#[derive(Clone)]
struct SharedListStack {
    specs: Vec<VecLayerSpec>,
    layers: Vec<LayerIdents>,
}

impl SharedListStack {
    const fn empty() -> Self {
        Self {
            specs: Vec::new(),
            layers: Vec::new(),
        }
    }

    fn with_group(&self, shape: &VecLayers, layers: &[LayerIdents]) -> Self {
        let mut combined = self.clone();
        combined.specs.extend(shape.layers.iter().cloned());
        combined.layers.extend(layers.iter().cloned());
        combined
    }

    fn shape(&self) -> Option<VecLayers> {
        NonEmpty::from_vec(self.specs.clone()).map(|layers| VecLayers {
            layers,
            inner_access: AccessChain::empty(),
        })
    }

    const fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }
}

enum InputAccess {
    Required(TokenStream),
    Optional(TokenStream),
}

/// A tuple projection rooted directly in the generated source-row binding.
/// Unlike `InputAccess`, this never refers to locals scoped to the fused row
/// push, so it remains valid inside a later replay loop.
struct ReplayTupleAccess(TokenStream);

struct ReplayedPrimitive {
    init: Vec<InitOp>,
    scan: ScanOp,
    series: TokenStream,
    name: String,
}

impl InputAccess {
    const fn expr(&self) -> &TokenStream {
        match self {
            Self::Required(expr) | Self::Optional(expr) => expr,
        }
    }

    const fn is_optional(&self) -> bool {
        matches!(self, Self::Optional(_))
    }
}

struct TupleBuilder<'a> {
    config: &'a MacroConfig,
    ident_scope: GeneratedIdentScope<'a>,
    row: &'a syn::Ident,
    replay: &'a TokenStream,
    row_capacity: &'a syn::Ident,
    sink: &'a syn::Ident,
    static_tuple_plan: StaticTuplePlan,
    next_terminal: usize,
    next_group: usize,
    terminal_start: usize,
    group_start: usize,
    init: Vec<InitOp>,
    freezes: Vec<PostScanOp>,
    finish: Vec<FinishGroup>,
    replay_lane: Vec<ReplayedPrimitive>,
}

impl TupleBuilder<'_> {
    fn try_collect_replayed_primitive(
        &mut self,
        leaf: crate::ir::PrimitiveLeaf<'_>,
        wrapper: &WrapperShape,
        replay_input: Option<&ReplayTupleAccess>,
        idx: usize,
        name: &str,
    ) -> Option<TokenStream> {
        let replay_input = replay_input?;
        if !self.static_tuple_plan.replays_terminal(leaf, wrapper, true) {
            return None;
        }
        let input_rows_exact = idents::input_rows_exact(self.ident_scope);
        let ctx = LeafCtx {
            base: BaseCtx {
                access: &replay_input.0,
                row_capacity: self.row_capacity,
                sink: self.sink,
                idx,
                name,
            },
            primitive_list_plan: None,
            row_replay: None,
            cardinality: LeafCardinality::InputRows,
            ident_scope: self.ident_scope,
            input_rows_exact: &input_rows_exact,
            decimal128_encode_trait: &self.config.runtime.decimal128_encode,
            paths: &self.config.external_paths,
        };
        let encoder = build_encoder_with_option_receiver(leaf, wrapper, &ctx, None);
        self.collect_replayed_primitive(encoder, name);
        Some(TokenStream::new())
    }

    fn build_group(
        &mut self,
        wrapper: &WrapperShape,
        elements: &NonEmpty<TupleNode>,
        input: InputAccess,
        replay_input: Option<ReplayTupleAccess>,
        prefix: &SharedListStack,
    ) -> TokenStream {
        self.flush_replay_lane();
        let group_idx = self.next_group;
        self.next_group += 1;
        let input_optional = input.is_optional();
        let (input_decl, input) = match input {
            InputAccess::Required(expr) => (TokenStream::new(), InputAccess::Required(expr)),
            InputAccess::Optional(expr) => {
                let input = idents::tuple_input(self.ident_scope, group_idx);
                (
                    quote! { let #input = #expr; },
                    InputAccess::Optional(quote! { #input }),
                )
            }
        };
        let effective_wrapper = inherit_parent_option(wrapper, input_optional);
        let replay_input = tuple_group_has_replay_access(&effective_wrapper)
            .then_some(replay_input)
            .flatten();
        let body = match &effective_wrapper {
            WrapperShape::Leaf(shape) => {
                let input_expr = input.expr();
                let base = quote! { &(#input_expr) };
                let tuple_value = idents::tuple_value(self.ident_scope, group_idx);
                let (resolved, tuple) = if shape.access().has_option() {
                    (
                        access_chain_to_option_ref(&base, shape.access()),
                        InputAccess::Optional(quote! { #tuple_value }),
                    )
                } else {
                    (
                        access_chain_to_ref(&base, shape.access()).expr,
                        InputAccess::Required(quote! { #tuple_value }),
                    )
                };
                let children = self.build_children(elements, &tuple, replay_input.as_ref(), prefix);
                if children.is_empty() {
                    TokenStream::new()
                } else {
                    quote! {
                        let #tuple_value = #resolved;
                        #children
                    }
                }
            }
            WrapperShape::Vec(shape) => {
                self.build_vec_group(group_idx, shape, elements, &input, prefix)
            }
        };
        self.flush_replay_lane();
        quote! {
            #input_decl
            #body
        }
    }

    fn build_vec_group(
        &mut self,
        group_idx: usize,
        shape: &VecLayers,
        elements: &NonEmpty<TupleNode>,
        input: &InputAccess,
        prefix: &SharedListStack,
    ) -> TokenStream {
        let layers: Vec<LayerIdents> = (0..shape.depth())
            .map(|layer| LayerIdents::tuple(self.ident_scope, group_idx, layer))
            .collect();
        let count = idents::tuple_item_count(self.ident_scope, group_idx);
        let access = input.expr();
        let pp = self.config.external_paths.prelude();
        let pa_root = self.config.external_paths.polars_arrow_root();
        let emitter = ShapeEmitter::vec(ShapeEmitterParts {
            row_capacity: self.row_capacity,
            shape,
            access,
            layers: &layers,
            pp,
            pa_root,
        });

        let offsets_decls = emitter.offsets_decls();
        let validity_decls = emitter.validity_decls();
        self.init.push(InitOp::new(quote! {
            let mut #count: usize = 0;
            #offsets_decls
            #validity_decls
        }));
        let validity_freeze = shape_freeze_validity_bitmaps(shape, &layers, pa_root);
        let offsets_freeze = shape_freeze_offsets_buffers(&layers, pa_root);
        self.freezes.push(PostScanOp::new(quote! {
            #validity_freeze
            #offsets_freeze
        }));

        let child_prefix = prefix.with_group(shape, &layers);
        let item = idents::tuple_item(self.ident_scope, group_idx);
        let tuple = if shape.inner_access.has_option() {
            InputAccess::Optional(access_chain_to_option_ref(
                &quote! { #item },
                &shape.inner_access,
            ))
        } else {
            InputAccess::Required(access_chain_to_ref(&quote! { #item }, &shape.inner_access).expr)
        };
        let children = self.build_children(elements, &tuple, None, &child_prefix);
        let leaf_body = |vec_bind: &TokenStream| {
            quote! {
                for #item in #vec_bind.iter() {
                    #children
                    #count += 1;
                }
            }
        };
        emitter.row_push(&leaf_body, &quote! { #count })
    }

    fn build_children(
        &mut self,
        elements: &NonEmpty<TupleNode>,
        tuple: &InputAccess,
        replay_tuple: Option<&ReplayTupleAccess>,
        prefix: &SharedListStack,
    ) -> TokenStream {
        let mut pushes = Vec::with_capacity(elements.len());
        for node in elements.iter() {
            let input = project_child(self.ident_scope, tuple, node.step());
            let replay_child =
                replay_tuple.map(|tuple| project_required_child(&tuple.0, node.step()));
            let push = match node.kind() {
                TupleNodeKind::Leaf(common) => {
                    let replay_input = replay_child.map(ReplayTupleAccess);
                    self.build_terminal(
                        common,
                        node.wrapper_shape(),
                        &input,
                        replay_input.as_ref(),
                        prefix,
                    )
                }
                TupleNodeKind::Tuple(children) => {
                    let replay_input =
                        replay_child.map(|child| ReplayTupleAccess(quote! { &(#child) }));
                    self.build_group(node.wrapper_shape(), children, input, replay_input, prefix)
                }
            };
            pushes.push(push);
        }
        quote! { #(#pushes)* }
    }

    fn build_terminal(
        &mut self,
        common: &ColumnCommon,
        wrapper: &WrapperShape,
        input: &InputAccess,
        replay_input: Option<&ReplayTupleAccess>,
        prefix: &SharedListStack,
    ) -> TokenStream {
        let idx = self.next_terminal;
        self.next_terminal += 1;
        let input_optional = input.is_optional();
        let effective_wrapper = inherit_parent_option(wrapper, input_optional);
        let input_expr = input.expr();

        match common.leaf_spec().route() {
            TerminalLeafRoute::Primitive(leaf) => {
                if let Some(push) = self.try_collect_replayed_primitive(
                    leaf,
                    &effective_wrapper,
                    replay_input,
                    idx,
                    common.name(),
                ) {
                    return push;
                }
                self.flush_replay_lane();
                let copied_access = (input_optional
                    && leaf.is_copy()
                    && matches!(wrapper, WrapperShape::Leaf(shape) if shape.is_bare()))
                .then(|| quote! { (#input_expr).copied() });
                let access = copied_access.as_ref().unwrap_or(input_expr);
                let input_rows_exact = idents::input_rows_exact(self.ident_scope);
                let ctx = LeafCtx {
                    base: BaseCtx {
                        access,
                        row_capacity: self.row_capacity,
                        sink: self.sink,
                        idx,
                        name: common.name(),
                    },
                    primitive_list_plan: crate::codegen::planner::PrimitiveListPolicy::for_wrapper(
                        leaf,
                        &effective_wrapper,
                        crate::codegen::planner::RowReplayCapability::Unavailable,
                    ),
                    row_replay: None,
                    cardinality: if prefix.is_empty()
                        && matches!(&effective_wrapper, WrapperShape::Leaf(_))
                    {
                        LeafCardinality::InputRows
                    } else {
                        LeafCardinality::Dynamic
                    },
                    ident_scope: self.ident_scope,
                    input_rows_exact: &input_rows_exact,
                    decimal128_encode_trait: &self.config.runtime.decimal128_encode,
                    paths: &self.config.external_paths,
                };
                let option_receiver = (input_optional && copied_access.is_none())
                    .then_some(crate::codegen::type_registry::PrimitiveExprReceiver::RefRef);
                let encoder = build_encoder_with_option_receiver(
                    leaf,
                    &effective_wrapper,
                    &ctx,
                    option_receiver,
                );
                self.collect_primitive(encoder, common.name(), idx, prefix)
            }
            TerminalLeafRoute::Nested(nested) => {
                self.flush_replay_lane();
                let ty = nested_type_path(nested);
                let name_policy = NestedNamePolicy::Field;
                let ctx = NestedLeafCtx {
                    base: BaseCtx {
                        access: input_expr,
                        row_capacity: self.row_capacity,
                        sink: self.sink,
                        idx,
                        name: common.name(),
                    },
                    name_policy: &name_policy,
                    ty: &ty,
                    columnar_trait: &self.config.runtime.columnar,
                    columnar_spec_trait: &self.config.runtime.columnar_spec,
                    paths: &self.config.external_paths,
                };
                let prefix_shape = prefix.shape();
                let lifecycle = prefix_shape.as_ref().map_or_else(
                    || build_nested_encoder(&effective_wrapper, &ctx),
                    |shape| {
                        build_nested_encoder_with_prefix(
                            &effective_wrapper,
                            SharedListPrefix {
                                shape,
                                layers: &prefix.layers,
                                ident_scope: self.ident_scope,
                            },
                            &ctx,
                        )
                    },
                );
                self.collect_plan(lifecycle)
            }
        }
    }

    fn collect_primitive(
        &mut self,
        plan: SeriesPlan,
        name: &str,
        idx: usize,
        prefix: &SharedListStack,
    ) -> TokenStream {
        let SeriesPlan {
            init,
            scan,
            post_scan,
            series,
            naming: _,
        } = plan;
        self.init.extend(init);
        if !post_scan.is_empty() {
            self.finish.push(FinishGroup::inline(post_scan, Vec::new()));
        }
        let wrapped = self.wrap_shared_prefix(series, idx, prefix);
        let sink = self.sink;
        let output_series = idents::tuple_output_series(self.ident_scope);
        let output_named = idents::tuple_output_named(self.ident_scope);
        self.finish.push(FinishGroup::scoped(
            Vec::new(),
            vec![EmitOp::new(quote! {
                let #output_series = #wrapped;
                let #output_named = #output_series.with_name(#name.into());
                #sink.push(#output_named.into())?;
            })],
        ));
        scan.into_tokens()
    }

    fn collect_replayed_primitive(&mut self, plan: SeriesPlan, name: &str) {
        let SeriesPlan {
            init,
            scan,
            post_scan,
            series,
            naming,
        } = plan;
        assert!(
            post_scan.is_empty(),
            "a replayed static tuple terminal must have no prior completion work",
        );
        assert_eq!(
            naming,
            SeriesNameState::AlreadyNamed,
            "a replayed static tuple terminal must be a named scalar series",
        );
        self.replay_lane.push(ReplayedPrimitive {
            init,
            scan,
            series,
            name: name.to_owned(),
        });
        if self.replay_lane.len() == STATIC_TUPLE_REPLAY_LANE_WIDTH {
            self.flush_replay_lane();
        }
    }

    fn flush_replay_lane(&mut self) {
        if self.replay_lane.is_empty() {
            return;
        }

        let row = self.row;
        let replay = self.replay;
        let sink = self.sink;
        let output_series = idents::tuple_output_series(self.ident_scope);
        let output_named = idents::tuple_output_named(self.ident_scope);
        let mut init = Vec::new();
        let mut scan = Vec::new();
        let mut outputs = Vec::new();
        for terminal in ::core::mem::take(&mut self.replay_lane) {
            init.extend(terminal.init);
            scan.push(terminal.scan);
            let series = terminal.series;
            let name = syn::LitStr::new(&terminal.name, proc_macro2::Span::call_site());
            outputs.push(EmitOp::new(quote! {{
                let #output_series = #series;
                let #output_named = #output_series.with_name(#name.into());
                #sink.push(#output_named.into())?;
            }}));
        }
        self.finish.push(FinishGroup::scoped(
            vec![PostScanOp::replay_rows(quote! {
                #(#init)*
                for #row in #replay {
                    #(#scan)*
                }
            })],
            outputs,
        ));
    }

    fn collect_plan(&mut self, plan: EncodePlan) -> TokenStream {
        let (init, scan, finish) = plan.into_parts();
        self.init.extend(init);
        self.finish.extend(finish);
        quote! { #(#scan)* }
    }

    fn wrap_shared_prefix(
        &self,
        series: TokenStream,
        idx: usize,
        prefix: &SharedListStack,
    ) -> TokenStream {
        let Some(shape) = prefix.shape() else {
            return series;
        };
        let pp = self.config.external_paths.prelude();
        let pa_root = self.config.external_paths.polars_arrow_root();
        let inner = idents::tuple_prefix_inner_series(self.ident_scope, idx);
        let rechunked = idents::tuple_prefix_rechunked(self.ident_scope, idx);
        let chunk = idents::tuple_prefix_chunk(self.ident_scope, idx);
        let logical_dtype = idents::tuple_logical_dtype(self.ident_scope);
        let wraps = shape_layer_wraps_clone(&shape, &prefix.layers);
        let arr_id_for_layer = |layer| idents::tuple_prefix_list_arr(self.ident_scope, idx, layer);
        let stack = shape_assemble_list_stack(
            quote! { #chunk },
            quote! { #chunk.dtype().clone() },
            &wraps,
            quote! { #logical_dtype },
            pp,
            pa_root,
            &arr_id_for_layer,
        );
        quote! {{
            let #inner: #pp::Series = #series;
            let #logical_dtype: #pp::DataType = #inner.dtype().clone();
            let #rechunked = #inner.rechunk();
            let #chunk: #pp::ArrayRef = #rechunked.chunks()[0].clone();
            #stack
        }}
    }
}

pub(in crate::codegen) fn build_tuple_field_emit(
    field: &TupleField,
    params: TupleFieldEmitParams<'_>,
) -> TupleFieldEmit {
    let TupleFieldEmitParams {
        config,
        ident_scope,
        terminal_start,
        group_start,
        row,
        replay,
        static_tuple_plan,
        row_capacity,
        sink,
    } = params;
    let root = crate::codegen::source_access::field_source_access(field.source(), row);
    let replay_root = static_tuple_plan
        .replays_root(field)
        .then(|| ReplayTupleAccess(quote! { &(#root) }));
    let mut builder = TupleBuilder {
        config,
        ident_scope,
        row,
        replay,
        row_capacity,
        sink,
        static_tuple_plan,
        next_terminal: terminal_start,
        next_group: group_start,
        terminal_start,
        group_start,
        init: Vec::new(),
        freezes: Vec::new(),
        finish: Vec::new(),
        replay_lane: Vec::new(),
    };
    let push = builder.build_group(
        field.wrapper_shape(),
        field.elements(),
        InputAccess::Required(root),
        replay_root,
        &SharedListStack::empty(),
    );
    debug_assert!(builder.replay_lane.is_empty());
    builder
        .finish
        .insert(0, FinishGroup::inline(builder.freezes, Vec::new()));

    TupleFieldEmit {
        plan: EncodePlan::new(builder.init, vec![ScanOp::new(push)], builder.finish),
        terminal_count: builder.next_terminal - builder.terminal_start,
        group_count: builder.next_group - builder.group_start,
    }
}

fn inherit_parent_option(wrapper: &WrapperShape, inherited: bool) -> WrapperShape {
    if !inherited {
        return wrapper.clone();
    }
    match wrapper {
        WrapperShape::Leaf(shape) => {
            WrapperShape::Leaf(LeafShape::from_access(shape.access().prepend_option()))
        }
        WrapperShape::Vec(shape) => {
            let mut shape = shape.clone();
            shape.layers[0].access = shape.layers[0].access.prepend_option();
            WrapperShape::Vec(shape)
        }
    }
}

fn project_child(
    ident_scope: GeneratedIdentScope<'_>,
    tuple: &InputAccess,
    step: TupleProjectionStep,
) -> InputAccess {
    match tuple {
        InputAccess::Required(tuple) => InputAccess::Required(project_required_child(tuple, step)),
        InputAccess::Optional(tuple) => {
            let parameter = idents::tuple_proj_param(ident_scope);
            let child = project_required_child(&quote! { #parameter }, step);
            InputAccess::Optional(quote! {
                (#tuple).map(|#parameter| &(#child))
            })
        }
    }
}

fn project_required_child(tuple: &TokenStream, step: TupleProjectionStep) -> TokenStream {
    let index = syn::Index::from(step.index);
    let mut child = quote! { (*(#tuple)).#index };
    for _ in 0..step.outer_smart_ptr_depth {
        child = quote! { (*(#child)) };
    }
    child
}

fn nested_type_path(nested: NestedLeaf<'_>) -> TokenStream {
    match nested {
        NestedLeaf::Struct(ty) => struct_type_tokens(ty),
        NestedLeaf::Generic(id) => quote! { #id },
    }
}

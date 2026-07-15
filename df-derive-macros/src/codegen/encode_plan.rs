//! Typed execution plan for generated column encoding.
//!
//! The source iterator still has one shared scan. Completion work remains in
//! ordered groups because changing the relative order or scope of freezes,
//! materialization, validation, and sink writes can change both semantics and
//! generated-code performance.

use proc_macro2::TokenStream;
use quote::{ToTokens, quote};

macro_rules! token_op {
    ($name:ident) => {
        pub(in crate::codegen) struct $name(TokenStream);

        impl $name {
            pub(in crate::codegen) const fn new(tokens: TokenStream) -> Self {
                Self(tokens)
            }
        }

        impl ToTokens for $name {
            fn to_tokens(&self, tokens: &mut TokenStream) {
                tokens.extend(self.0.clone());
            }
        }
    };
}

token_op!(InitOp);
token_op!(ScanOp);
token_op!(EmitOp);

impl ScanOp {
    pub(in crate::codegen) fn into_tokens(self) -> TokenStream {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::codegen) struct PlanRequirements {
    row_replay: bool,
}

impl PlanRequirements {
    pub(in crate::codegen) const fn requires_row_replay(self) -> bool {
        self.row_replay
    }

    const fn union(self, other: Self) -> Self {
        Self {
            row_replay: self.row_replay || other.row_replay,
        }
    }
}

pub(in crate::codegen) struct PostScanOp {
    tokens: TokenStream,
    requirements: PlanRequirements,
}

impl PostScanOp {
    pub(in crate::codegen) fn new(tokens: TokenStream) -> Self {
        Self {
            tokens,
            requirements: PlanRequirements::default(),
        }
    }

    pub(in crate::codegen) const fn replay_rows(tokens: TokenStream) -> Self {
        Self {
            tokens,
            requirements: PlanRequirements { row_replay: true },
        }
    }

    const fn requirements(&self) -> PlanRequirements {
        self.requirements
    }
}

impl ToTokens for PostScanOp {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.extend(self.tokens.clone());
    }
}

/// An ordered completion unit.
///
/// `Inline` retains bindings for later groups. `Scoped` deliberately emits a
/// block, matching the lifetime and identifier isolation of the former
/// hand-written builder blocks.
pub(in crate::codegen) enum FinishGroup {
    Inline {
        post_scan: Vec<PostScanOp>,
        emit: Vec<EmitOp>,
    },
    Scoped {
        post_scan: Vec<PostScanOp>,
        emit: Vec<EmitOp>,
    },
}

impl FinishGroup {
    pub(in crate::codegen) const fn inline(post_scan: Vec<PostScanOp>, emit: Vec<EmitOp>) -> Self {
        Self::Inline { post_scan, emit }
    }

    pub(in crate::codegen) const fn scoped(post_scan: Vec<PostScanOp>, emit: Vec<EmitOp>) -> Self {
        Self::Scoped { post_scan, emit }
    }

    fn requirements(&self) -> PlanRequirements {
        let post_scan = match self {
            Self::Inline { post_scan, .. } | Self::Scoped { post_scan, .. } => post_scan,
        };
        post_scan
            .iter()
            .fold(PlanRequirements::default(), |requirements, operation| {
                requirements.union(operation.requirements())
            })
    }
}

impl ToTokens for FinishGroup {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Self::Inline { post_scan, emit } => {
                tokens.extend(quote! {
                    #(#post_scan)*
                    #(#emit)*
                });
            }
            Self::Scoped { post_scan, emit } => {
                tokens.extend(quote! {{
                    #(#post_scan)*
                    #(#emit)*
                }});
            }
        }
    }
}

/// A primitive encoder before its schema slot is acquired and committed.
pub(in crate::codegen) struct SeriesPlan {
    pub init: Vec<InitOp>,
    pub scan: ScanOp,
    pub post_scan: Vec<PostScanOp>,
    pub materialize: TokenStream,
    pub output_slot: syn::Ident,
}

impl SeriesPlan {
    pub(in crate::codegen) fn leaf(
        init: Vec<TokenStream>,
        scan: TokenStream,
        materialize: TokenStream,
        output_slot: syn::Ident,
    ) -> Self {
        Self {
            init: init.into_iter().map(InitOp::new).collect(),
            scan: ScanOp::new(scan),
            post_scan: Vec::new(),
            materialize,
            output_slot,
        }
    }

    pub(in crate::codegen) const fn new(
        init: Vec<InitOp>,
        scan: ScanOp,
        post_scan: Vec<PostScanOp>,
        materialize: TokenStream,
        output_slot: syn::Ident,
    ) -> Self {
        Self {
            init,
            scan,
            post_scan,
            materialize,
            output_slot,
        }
    }
}

/// The complete generated program for one field or the whole struct.
#[derive(Default)]
pub(in crate::codegen) struct EncodePlan {
    init: Vec<InitOp>,
    scan: Vec<ScanOp>,
    finish: Vec<FinishGroup>,
    requirements: PlanRequirements,
}

impl EncodePlan {
    pub(in crate::codegen) fn new(
        init: Vec<InitOp>,
        scan: Vec<ScanOp>,
        finish: Vec<FinishGroup>,
    ) -> Self {
        let requirements = finish
            .iter()
            .fold(PlanRequirements::default(), |requirements, group| {
                requirements.union(group.requirements())
            });
        Self {
            init,
            scan,
            finish,
            requirements,
        }
    }

    pub(in crate::codegen) fn append(&mut self, mut other: Self) {
        self.init.append(&mut other.init);
        self.scan.append(&mut other.scan);
        self.finish.append(&mut other.finish);
        self.requirements = self.requirements.union(other.requirements);
    }

    pub(in crate::codegen) const fn requirements(&self) -> PlanRequirements {
        self.requirements
    }

    pub(in crate::codegen) fn into_parts(self) -> (Vec<InitOp>, Vec<ScanOp>, Vec<FinishGroup>) {
        (self.init, self.scan, self.finish)
    }

    #[cfg(test)]
    pub(in crate::codegen) fn row_replay_group_count(&self) -> usize {
        self.finish
            .iter()
            .filter(|group| group.requirements().requires_row_replay())
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proc_macro2::TokenTree;

    #[test]
    fn finish_groups_keep_order_and_scope_explicit() {
        let inline = FinishGroup::inline(
            vec![PostScanOp::new(quote! { freeze(); })],
            vec![EmitOp::new(quote! { emit_one(); })],
        );
        let scoped = FinishGroup::scoped(
            vec![PostScanOp::new(quote! { materialize(); })],
            vec![EmitOp::new(quote! { emit_two(); })],
        );

        let rendered = quote! { #inline #scoped };
        let trees: Vec<TokenTree> = rendered.into_iter().collect();
        assert!(matches!(trees.last(), Some(TokenTree::Group(_))));
        assert_eq!(
            quote! { #inline #scoped }.to_string(),
            quote! { freeze(); emit_one(); { materialize(); emit_two(); } }.to_string(),
        );
    }

    #[test]
    fn row_replay_is_derived_from_post_scan_operations() {
        let plain = EncodePlan::new(
            Vec::new(),
            Vec::new(),
            vec![FinishGroup::inline(
                vec![PostScanOp::new(quote! { finish(); })],
                Vec::new(),
            )],
        );
        assert!(!plain.requirements().requires_row_replay());

        let replay = EncodePlan::new(
            Vec::new(),
            Vec::new(),
            vec![FinishGroup::scoped(
                vec![PostScanOp::replay_rows(quote! { replay(); })],
                Vec::new(),
            )],
        );
        assert!(replay.requirements().requires_row_replay());
    }

    #[test]
    fn append_preserves_phase_order_and_replay_requirements() {
        let mut plan = EncodePlan::new(
            vec![InitOp::new(quote! { init_one(); })],
            vec![ScanOp::new(quote! { scan_one(); })],
            vec![FinishGroup::inline(
                vec![PostScanOp::new(quote! { finish_one(); })],
                Vec::new(),
            )],
        );
        plan.append(EncodePlan::new(
            vec![InitOp::new(quote! { init_two(); })],
            vec![ScanOp::new(quote! { scan_two(); })],
            vec![FinishGroup::scoped(
                vec![PostScanOp::replay_rows(quote! { finish_two(); })],
                Vec::new(),
            )],
        ));

        assert!(plan.requirements().requires_row_replay());
        assert_eq!(plan.row_replay_group_count(), 1);
        let (init, scan, finish) = plan.into_parts();
        assert_eq!(
            quote! { #(#init)* }.to_string(),
            quote! { init_one(); init_two(); }.to_string(),
        );
        assert_eq!(
            quote! { #(#scan)* }.to_string(),
            quote! { scan_one(); scan_two(); }.to_string(),
        );
        assert_eq!(
            quote! { #(#finish)* }.to_string(),
            quote! { finish_one(); { finish_two(); } }.to_string(),
        );
    }
}

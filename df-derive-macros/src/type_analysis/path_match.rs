use syn::{PathArguments, TypePath};

pub(super) struct PathView<'a> {
    path: &'a syn::Path,
}

impl<'a> PathView<'a> {
    pub(super) fn from_type_path(type_path: &'a TypePath) -> Option<Self> {
        type_path.qself.is_none().then_some(Self {
            path: &type_path.path,
        })
    }

    pub(super) fn exact_no_args(&self, segments: &[&str]) -> bool {
        self.path.segments.len() == segments.len()
            && self
                .path
                .segments
                .iter()
                .zip(segments)
                .all(|(segment, expected)| {
                    segment.ident == *expected && matches!(segment.arguments, PathArguments::None)
                })
    }

    pub(super) fn exact_with_leaf_args(&self, segments: &[&str]) -> bool {
        self.path.segments.len() == segments.len()
            && self.path.segments.iter().zip(segments).enumerate().all(
                |(idx, (segment, expected))| {
                    segment.ident == *expected
                        && (idx + 1 == segments.len()
                            || matches!(segment.arguments, PathArguments::None))
                },
            )
    }

    pub(super) fn prefix_no_args(&self, prefix: &[&str]) -> bool {
        self.path.segments.len() > prefix.len()
            && self
                .path
                .segments
                .iter()
                .zip(prefix)
                .all(|(segment, expected)| {
                    segment.ident == *expected && matches!(segment.arguments, PathArguments::None)
                })
    }

    pub(super) fn leaf(&self) -> Option<&'a syn::PathSegment> {
        self.path.segments.last()
    }

    pub(super) fn len(&self) -> usize {
        self.path.segments.len()
    }
}

pub(super) fn path_is_exact_no_args(type_path: &TypePath, segments: &[&str]) -> bool {
    PathView::from_type_path(type_path).is_some_and(|path| path.exact_no_args(segments))
}

pub(super) fn path_is_exact_with_leaf_args(type_path: &TypePath, segments: &[&str]) -> bool {
    PathView::from_type_path(type_path).is_some_and(|path| path.exact_with_leaf_args(segments))
}

pub(super) fn path_prefix_is_no_args(type_path: &TypePath, prefix: &[&str]) -> bool {
    PathView::from_type_path(type_path).is_some_and(|path| path.prefix_no_args(prefix))
}

pub(super) fn wrapper_path_matches(type_path: &TypePath, bare: &str, qualified: &[&str]) -> bool {
    PathView::from_type_path(type_path).is_some_and(|path| {
        path.exact_with_leaf_args(&[bare]) || path.exact_with_leaf_args(qualified)
    })
}

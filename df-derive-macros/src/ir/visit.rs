use super::LeafSpec;

impl LeafSpec {
    pub fn walk_terminal_leaves<'a>(&'a self, f: &mut impl FnMut(&'a Self)) {
        match self {
            Self::Tuple(elements) => {
                for element in elements {
                    element.leaf_spec.walk_terminal_leaves(f);
                }
            }
            leaf => f(leaf),
        }
    }
}

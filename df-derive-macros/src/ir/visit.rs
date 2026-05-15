use super::{LeafSpec, TupleElement};

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

    pub fn any_terminal_leaf(&self, mut pred: impl FnMut(&Self) -> bool) -> bool {
        let mut found = false;
        self.walk_terminal_leaves(&mut |leaf| {
            if pred(leaf) {
                found = true;
            }
        });
        found
    }

    pub fn walk_tuple_elements<'a>(&'a self, f: &mut impl FnMut(&'a TupleElement)) {
        if let Self::Tuple(elements) = self {
            for element in elements {
                f(element);
                element.leaf_spec.walk_tuple_elements(f);
            }
        }
    }
}

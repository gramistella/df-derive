/// One transparent access step peeled while analyzing a field type, excluding
/// `Vec` itself. The normalized wrapper shape stores these steps at the
/// boundary where they occur so codegen can dereference smart pointers before
/// walking the next wrapper layer instead of blindly dereferencing at the leaf.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessStep {
    Option,
    SmartPtr,
}

/// Transparent access steps between two semantic wrapper boundaries: from the
/// field access to a `Vec`, from one `Vec` item to the next `Vec`, or from the
/// innermost `Vec` item to the leaf.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AccessChain {
    pub steps: Vec<AccessStep>,
}

impl AccessChain {
    pub const fn empty() -> Self {
        Self { steps: Vec::new() }
    }

    pub fn option_layers(&self) -> usize {
        self.steps
            .iter()
            .filter(|step| matches!(step, AccessStep::Option))
            .count()
    }

    pub const fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    pub fn is_single_plain_option(&self) -> bool {
        self.steps == [AccessStep::Option]
    }

    pub fn is_only_options(&self) -> bool {
        self.steps
            .iter()
            .all(|step| matches!(step, AccessStep::Option))
    }
}

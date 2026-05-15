use crate::ir::{LeafSpec, NumericKind};
use crate::type_analysis::{AnalyzedBase, RawWrapper};

use super::diagnostics;

pub(super) fn parse_as_binary_shape(
    field: &syn::Field,
    field_display_name: &str,
    base: &AnalyzedBase,
    wrappers: &[RawWrapper],
) -> Result<(LeafSpec, Vec<RawWrapper>), syn::Error> {
    if matches!(base, AnalyzedBase::CowBytes | AnalyzedBase::BorrowedBytes) {
        return Ok((LeafSpec::Binary, wrappers.to_vec()));
    }
    if matches!(base, AnalyzedBase::BorrowedSlice) {
        return Err(diagnostics::binary_borrowed_slice(
            field,
            field_display_name,
        ));
    }
    if matches!(base, AnalyzedBase::CowSlice) {
        return Err(diagnostics::binary_cow_slice(field, field_display_name));
    }
    if !matches!(base, AnalyzedBase::Numeric(NumericKind::U8)) {
        return Err(diagnostics::binary_wrong_base(field, field_display_name));
    }
    match wrappers.last() {
        None => Err(diagnostics::bare_binary_u8(field, field_display_name)),
        Some(RawWrapper::Option) => {
            if wrappers.len() == 1 {
                Err(diagnostics::bare_binary_u8(field, field_display_name))
            } else {
                Err(diagnostics::binary_inner_option(field, field_display_name))
            }
        }
        Some(RawWrapper::Vec) => {
            let mut trimmed = wrappers.to_vec();
            trimmed.pop();
            Ok((LeafSpec::Binary, trimmed))
        }
        Some(RawWrapper::SmartPtr) => {
            Err(diagnostics::binary_wrong_base(field, field_display_name))
        }
    }
}

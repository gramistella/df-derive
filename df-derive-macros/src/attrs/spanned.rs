use proc_macro2::Span;

/// Parsed value paired with the source span for duplicate/conflict diagnostics.
pub struct Spanned<T> {
    /// Parsed value.
    pub value: T,
    /// Span of the attribute key or syntax that produced the value.
    pub span: Span,
}

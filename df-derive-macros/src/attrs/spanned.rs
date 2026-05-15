use proc_macro2::Span;

#[derive(Clone, Copy, Debug)]
pub struct Spanned<T> {
    pub value: T,
    pub span: Span,
}

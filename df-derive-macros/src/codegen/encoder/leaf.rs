//! Primitive leaf builders and shared decl helpers.
//!
//! Polars folds every nested None into one validity bit; deeper Option stacks
//! reach these builders through a collapsed single-Option access.

use crate::codegen::type_registry::{PrimitiveExprReceiver, ScalarTransform};
use crate::ir::{DateTimeUnit, DurationSource, NumericKind, StringyBase};
use proc_macro2::TokenStream;
use quote::quote;

use super::idents;
use super::{LeafCardinality, LeafCtx};

#[derive(Clone, Copy)]
pub(super) enum LeafArmKind {
    Bare,
    Option {
        some_receiver: PrimitiveExprReceiver,
    },
}

pub(super) struct LeafArm {
    pub decls: Vec<TokenStream>,
    pub push: TokenStream,
    pub series: TokenStream,
}

pub(super) fn vec_decl(buf: &syn::Ident, elem: &TokenStream, ctx: &LeafCtx<'_>) -> TokenStream {
    let row_capacity = ctx.base.row_capacity;
    quote! {
        let mut #buf: ::std::vec::Vec<#elem> =
            ::std::vec::Vec::with_capacity(#row_capacity);
    }
}

pub(super) fn mb_decl(ident: &syn::Ident, pa_root: &TokenStream, ctx: &LeafCtx<'_>) -> TokenStream {
    let row_capacity = ctx.base.row_capacity;
    quote! {
        let mut #ident: #pa_root::bitmap::MutableBitmap =
            #pa_root::bitmap::MutableBitmap::with_capacity(#row_capacity);
    }
}

fn optional_view_validity_decl(
    ident: &syn::Ident,
    pa_root: &TokenStream,
    ctx: &LeafCtx<'_>,
) -> TokenStream {
    match ctx.cardinality {
        LeafCardinality::InputRows => {
            let row_capacity = ctx.base.row_capacity;
            let input_rows_exact = ctx.input_rows_exact;
            quote! {
                let mut #ident: #pa_root::bitmap::MutableBitmap =
                    if #input_rows_exact {
                        #pa_root::bitmap::MutableBitmap::from_len_set(#row_capacity)
                    } else {
                        #pa_root::bitmap::MutableBitmap::with_capacity(#row_capacity)
                    };
            }
        }
        LeafCardinality::Dynamic => mb_decl(ident, pa_root, ctx),
    }
}

/// Row-aligned leaves update prefilled validity in place, growing when a
/// source iterator understates its exact size hint. Dynamic leaves append.
fn update_optional_view_validity(
    cardinality: LeafCardinality,
    validity: &syn::Ident,
    values: &syn::Ident,
    valid: bool,
) -> TokenStream {
    match cardinality {
        LeafCardinality::InputRows if valid => {
            quote! {
                if #values.len() == #validity.len() {
                    #validity.push(true);
                }
            }
        }
        LeafCardinality::InputRows => {
            quote! {
                if #values.len() == #validity.len() {
                    #validity.push(true);
                }
                #validity.set(#values.len(), false);
            }
        }
        LeafCardinality::Dynamic => quote! { #validity.push(#valid); },
    }
}

pub(super) fn mbva_decl(buf: &syn::Ident, pa_root: &TokenStream, ctx: &LeafCtx<'_>) -> TokenStream {
    let row_capacity = ctx.base.row_capacity;
    quote! {
        let mut #buf: #pa_root::array::MutableBinaryViewArray<str> =
            #pa_root::array::MutableBinaryViewArray::<str>::with_capacity(#row_capacity);
    }
}

pub(super) fn mbva_bytes_decl(
    buf: &syn::Ident,
    pa_root: &TokenStream,
    ctx: &LeafCtx<'_>,
) -> TokenStream {
    let row_capacity = ctx.base.row_capacity;
    quote! {
        let mut #buf: #pa_root::array::MutableBinaryViewArray<[u8]> =
            #pa_root::array::MutableBinaryViewArray::<[u8]>::with_capacity(#row_capacity);
    }
}

/// `MutableBitmap -> Option<Bitmap>` preserves the no-null fast path.
pub(super) fn validity_into_option(validity: &syn::Ident, pa_root: &TokenStream) -> TokenStream {
    quote! {
        ::std::convert::Into::<::std::option::Option<#pa_root::bitmap::Bitmap>>::into(
            #validity,
        )
    }
}

pub(super) fn string_chunked_series(
    name: &str,
    arr_expr: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    quote! {
        #pp::IntoSeries::into_series(
            #pp::StringChunked::with_chunk(#name.into(), { #arr_expr }),
        )
    }
}

pub(super) fn binary_chunked_series(
    name: &str,
    arr_expr: &TokenStream,
    pp: &TokenStream,
) -> TokenStream {
    quote! {
        #pp::IntoSeries::into_series(
            #pp::BinaryChunked::with_chunk(#name.into(), { #arr_expr }),
        )
    }
}

pub(super) fn named_from_buf(name: &str, buf: &syn::Ident, pp: &TokenStream) -> TokenStream {
    quote! { <#pp::Series as #pp::NamedFrom<_, _>>::new(#name.into(), &#buf) }
}

pub(super) fn numeric_leaf(ctx: &LeafCtx<'_>, kind: NumericKind, arm: LeafArmKind) -> LeafArm {
    let info = crate::codegen::type_registry::numeric_info_for(kind, ctx.paths);
    let buf = idents::primitive_buf(ctx.base.idx);
    let validity = idents::primitive_validity(ctx.base.idx);
    let native = &info.native;
    let chunked = &info.chunked;
    let access = ctx.base.access;
    let name = ctx.base.name;
    let pp = ctx.paths.prelude();
    let pa_root = ctx.paths.polars_arrow_root();

    match arm {
        LeafArmKind::Bare => {
            let bare_value = if kind.is_nonzero() || info.widen_from.is_some() {
                crate::codegen::type_registry::numeric_stored_value(
                    kind,
                    quote! { #access },
                    native,
                )
            } else {
                quote! { #access }
            };
            let bare_push = quote! { #buf.push({ #bare_value }); };
            let bare_series = quote! {
                #pp::IntoSeries::into_series(#chunked::from_vec(#name.into(), #buf))
            };
            LeafArm {
                decls: vec![vec_decl(&buf, native, ctx)],
                push: bare_push,
                series: bare_series,
            }
        }
        LeafArmKind::Option { .. } => {
            let v = idents::leaf_value();
            let some_push_value =
                crate::codegen::type_registry::numeric_stored_value(kind, quote! { #v }, native);
            let option_push = quote! {
                match #access {
                    ::std::option::Option::Some(#v) => {
                        #buf.push(#some_push_value);
                        #validity.push(true);
                    }
                    ::std::option::Option::None => {
                        #buf.push(<#native as ::std::default::Default>::default());
                        #validity.push(false);
                    }
                }
            };
            let valid_opt = validity_into_option(&validity, pa_root);
            let option_series = quote! {{
                let arr = #pa_root::array::PrimitiveArray::<#native>::new(
                    <#native as #pa_root::types::NativeType>::PRIMITIVE.into(),
                    #buf.into(),
                    #valid_opt,
                );
                #pp::IntoSeries::into_series(#chunked::with_chunk(#name.into(), arr))
            }};
            LeafArm {
                decls: vec![
                    vec_decl(&buf, native, ctx),
                    mb_decl(&validity, pa_root, ctx),
                ],
                push: option_push,
                series: option_series,
            }
        }
    }
}

/// `String` leaf uses `MutableBinaryViewArray<str>` to avoid a `Vec<&str>` pass.
pub(super) fn string_leaf(ctx: &LeafCtx<'_>, arm: LeafArmKind) -> LeafArm {
    let buf = idents::primitive_buf(ctx.base.idx);
    let validity = idents::primitive_validity(ctx.base.idx);
    let access = ctx.base.access;
    let name = ctx.base.name;
    let pp = ctx.paths.prelude();
    let pa_root = ctx.paths.polars_arrow_root();

    match arm {
        LeafArmKind::Bare => {
            let bare_push = quote! { #buf.push_value_ignore_validity((#access).as_str()); };
            let bare_series = string_chunked_series(name, &quote! { #buf.freeze() }, pp);
            LeafArm {
                decls: vec![mbva_decl(&buf, pa_root, ctx)],
                push: bare_push,
                series: bare_series,
            }
        }
        LeafArmKind::Option { .. } => {
            let v = idents::leaf_value();
            let valid = update_optional_view_validity(ctx.cardinality, &validity, &buf, true);
            let null = update_optional_view_validity(ctx.cardinality, &validity, &buf, false);
            let option_push = quote! {
                match &(#access) {
                    ::std::option::Option::Some(#v) => {
                        #valid
                        #buf.push_value_ignore_validity(#v.as_str());
                    }
                    ::std::option::Option::None => {
                        #null
                        #buf.push_value_ignore_validity("");
                    }
                }
            };
            let valid_opt = validity_into_option(&validity, pa_root);
            let series = string_chunked_series(
                name,
                &quote! { #buf.freeze().with_validity(#valid_opt) },
                pp,
            );
            let option_series = quote! {{
                #validity.resize(#buf.len(), true);
                #series
            }};
            LeafArm {
                decls: vec![
                    mbva_decl(&buf, pa_root, ctx),
                    optional_view_validity_decl(&validity, pa_root, ctx),
                ],
                push: option_push,
                series: option_series,
            }
        }
    }
}

/// `Binary` leaf for `#[df_derive(as_binary)]` over a `Vec<u8>` shape.
pub(super) fn binary_leaf(ctx: &LeafCtx<'_>, arm: LeafArmKind) -> LeafArm {
    let buf = idents::primitive_buf(ctx.base.idx);
    let validity = idents::primitive_validity(ctx.base.idx);
    let access = ctx.base.access;
    let name = ctx.base.name;
    let pp = ctx.paths.prelude();
    let pa_root = ctx.paths.polars_arrow_root();

    match arm {
        LeafArmKind::Bare => {
            let bytes = bytes_ref_expr(&quote! { &(#access) });
            let bare_push = quote! { #buf.push_value_ignore_validity(#bytes); };
            let bare_series = binary_chunked_series(name, &quote! { #buf.freeze() }, pp);
            LeafArm {
                decls: vec![mbva_bytes_decl(&buf, pa_root, ctx)],
                push: bare_push,
                series: bare_series,
            }
        }
        LeafArmKind::Option { .. } => {
            let v = idents::leaf_value();
            let empty = quote! { &[][..] };
            let bytes = bytes_ref_expr(&quote! { #v });
            let valid = update_optional_view_validity(ctx.cardinality, &validity, &buf, true);
            let null = update_optional_view_validity(ctx.cardinality, &validity, &buf, false);
            let option_push = quote! {
                match &(#access) {
                    ::std::option::Option::Some(#v) => {
                        #valid
                        #buf.push_value_ignore_validity(#bytes);
                    }
                    ::std::option::Option::None => {
                        #null
                        #buf.push_value_ignore_validity(#empty);
                    }
                }
            };
            let valid_opt = validity_into_option(&validity, pa_root);
            let series = binary_chunked_series(
                name,
                &quote! { #buf.freeze().with_validity(#valid_opt) },
                pp,
            );
            let option_series = quote! {{
                #validity.resize(#buf.len(), true);
                #series
            }};
            LeafArm {
                decls: vec![
                    mbva_bytes_decl(&buf, pa_root, ctx),
                    optional_view_validity_decl(&validity, pa_root, ctx),
                ],
                push: option_push,
                series: option_series,
            }
        }
    }
}

fn bytes_ref_expr(binding: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    quote! { ::core::convert::AsRef::<[u8]>::as_ref(#binding) }
}

fn bool_option_parts(
    ctx: &LeafCtx<'_>,
    buf: &syn::Ident,
    validity: &syn::Ident,
    access: &TokenStream,
    pa_root: &TokenStream,
) -> (Vec<TokenStream>, TokenStream, TokenStream) {
    match ctx.cardinality {
        LeafCardinality::InputRows => {
            let row_capacity = ctx.base.row_capacity;
            let input_rows_exact = ctx.input_rows_exact;
            let row_idx = idents::primitive_row_idx(ctx.ident_scope, ctx.base.idx);
            (
                vec![
                    quote! {
                        let mut #buf: #pa_root::bitmap::MutableBitmap =
                            if #input_rows_exact {
                                #pa_root::bitmap::MutableBitmap::from_len_zeroed(#row_capacity)
                            } else {
                                #pa_root::bitmap::MutableBitmap::with_capacity(#row_capacity)
                            };
                    },
                    quote! {
                        let mut #validity: #pa_root::bitmap::MutableBitmap =
                            if #input_rows_exact {
                                #pa_root::bitmap::MutableBitmap::from_len_set(#row_capacity)
                            } else {
                                #pa_root::bitmap::MutableBitmap::with_capacity(#row_capacity)
                            };
                    },
                    quote! { let mut #row_idx: usize = 0; },
                ],
                quote! {
                    if #row_idx == #buf.len() {
                        #buf.push(false);
                        #validity.push(true);
                    }
                    match (#access) {
                        ::std::option::Option::Some(true) => {
                            #buf.set(#row_idx, true);
                        }
                        ::std::option::Option::Some(false) => {}
                        ::std::option::Option::None => {
                            #validity.set(#row_idx, false);
                        }
                    }
                    #row_idx += 1;
                },
                quote! {
                    #buf.resize(#row_idx, false);
                    #validity.resize(#row_idx, true);
                },
            )
        }
        LeafCardinality::Dynamic => (
            vec![mb_decl(buf, pa_root, ctx), mb_decl(validity, pa_root, ctx)],
            quote! {
                match (#access) {
                    ::std::option::Option::Some(true) => {
                        #buf.push(true);
                        #validity.push(true);
                    }
                    ::std::option::Option::Some(false) => {
                        #buf.push(false);
                        #validity.push(true);
                    }
                    ::std::option::Option::None => {
                        #buf.push(false);
                        #validity.push(false);
                    }
                }
            },
            TokenStream::new(),
        ),
    }
}

/// `bool` option arms use a values bitmap plus a validity bitmap.
pub(super) fn bool_leaf(ctx: &LeafCtx<'_>, arm: LeafArmKind) -> LeafArm {
    let buf = idents::primitive_buf(ctx.base.idx);
    let validity = idents::primitive_validity(ctx.base.idx);
    let access = ctx.base.access;
    let name = ctx.base.name;
    let pp = ctx.paths.prelude();
    let pa_root = ctx.paths.polars_arrow_root();

    match arm {
        LeafArmKind::Bare => {
            let bare_push = quote! { #buf.push({ #access }); };
            let bare_series = named_from_buf(name, &buf, pp);
            LeafArm {
                decls: vec![vec_decl(&buf, &quote! { bool }, ctx)],
                push: bare_push,
                series: bare_series,
            }
        }
        LeafArmKind::Option { .. } => {
            let (decls, option_push, finish) =
                bool_option_parts(ctx, &buf, &validity, access, pa_root);
            let valid_opt = validity_into_option(&validity, pa_root);
            let option_series = quote! {{
                #finish
                let arr = #pa_root::array::BooleanArray::new(
                    #pa_root::datatypes::ArrowDataType::Boolean,
                    ::std::convert::Into::<#pa_root::bitmap::Bitmap>::into(#buf),
                    #valid_opt,
                );
                #pp::IntoSeries::into_series(
                    #pp::BooleanChunked::with_chunk(#name.into(), arr),
                )
            }};
            LeafArm {
                decls,
                push: option_push,
                series: option_series,
            }
        }
    }
}

fn mapped_push(ctx: &LeafCtx<'_>, leaf: ScalarTransform, arm: LeafArmKind) -> TokenStream {
    let buf = idents::primitive_buf(ctx.base.idx);
    let access = ctx.base.access;
    let decimal_trait = ctx.decimal128_encode_trait;
    match arm {
        LeafArmKind::Bare => {
            let mapped_bare = crate::codegen::type_registry::map_primitive_expr(
                access,
                crate::codegen::type_registry::PrimitiveExprReceiver::Place,
                leaf,
                decimal_trait,
                ctx.paths,
            );
            quote! { #buf.push({ #mapped_bare }); }
        }
        LeafArmKind::Option { some_receiver } => {
            let v = idents::leaf_value();
            let some_var = quote! { #v };
            let mapped_some = crate::codegen::type_registry::map_primitive_expr(
                &some_var,
                some_receiver,
                leaf,
                decimal_trait,
                ctx.paths,
            );
            quote! {
                match &(#access) {
                    ::std::option::Option::Some(#v) => {
                        #buf.push(::std::option::Option::Some({ #mapped_some }));
                    }
                    ::std::option::Option::None => {
                        #buf.push(::std::option::Option::None);
                    }
                }
            }
        }
    }
}

pub(super) fn decimal_leaf(
    ctx: &LeafCtx<'_>,
    precision: u8,
    scale: u8,
    arm: LeafArmKind,
) -> LeafArm {
    let buf = idents::primitive_buf(ctx.base.idx);
    let name = ctx.base.name;
    let pp = ctx.paths.prelude();
    let int128 = ctx.paths.int128_chunked();
    let p = precision as usize;
    let s = scale as usize;
    let leaf = ScalarTransform::Decimal { precision, scale };
    let push = mapped_push(ctx, leaf, arm);
    match arm {
        LeafArmKind::Bare => {
            let bare_series = quote! {{
                let ca = #int128::from_vec(#name.into(), #buf);
                #pp::IntoSeries::into_series(ca.into_decimal_unchecked(#p, #s))
            }};
            LeafArm {
                decls: vec![vec_decl(&buf, &quote! { i128 }, ctx)],
                push,
                series: bare_series,
            }
        }
        LeafArmKind::Option { .. } => {
            let option_series = quote! {{
                let ca = <#int128 as #pp::NewChunkedArray<_, _>>::from_iter_options(
                    #name.into(),
                    #buf.into_iter(),
                );
                #pp::IntoSeries::into_series(ca.into_decimal_unchecked(#p, #s))
            }};
            LeafArm {
                decls: vec![vec_decl(&buf, &quote! { ::std::option::Option<i128> }, ctx)],
                push,
                series: option_series,
            }
        }
    }
}

pub(super) fn datetime_leaf(ctx: &LeafCtx<'_>, unit: DateTimeUnit, arm: LeafArmKind) -> LeafArm {
    mapped_cast_leaf(ctx, ScalarTransform::DateTime(unit), &quote! { i64 }, arm)
}

pub(super) fn naive_datetime_leaf(
    ctx: &LeafCtx<'_>,
    unit: DateTimeUnit,
    arm: LeafArmKind,
) -> LeafArm {
    mapped_cast_leaf(
        ctx,
        ScalarTransform::NaiveDateTime(unit),
        &quote! { i64 },
        arm,
    )
}

pub(super) fn naive_date_leaf(ctx: &LeafCtx<'_>, arm: LeafArmKind) -> LeafArm {
    mapped_cast_leaf(ctx, ScalarTransform::NaiveDate, &quote! { i32 }, arm)
}

pub(super) fn naive_time_leaf(ctx: &LeafCtx<'_>, arm: LeafArmKind) -> LeafArm {
    mapped_cast_leaf(ctx, ScalarTransform::NaiveTime, &quote! { i64 }, arm)
}

pub(super) fn duration_leaf(
    ctx: &LeafCtx<'_>,
    unit: DateTimeUnit,
    source: DurationSource,
    arm: LeafArmKind,
) -> LeafArm {
    mapped_cast_leaf(
        ctx,
        ScalarTransform::Duration { unit, source },
        &quote! { i64 },
        arm,
    )
}

/// Shared mapped-scalar finish path for temporal and duration leaves.
fn mapped_cast_leaf(
    ctx: &LeafCtx<'_>,
    leaf: ScalarTransform,
    native: &TokenStream,
    arm: LeafArmKind,
) -> LeafArm {
    let buf = idents::primitive_buf(ctx.base.idx);
    let name = ctx.base.name;
    let push = mapped_push(ctx, leaf, arm);
    let dtype = leaf.dtype(ctx.paths);
    let series_new = named_from_buf(name, &buf, ctx.paths.prelude());
    let series_finish = quote! {{
        let mut s = #series_new;
        s = s.cast(&#dtype)?;
        s
    }};
    match arm {
        LeafArmKind::Bare => LeafArm {
            decls: vec![vec_decl(&buf, native, ctx)],
            push,
            series: series_finish,
        },
        LeafArmKind::Option { .. } => LeafArm {
            decls: vec![vec_decl(
                &buf,
                &quote! { ::std::option::Option<#native> },
                ctx,
            )],
            push,
            series: series_finish,
        },
    }
}

/// `as_string` uses one reusable scratch buffer plus the string-view array.
pub(super) fn as_string_leaf(ctx: &LeafCtx<'_>, arm: LeafArmKind) -> LeafArm {
    let buf = idents::primitive_buf(ctx.base.idx);
    let scratch = idents::primitive_str_scratch(ctx.base.idx);
    let validity = idents::primitive_validity(ctx.base.idx);
    let access = ctx.base.access;
    let name = ctx.base.name;
    let pp = ctx.paths.prelude();
    let pa_root = ctx.paths.polars_arrow_root();
    let scratch_decl =
        quote! { let mut #scratch: ::std::string::String = ::std::string::String::new(); };

    match arm {
        LeafArmKind::Bare => {
            let bare_push = quote! {
                {
                    use ::std::fmt::Write as _;
                    #scratch.clear();
                    ::std::write!(&mut #scratch, "{}", &(#access)).map_err(|__df_fmt_err| {
                        #pp::polars_err!(
                            ComputeError:
                            "df-derive: as_string Display formatting failed: {}",
                            __df_fmt_err,
                        )
                    })?;
                    #buf.push_value_ignore_validity(#scratch.as_str());
                }
            };
            let bare_series = string_chunked_series(name, &quote! { #buf.freeze() }, pp);
            LeafArm {
                decls: vec![mbva_decl(&buf, pa_root, ctx), scratch_decl],
                push: bare_push,
                series: bare_series,
            }
        }
        LeafArmKind::Option { .. } => {
            let v = idents::leaf_value();
            let valid = update_optional_view_validity(ctx.cardinality, &validity, &buf, true);
            let null = update_optional_view_validity(ctx.cardinality, &validity, &buf, false);
            let option_push = quote! {
                match &(#access) {
                    ::std::option::Option::Some(#v) => {
                        use ::std::fmt::Write as _;
                        #scratch.clear();
                        ::std::write!(&mut #scratch, "{}", #v).map_err(|__df_fmt_err| {
                            #pp::polars_err!(
                                ComputeError:
                                "df-derive: as_string Display formatting failed: {}",
                                __df_fmt_err,
                            )
                        })?;
                        #valid
                        #buf.push_value_ignore_validity(#scratch.as_str());
                    }
                    ::std::option::Option::None => {
                        #null
                        #buf.push_value_ignore_validity("");
                    }
                }
            };
            let valid_opt = validity_into_option(&validity, pa_root);
            let series = string_chunked_series(
                name,
                &quote! { #buf.freeze().with_validity(#valid_opt) },
                pp,
            );
            let option_series = quote! {{
                #validity.resize(#buf.len(), true);
                #series
            }};
            LeafArm {
                decls: vec![
                    mbva_decl(&buf, pa_root, ctx),
                    scratch_decl,
                    optional_view_validity_decl(&validity, pa_root, ctx),
                ],
                push: option_push,
                series: option_series,
            }
        }
    }
}

/// `as_str` borrows long enough to copy into `MutableBinaryViewArray<str>`.
pub(super) fn as_str_leaf(ctx: &LeafCtx<'_>, base: &StringyBase, arm: LeafArmKind) -> LeafArm {
    let buf = idents::primitive_buf(ctx.base.idx);
    let validity = idents::primitive_validity(ctx.base.idx);
    let access = ctx.base.access;
    let name = ctx.base.name;
    let pp = ctx.paths.prelude();
    let pa_root = ctx.paths.polars_arrow_root();
    match arm {
        LeafArmKind::Bare => {
            let bare_value = super::stringy_value_expr(base, access, super::StringyExprKind::Bare);
            let bare_push = quote! { #buf.push_value_ignore_validity(#bare_value); };
            let bare_series = string_chunked_series(name, &quote! { #buf.freeze() }, pp);
            LeafArm {
                decls: vec![mbva_decl(&buf, pa_root, ctx)],
                push: bare_push,
                series: bare_series,
            }
        }
        LeafArmKind::Option { .. } => {
            let v = idents::leaf_value();
            let option_value =
                super::stringy_value_expr(base, access, super::StringyExprKind::OptionDeref);
            let valid = update_optional_view_validity(ctx.cardinality, &validity, &buf, true);
            let null = update_optional_view_validity(ctx.cardinality, &validity, &buf, false);
            let option_push = quote! {
                match #option_value {
                    ::std::option::Option::Some(#v) => {
                        #valid
                        #buf.push_value_ignore_validity(#v);
                    }
                    ::std::option::Option::None => {
                        #null
                        #buf.push_value_ignore_validity("");
                    }
                }
            };
            let valid_opt = validity_into_option(&validity, pa_root);
            let series = string_chunked_series(
                name,
                &quote! { #buf.freeze().with_validity(#valid_opt) },
                pp,
            );
            let option_series = quote! {{
                #validity.resize(#buf.len(), true);
                #series
            }};
            LeafArm {
                decls: vec![
                    mbva_decl(&buf, pa_root, ctx),
                    optional_view_validity_decl(&validity, pa_root, ctx),
                ],
                push: option_push,
                series: option_series,
            }
        }
    }
}

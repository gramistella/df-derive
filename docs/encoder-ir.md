# Encoder IR

## Why this exists

Before the encoder IR, code generation was a tree of hand-tuned per-shape
emitters. Every supported wrapper combination — bare leaf, `Option<T>`,
`Vec<T>`, `Option<Vec<T>>`, `Vec<Option<T>>`, `Option<Vec<Option<T>>>`,
`Vec<Vec<T>>` for primitives and again for nested structs, and so on — had
its own emitter that built a populator from scratch. Adding a new base type
or wrapper shape meant touching every emitter. Bug fixes (validity-bitmap
edge cases, offset-buffer regressions) had to be replicated per emitter.
The two-track split between primitive shapes and nested-struct shapes was
maintained twice over.

The encoder IR replaces those bespoke emitters with a small composable
language. Each leaf type knows how to emit one column for the unwrapped
shape; combinators (`option(...)`, `vec(...)`) wrap a leaf to add `Option`
semantics or list-array layers. Per-field codegen picks a leaf, then folds
the wrappers right-to-left over it. New wrapper shapes compose for free;
new base types only need a leaf builder.

## The IR speaks the encoder's vocabulary

The parser does not hand the encoder a raw `(base, override, [Option, Vec,
…])` tuple. Instead it folds the parser's legality matrix once at parse
time and produces two semantically narrowed objects per field: a leaf
specifier and a wrapper specifier.

The leaf specifier names the unwrapped element shape directly — every
parser-accepted `(base, override)` combination becomes one variant. There
is no later coercion site where mismatches like "stringy override on a
non-stringy base" or "datetime transform on a non-datetime base" need a
defensive panic; the parser rejects the combination at parse time, and the
leaf specifier the encoder receives is, by construction, one of the
encoder's accepted shapes.

The wrapper specifier names the encoder's wrapper-stack vocabulary
directly. Consecutive `Option`s collapse into per-position counts: above
each `Vec` (folded into list-level validity), immediately surrounding the
leaf (folded into per-element validity), or for the no-`Vec` shape, all in
one bucket (folded into a single `Option<&T>` access expression for the
encoder's option-leaf machinery). The raw count is preserved (not boolean)
so the encoder can decide between a direct match (one Option) and a
multi-Option `as_ref().and_then(...)` collapse (two or more) — Polars
folds them all into one validity bit either way, but the access expression
shapes differ.

The split benefits both directions. The encoder destructures a single,
semantically-narrowed enum and never sees a parser-impossible combination.
The parser does the legality work once, where the user's source span is
still available for actionable errors, instead of letting an `unreachable!`
panic surface deep inside the macro expansion. Any future leaf shape (a
new primitive, a new override) is added by extending the leaf specifier
and the corresponding leaf builder — no cross-coordination across separate
classifier functions.

## The two leaf kinds

Leaves come in two flavors, and both are needed.

**Per-element-push** leaves consume one value at a time inside a per-row
loop. The leaf emits a `push` token stream that runs once per row, plus a
finishing expression that turns the accumulated values into a `Series`.
Numerics, strings, decimals, dates, booleans — every primitive base type —
use this kind. Per-element push avoids any per-row trait indirection: the
push expression is monomorphic in the leaf type and inlines into the
populator's tight loop.

**Collect-then-bulk** leaves do not push per row. They accumulate
`&Inner` references across all rows, then dispatch a single
`<Inner as Columnar>::columnar_from_refs(&refs)` call that materializes
every inner schema column at once. Nested user structs and generic type
parameters use this kind, because `Inner` may itself be a derived type
whose populator pipeline has its own setup cost — paying that cost once per
batch is dramatically cheaper than once per row.

The two kinds cannot be unified: per-element push wins on primitives where
the trait call frame would dominate, and collect-then-bulk wins on nested
structs where amortizing the inner setup is the entire point. The encoder
selects the kind from the base type.

## Why `vec(...)` must fuse over consecutive layers

A `Vec<Vec<…<Vec<T>>>>` field is conceptually N independent `LargeListArray`
wraps, each owning an offsets buffer that partitions the layer below it.
A naive emission would build the inner column once, wrap it in a list, then
re-walk every row to build the next layer's offsets, then re-walk again,
N times. This is O(layer count × leaf count) per row.

The encoder fuses consecutive `Vec` layers into a single bulk emission. It
emits one flat values buffer at the deepest layer, one pair of offsets per
layer, optional outer-list validity bitmaps for layers that have an
adjoining `Option`, and stacks the resulting `LargeListArray`s in one
block. The per-row work walks the wrapper stack once and pushes into the
deepest values buffer plus N offset deltas — O(total leaf count), with N
bounded by a constant overhead.

The bulk-fusion invariant is what makes deep-list shapes practical. Any
encoder change that breaks fusion (e.g. emitting one populator per layer
and then composing them) regresses deep-`Vec` benches by the layer count.

## Why `unsafe` is localized

Building a Polars `Series` from a pre-constructed Arrow array requires
`Series::from_chunks_and_dtype_unchecked`, which is `unsafe`. The bulk-`Vec`
emission path needs this call, because it constructs the `LargeListArray`
manually and wants to hand it to Polars without re-validating dtype matching
that the encoder already guarantees by construction.

If the `unsafe` call lived inside an impl method on `Self`, the
`clippy::unsafe_derive_deserialize` lint would fire on any user type that
combined `#[derive(ToDataFrame, Deserialize)]`. The lint inspects impl
methods on the type, sees `unsafe`, and warns about deserializing into a
type that has `unsafe` impls — which is irrelevant here, because the
`unsafe` is not exposed to deserialization, but the lint cannot know that.

The fix: the entire `unsafe` call lives in a free helper inside the
per-derive anonymous-`const` scope, hidden from the user's namespace.
Inherent impls inside an anonymous const still apply to the outer type
(the same trick `serde_derive` uses), so the visible impl methods are
`unsafe`-free and the lint stops firing. The helper is `#[inline(always)]`
to collapse the call site — plain `#[inline]` left a measurable
regression on bulk-`Vec` benches. Its signature is fully concrete (no
generics) so a single instantiation per derive serves every nested-`Vec`
emitter site.

The invariant: any new emission that calls
`Series::from_chunks_and_dtype_unchecked` must route through the helper.
Inlining the `unsafe` call into a `Self::__df_derive_*` method would
re-trigger the lint.

## Direct-array fast paths

For several common shapes, the encoder skips `Series::new` and typed
builders entirely and constructs the underlying polars-arrow array
directly. The shapes that go through the direct-array path:

- `Vec<numeric>` and `Vec<Option<numeric>>` for every numeric base type
  (including `Decimal` and `DateTime` after their primitive transform).
- `Vec<Vec<…>>` over numerics — fused as described above, with the deepest
  layer materialized as a primitive array, not a typed builder.
- `Vec<Struct>`, `Vec<Option<Struct>>`, `Option<Vec<Struct>>`, and
  `Option<Vec<Option<Struct>>>` — the inner derive's
  `columnar_from_refs` produces inner Series chunks that are stitched
  into stacked `LargeListArray`s without round-tripping through Series
  builders.
- The `Binary` leaf opted into by `#[df_derive(as_binary)]` accumulates
  through `MutableBinaryViewArray<[u8]>` and freezes into a
  `BinaryViewArray`, parallel to the `String` leaf's `Utf8ViewArray`
  path. Inner-Option arms add a parallel `MutableBitmap` validity
  buffer and push an empty slice on `None`, exactly as the
  `String`-with-inner-`Option` path does.

These paths win consistently — usually 5×, sometimes 60× — because they
avoid: Series-name allocation, dtype reinterpretation, validity-bitmap
copies, and the typed-builder's per-element method-call overhead. They also
share offset-buffer construction with the bulk-fusion path, so adding a new
direct-array shape composes with deep-`Vec` fusion automatically.

A non-obvious pitfall on the way: removing a small intermediate `Vec<&T>`
collection in the populator entry can regress nested/string-heavy shapes.
The collection looks like a wasted allocation but its presence anchors the
register state of the tight populator loop in a way LLVM optimizes well;
removing it can move the iterator into a less friendly representation. The
encoder keeps the collection for shapes where benches show it helps.

## Smart-pointer transparency

`Box<T>`, `Rc<T>`, `Arc<T>`, and `Cow<'_, T>` (with sized inner) peel off
the field type at parse time without contributing to the wrapper stack. The
encoder operates on the inner type; the column shape, schema dtype, and
runtime materialization are identical to the bare-T field. The smart-pointer
layers count separately (split into "outer" — above any wrapper — and
"inner" — below a wrapper but above the leaf) so the codegen can emit the
right `*` derefs at the access expression and at the per-element leaf
binding. Most leaf push sites already work via method-call autoderef
(`arc_string.as_str()` resolves through `Deref`); the explicit derefs are
only needed where pattern-binding-by-value or numeric `as` casts skip
autoderef.

`Cow<'_, str>` and `Cow<'_, [T]>` are rejected at parse time. The unsized
inner types collapse the autoderef chain at `str`'s unstable inherent
`as_str()` method (or the lack of `Vec`-shaped iteration for `[T]`); the
existing leaves all expect sized inner shapes. Users should write `String`
or `Vec<T>` directly for those cases.

## Tuple fields

A field whose type is a tuple flattens into one column per primitive leaf
contained in the tuple, recursively. Element columns are named
`<field>.field_<i>`, mirroring the dot-notation convention nested-struct
fields use. Field-level attributes (`as_str`, `as_string`, `as_binary`,
`decimal`, `time_unit`) do not apply to tuple-typed fields — attributes
select a single leaf classification, and a tuple field has multiple leaves.
Users who need per-element attributes hoist the tuple into a named struct.

The unit type `()` is rejected at parse time. A unit-typed field would
contribute zero columns, which collides with the parser's invariant that
every field produces at least one schema entry. This is also why the
HashMap-rejection hint pointing at `Vec<(K, V)>` is now actionable: the
manual conversion compiles.

The parent field's outer wrapper stack distributes across every element
column. Three composition rules:

- **No parent wrappers** (bare `(A, B)`, possibly with smart pointers
  peeled off the parent field). Each element column reaches its leaf via
  a static projection `(parent_access).<i>`. Smart-pointer derefs around
  the element type stack on top of the projection.
- **Parent has only `Option`s** (`Option<(A, B)>`,
  `Option<...<Option<(A, B)>>>`). The parent's collapsed Options become
  one row-level validity bit; per row, the projection is a `.map(|t|
  &t.<i>)` lambda that yields `Option<&Inner>` (or `Option<Inner>` for
  `Copy` primitive leaves). The element's own wrappers compose under
  this single Option layer.
- **Parent has at least one `Vec`** (`Vec<(A, B)>`,
  `Vec<Option<(A, B)>>`, `Option<Vec<(A, B)>>`, …). The composed wrapper
  stack is parent + element layers. The projection happens at the
  parent/element boundary: when the element has no own wrappers, it's
  applied at the per-element value expression (`flat.push((*v).<i>)`);
  when the element has its own `Vec` layers, the iteration source at the
  boundary projects (`for w in v.<i>.iter()`).

The tuple emitter is its own per-element-push pipeline, parallel to the
primitive-vec emitter — it owns its layer-namespaced offsets/validity
buffers and replicates the bulk-fusion invariant on the composed shape.
Sharing a leaf-pieces builder with the primitive path keeps every leaf
kind (numeric / String / Bool / Decimal / DateTime / NaiveDate /
NaiveTime / Duration / Binary) supported without duplication.

Nested tuples (`((A, B), C)`) compose recursively with the static
projection path: each level appends its `.<i>` to the access expression
and projects into the inner tuple's elements. Tuple elements that are
themselves nested-struct types (`(Inner, i32)`) route through the standard
nested-struct collect-then-bulk pipeline with a synthesized access that
includes the projection.

## Coverage

The encoder is total on parser-validated input: every primitive shape — bare
numeric / `String` / `Bool` / `Binary` / `Decimal` / `DateTime` / `Date` /
`Time` / `Duration` leaves, arbitrary `Option<…<Option<T>>>` stacks, and
every vec-bearing wrapper stack including `[Option, Vec, ...]` and deeper
nestings — flows through the encoder IR. Combinations the parser cannot
construct (e.g. `DateTimeToInt` on a non-`DateTime` base, `as_str` on a
non-stringy base, `as_binary` on a non-`Vec<u8>` shape) panic in
`build_leaf` rather than returning `None`.

The temporal leaf family — `DateTime`, `Date`, `Time`, `Duration` — all
share the same shape: a per-row mapping expression produces an `i32` (Date)
or `i64` (DateTime / Time / Duration) physical, accumulated into a
`Vec<i32>` or `Vec<i64>`, then handed to `Series::new` and finished with
`.cast(&dtype)?`. Polars's runtime cast-from-numeric short-circuits when
the source numeric type is the dtype's natural physical, so no copy
happens. The unit attribute (`#[df_derive(time_unit = "ms"|"us"|"ns")]`)
applies to the two leaves with a unit choice — `DateTime` and `Duration` —
and is rejected for `Date` and `Time` (both have a fixed encoding). Std
and chrono `Duration` differ only in the per-row mapping expression
(fallible `as_nanos()`/`u128`-narrowing for std, `num_nanoseconds()` /
`Option<i64>` for chrono); the leaf and vec-leaf shapes are identical.

The `Binary` leaf is opt-in via `#[df_derive(as_binary)]` on a `Vec<u8>`
shape; the parser strips the innermost `Vec` from the wrapper stack and
substitutes the `Binary` leaf in place of the `u8` numeric leaf, so the
encoder sees the same shape it would for any other byte-blob primitive
(bare, `Option`, or one outer `Vec` layer for `Vec<Vec<u8>>` and
`Vec<Option<Vec<u8>>>`). The default behavior — `Vec<u8>` without the
attribute — remains a numeric `List(UInt8)` shape, so the opt-in is the
single decision point for choosing between the two representations.

Two implementation notes worth recording for future reference:

- **`[Option, Vec, ...]` over typed-builder primitives.** Shapes that lead
  with `Option<Vec<...>>` over a primitive (numeric, `String`, `Decimal`,
  or `DateTime`) used to route through the typed
  `ListPrimitiveChunkedBuilder` / `ListStringChunkedBuilder` because
  `append_iter`'s `extend_trusted_len_unchecked` was tighter than the
  encoder's per-element `flat.push + validity.set` loop. The encoder is now
  faster on virtually every shape — particularly `Decimal` shapes, where
  the typed-builder route collected through a fallible scratch `Vec` — so
  the carve-out was retired. A small `Option<Vec<DateTime>>` regression
  (~8-12%) was accepted in exchange for the much larger speedups elsewhere.
- **`isize`/`usize` base types.** The encoder widens these to `i64`/`u64`
  at the codegen boundary so they reuse the i64/u64 vec-emit path; the IR
  carries no separate `isize`/`usize` storage type.

## Invariants

- Every base type registered in the type registry must produce either a
  per-element-push leaf or a collect-then-bulk leaf; the encoder picks
  based on the leaf kind, not the base type, so a misclassified leaf
  silently routes through the wrong populator.
- `vec(...)` must fuse over consecutive `Vec` layers. Emitting one
  populator per layer breaks the O(total leaf count) guarantee on deep-list
  shapes.
- Consecutive `Option` wrappers above a `Vec` collapse to a single
  list-level validity bit. Polars cannot represent two distinct null
  states at one level, so `Some(None)` and `None` are indistinguishable in
  the column — this is a documented contract, not a bug.
- All `Series::from_chunks_and_dtype_unchecked` calls must route through
  the per-derive `__df_derive_assemble_list_series_unchecked` helper.
  Inlining this `unsafe` into an impl method on `Self` re-triggers
  `clippy::unsafe_derive_deserialize` for users of
  `#[derive(ToDataFrame, Deserialize)]`.
- Direct-array fast paths must produce arrays whose arrow dtype matches
  the logical Polars dtype declared in the schema; the
  `from_chunks_and_dtype_unchecked` safety argument depends on this
  match.

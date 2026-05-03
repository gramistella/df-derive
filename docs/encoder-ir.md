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
semantics or list-array layers. Per-field codegen normalizes the wrapper
stack, picks a leaf, then folds the wrappers right-to-left over it. New
wrapper shapes compose for free; new base types only need a leaf builder.

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

## Legacy fallback shapes

A handful of wrapper-shape combinations don't (yet) compose through the
encoder and stay on a separate emitter path:

- **Multi-`Option` primitives.** Shapes like `Option<Option<numeric>>`
  (a primitive base under two or more consecutive `Option`s, no `Vec`
  involved) fall through to the legacy primitive emitter. Polars collapses
  the consecutive `Option`s into one validity bit anyway, so the runtime
  semantics are identical; the encoder simply doesn't model the shape yet.
- **`isize`/`usize` base types.** Because their width is platform-dependent,
  the encoder doesn't have a leaf for them; vec-bearing shapes over `isize`
  / `usize` also fall through. The legacy emitter widens to `i64`/`u64` and
  handles the shape as a plain integer.
- **`[Option, Vec, ...]` typed-builder carve-out.** A small number of
  shapes that lead with `Option<Vec<...>>` over a primitive
  (numeric, `String`, `Decimal`, or `DateTime`) use a typed list builder
  because the `Option` outside the `Vec` interacts with list-level validity
  in a way the typed builder handles cleanly. Bench numbers show no benefit
  from forcing them through the direct-array path; keeping them on the
  typed builder costs nothing.

These carve-outs survived because they are coverage-correct, perf-neutral,
and the encoder's job is to subsume bespoke emitters where the
generalization pays off — not to absorb every corner case for its own
sake. The legacy paths are small, isolated, and tested.

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

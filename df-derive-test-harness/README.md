# df-derive-test-harness

Contract test harness for [`df-derive`]'s `Decimal128Encode` trait.

This is an unpublished workspace crate used by the repository's own tests. It is
not part of the public crates.io release surface.

`df-derive` lowers `Decimal` fields straight to a polars
`Decimal(precision, scale)` column by going through a user-pluggable trait:

```rust,ignore
pub trait Decimal128Encode {
    fn try_to_i128_mantissa(&self, target_scale: u32) -> Option<i128>;
}
```

The contract is **load-bearing**: implementations MUST round half-to-even
(banker's rounding) on scale-down, matching polars's own `str_to_dec128`.
A backend that disagrees (e.g. half-away-from-zero, truncation) silently
produces a column whose bytes diverge from the historical
`to_string + cast` path the codegen replaced.

This crate exposes [`assert_decimal128_encode_contract`], a single function
that cross-checks any user backend against polars's `str_to_dec128` on a
battery of inputs covering positive and negative half-tie boundaries,
non-tie scale-down, scale-up, near-`i128::MAX` magnitudes, and scale-up
overflow (where the contract requires `None`).

## Usage

Within this workspace, call from a `#[test]`, passing a closure that
parses a literal and rescales it via your `Decimal128Encode` impl:

```rust,ignore
use df_derive_test_harness::assert_decimal128_encode_contract;
use my_decimal::MyDecimal;
use std::str::FromStr as _;

#[test]
fn my_decimal_matches_polars_rounding() {
    assert_decimal128_encode_contract(|literal, target_scale| {
        let value = MyDecimal::from_str(literal).unwrap();
        <MyDecimal as my_runtime::Decimal128Encode>::try_to_i128_mantissa(
            &value,
            target_scale,
        )
    });
}
```

If the closure ever returns a mantissa that disagrees with polars's
`str_to_dec128`, the harness panics with both values, the input literal,
and the target `(precision, scale)`, so the failure points directly at the
rounding direction that diverged.

## License

MIT, matching the parent crate.

[`df-derive`]: https://docs.rs/df-derive
[`assert_decimal128_encode_contract`]: src/lib.rs

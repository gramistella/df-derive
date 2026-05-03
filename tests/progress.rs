#[test]
fn tests() {
    let t = trybuild::TestCases::new();

    // These files should compile successfully.
    t.pass("tests/pass/01-simple-struct.rs");
    t.pass("tests/pass/02-all-supported-types.rs");
    t.pass("tests/pass/03-options-and-nulls.rs");
    t.pass("tests/pass/04-empty-struct.rs");
    t.pass("tests/pass/05-nested-structs.rs");
    t.pass("tests/pass/06-vec-consistency-test.rs");
    t.pass("tests/pass/07-edge-cases-test.rs");
    t.pass("tests/pass/08-vec-custom-struct-test.rs");
    t.pass("tests/pass/09-vec-custom-struct-edge-cases.rs");
    t.pass("tests/pass/10-vec-custom-struct-validation.rs");
    t.pass("tests/pass/11-large-scale-stress.rs");
    t.pass("tests/pass/12-self-referential.rs");
    t.pass("tests/pass/13-field-as-string-attribute.rs");
    t.pass("tests/pass/14-decimal-datetime.rs");
    t.pass("tests/pass/15-as-string-on-struct.rs");
    t.pass("tests/pass/16-tuple-structs.rs");
    t.pass("tests/pass/17-nested-tuple-structs.rs");
    t.pass("tests/pass/18-custom-trait.rs");
    t.pass("tests/pass/19-complex-wrappers.rs");
    t.pass("tests/pass/20-generics.rs");
    t.pass("tests/pass/21-field-as-str-attribute.rs");
    t.pass("tests/pass/22-as-str-on-struct.rs");
    t.pass("tests/pass/23-as-str-redundant-on-string.rs");
    t.pass("tests/pass/24-decimal-and-time-unit-attrs.rs");
    t.pass("tests/pass/25-doubly-optional-nested.rs");
    t.pass("tests/pass/26-generics-no-clone.rs");
    t.pass("tests/pass/27-decimal-i128-direct.rs");
    t.pass("tests/pass/28-option-vec-struct-validity.rs");
    t.pass("tests/pass/29-decimal128-encode-trait.rs");
    t.pass("tests/pass/30-vec-option-struct-bulk.rs");
    t.pass("tests/pass/31-option-vec-option-struct-bulk.rs");
    t.pass("tests/pass/32-vec-vec-struct-bulk.rs");
    t.pass("tests/pass/33-vec-option-vec-mid-stack.rs");

    // These files should fail to compile.
    t.compile_fail("tests/fail/96-fail-derive-on-union.rs");
    t.compile_fail("tests/fail/97-fail-derive-on-enum.rs");
    t.compile_fail("tests/fail/98-fail-unsupported-type.rs");
    t.compile_fail("tests/fail/99-fail-as-str-and-as-string-conflict.rs");
    t.compile_fail("tests/fail/100-fail-unknown-field-attribute.rs");
    t.compile_fail("tests/fail/101-fail-decimal-on-non-decimal.rs");
    t.compile_fail("tests/fail/102-fail-time-unit-on-non-datetime.rs");
    t.compile_fail("tests/fail/103-fail-time-unit-invalid-value.rs");
    t.compile_fail("tests/fail/104-fail-decimal-precision-out-of-range.rs");
    t.compile_fail("tests/fail/105-fail-decimal-with-as-string.rs");
    t.compile_fail("tests/fail/106-fail-as-str-on-non-string-base.rs");
}

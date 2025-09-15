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

    // This file should fail to compile.
    t.compile_fail("tests/fail/97-fail-generics.rs");
    t.compile_fail("tests/fail/98-fail-unsupported-type.rs");
}

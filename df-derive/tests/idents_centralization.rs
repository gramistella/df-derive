//! Safety net for the encoder identifier registry at
//! `df-derive-macros/src/codegen/encoder/idents.rs`.
//!
//! Every `__df_derive_*` identifier the macro emits must come from that
//! registry. New emitters that mint identifiers as raw `format_ident!` /
//! `quote!` literals can silently shadow other emitters' bindings when
//! their wrapper combinations stack inside one generated `const _: () = { ... };`
//! scope. The registry is the single source of truth that prevents this.
//!
//! This test scans every source file under `src/codegen/` (recursively) other
//! than `idents.rs` itself for the literal substring `__df_derive_` outside
//! of comments. If any are found the test fails with a per-occurrence list,
//! pointing the developer at the registry file as the place to add a function
//! and route the literal through it.
//!
//! The check is intentionally a substring search — not a tokenizer or AST
//! walk — so it catches the literal regardless of whether it appears inside
//! a `format_ident!`, a raw `quote!` ident, a string literal, an
//! `extern fn` name, or any other surface form. Comment stripping is done
//! line-by-line for `//` (covers `//!` and `///`) and span-by-span for
//! `/* ... */` so doc-comments referring to the registered names by name
//! don't trip the guard.

use std::fs;
use std::path::{Path, PathBuf};

/// Strip Rust line comments (`//`, `///`, `//!`) and `/* ... */` block
/// comments from `src`. The output preserves byte counts (replaces stripped
/// regions with spaces) so failure messages can still cite line numbers
/// derived from the original source.
///
/// String / char literals are NOT preserved — a `"__df_derive_foo"` inside
/// a string literal must still trip the guard, because that's exactly the
/// shape of `quote!{ __df_derive_foo }` after macro expansion-time string
/// interpolation; flagging it is desired.
fn strip_comments(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        // Block comment: replace with spaces, preserve newlines for line
        // numbering.
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            out.push(b' ');
            out.push(b' ');
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                out.push(if bytes[i] == b'\n' { b'\n' } else { b' ' });
                i += 1;
            }
            if i + 1 < bytes.len() {
                out.push(b' ');
                out.push(b' ');
                i += 2;
            } else {
                while i < bytes.len() {
                    out.push(b' ');
                    i += 1;
                }
            }
            continue;
        }
        // Line comment (covers `//`, `///`, `//!`): replace through the
        // newline (exclusive) with spaces.
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' {
                out.push(b' ');
                i += 1;
            }
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    // Safe: we only replaced ASCII bytes with ASCII spaces, leaving every
    // non-comment byte untouched, so the result is still valid UTF-8.
    String::from_utf8(out).expect("comment stripper preserves UTF-8")
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir({}): {e}", dir.display()));
    for entry in entries {
        let entry = entry.expect("read_dir entry");
        let path = entry.path();
        let file_type = entry.file_type().expect("file_type");
        if file_type.is_dir() {
            collect_rs_files(&path, out);
        } else if file_type.is_file() && path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn no_uncentralized_df_derive_idents() {
    let manifest =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set by cargo test");
    let repo_root = PathBuf::from(&manifest)
        .parent()
        .expect("facade crate lives under the workspace root")
        .to_path_buf();
    let codegen_root = repo_root
        .join("df-derive-macros")
        .join("src")
        .join("codegen");
    assert!(
        codegen_root.is_dir(),
        "expected codegen dir at {}",
        codegen_root.display()
    );

    // The registry file itself is the only place every `__df_derive_*`
    // literal is allowed to live. Compute its canonical path so the
    // filter survives symlinks and path-component oddities (drive
    // letters, `.` vs no `.`, etc).
    let registry_path = codegen_root.join("encoder").join("idents.rs");
    let registry_canon = fs::canonicalize(&registry_path).unwrap_or(registry_path);

    let mut files = Vec::new();
    collect_rs_files(&codegen_root, &mut files);
    files.sort();

    let needle = "__df_derive_";
    let mut violations: Vec<String> = Vec::new();
    for path in &files {
        let path_canon = fs::canonicalize(path).unwrap_or_else(|_| path.clone());
        if path_canon == registry_canon {
            continue;
        }
        let src =
            fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let stripped = strip_comments(&src);
        // Walk line-by-line so the failure message can cite line numbers
        // pointing at the original source.
        for (lineno, line) in stripped.lines().enumerate() {
            if line.contains(needle) {
                let original_line = src.lines().nth(lineno).unwrap_or("<line out of range>");
                let display_path = path
                    .strip_prefix(&repo_root)
                    .map_or_else(|_| path.clone(), Path::to_path_buf);
                violations.push(format!(
                    "  {}:{}: {}",
                    display_path.display(),
                    lineno + 1,
                    original_line.trim_end(),
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "found `{needle}` literals outside `df-derive-macros/src/codegen/encoder/idents.rs`. \
         Every `__df_derive_*` identifier the macro emits must come from a \
         function or constant in `df-derive-macros/src/codegen/encoder/idents.rs` — add one \
         there and route the literal through it.\n\nviolations:\n{}",
        violations.join("\n"),
    );
}

//! Safety net for generated external crate paths.
//!
//! `src/codegen/polars_paths.rs` is the only place codegen should spell raw
//! `::polars` or `::polars_arrow` roots. Everywhere else should go through the
//! resolver helpers so downstream dependency renames keep working.

use std::fs;
use std::path::{Path, PathBuf};

fn strip_comments(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
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
            }
            continue;
        }
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
fn generated_polars_roots_are_centralized() {
    let manifest =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set by cargo test");
    let codegen_root = PathBuf::from(&manifest).join("src").join("codegen");
    let allowed = fs::canonicalize(codegen_root.join("polars_paths.rs"))
        .expect("polars_paths.rs should exist");

    let mut files = Vec::new();
    collect_rs_files(&codegen_root, &mut files);
    files.sort();

    let mut violations: Vec<String> = Vec::new();
    for path in &files {
        let path_canon = fs::canonicalize(path).unwrap_or_else(|_| path.clone());
        if path_canon == allowed {
            continue;
        }
        let src =
            fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let stripped = strip_comments(&src);
        for (lineno, line) in stripped.lines().enumerate() {
            if line.contains("::polars::") || line.contains("::polars_arrow::") {
                let original = src.lines().nth(lineno).unwrap_or("<line out of range>");
                let display_path = path
                    .strip_prefix(&manifest)
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|_| path.clone());
                violations.push(format!(
                    "  {}:{}: {}",
                    display_path.display(),
                    lineno + 1,
                    original.trim_end(),
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "generated Polars roots must go through src/codegen/polars_paths.rs \
         so dependency renames are honored.\n\nviolations:\n{}",
        violations.join("\n"),
    );
}

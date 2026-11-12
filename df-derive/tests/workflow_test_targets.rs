//! The workflows run this package's integration tests one target at a time,
//! by name, so that the heavy targets get a job each. A target that is added
//! here and not named there never runs in CI. One that is removed here and
//! still named there fails its job. This test keeps the two lists equal.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

/// The workflows that run the tests: on every push, and before a release.
const WORKFLOWS: [&str; 2] = ["ci.yml", "publish.yml"];

/// This package's test targets: the `.rs` files directly in `tests/`.
fn test_targets(package: &Path) -> BTreeSet<String> {
    fs::read_dir(package.join("tests"))
        .expect("read the tests directory")
        .map(|entry| entry.expect("read a directory entry").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .filter_map(|path| Some(path.file_stem()?.to_str()?.to_owned()))
        .collect()
}

/// The targets a workflow selects with `--test <name>`.
fn named_in(workflow: &str) -> BTreeSet<String> {
    workflow
        .split("--test ")
        .skip(1)
        .filter_map(|rest| rest.split_whitespace().next())
        .map(str::to_owned)
        .collect()
}

#[test]
fn workflows_name_every_test_target_and_no_other() {
    let package = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workflows = package.join("../.github/workflows");
    // A copy of the package from crates.io has no workflows beside it.
    if !workflows.is_dir() {
        return;
    }

    let targets = test_targets(package);
    for file in WORKFLOWS {
        let workflow = fs::read_to_string(workflows.join(file)).expect("read the workflow");
        assert_eq!(
            named_in(&workflow),
            targets,
            "`--test` names in .github/workflows/{file} (left) against the files in tests/ (right)"
        );
    }
}

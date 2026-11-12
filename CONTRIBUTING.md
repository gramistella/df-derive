### Contributing to df-derive

Thanks for your interest in contributing! This guide explains how to set up your environment, run checks, and submit changes.

#### Code of Conduct

Participation in this project is governed by our Code of Conduct (`CODE_OF_CONDUCT.md`). By participating, you agree to abide by it. If you witness or experience unacceptable behavior, email: letter.brigade_7s@icloud.com.

#### Getting Started

- Ensure you have a recent stable Rust toolchain installed. We recommend using `rustup`.
- Clone the repository and navigate into it.
- Install `just` (optional but recommended) to use the provided recipes: `cargo install just`.

#### Development Workflow

1. Create a new branch for your change.
2. Make small, focused commits with clear messages.
3. Keep changes well-formatted and idiomatic.

#### Running Checks

You can use either plain `cargo` commands or the `just` recipes.

- Lint (Clippy, warnings as errors):
  - `just lint`
  - or `cargo clippy --workspace --all-features --all-targets -- -W clippy::all -W clippy::cargo -W clippy::pedantic -W clippy::nursery -D warnings`

- Tests (all features):
  - `just test`
  - or `cargo test --workspace --all-features`
  - A file directly in `df-derive/tests/` is a test target of its own. The
    workflows run targets by name, so a new one has to be added to
    `.github/workflows/ci.yml` and `.github/workflows/publish.yml`.
    `just test` fails until it is.

- Benchmarks:
  - One Criterion target: `cargo bench -p df-derive --bench <name>`, with the
    targets listed in `df-derive/Cargo.toml`.
  - Instruction counts (needs valgrind): `just bench-instructions`.

Criterion compares against whatever results are already in
`target/criterion`. To compare two revisions, run the same targets in a
separate checkout of each and compare the `new/estimates.json` files by
benchmark name.

#### Commit Style

- Keep subject lines under ~72 chars, use the imperative mood: "Add X", "Fix Y".
- Reference issues when applicable (e.g., `Fixes #123`).

#### Pull Requests

- Ensure CI is green locally (lint, tests) before opening a PR.
- Describe the problem and solution clearly. Include benchmarks if performance-related.
- Add tests where appropriate.

#### Reporting Bugs and Requesting Features

- Open an issue with clear reproduction steps or motivation.
- For bugs, include Rust version (`rustc --version`) and platform details.

#### Security

If you discover a security-related issue, please do not open a public issue. Email: letter.brigade_7s@icloud.com.

#### Contact

For any questions or to discuss contributions, reach out at: letter.brigade_7s@icloud.com.


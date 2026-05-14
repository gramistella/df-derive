set shell := ["bash", "-c"]

lint :
	cargo clippy --workspace --all-features --all-targets -- \
		-W clippy::all -W clippy::cargo -W clippy::pedantic -W clippy::nursery -D warnings

test:
	cargo test --workspace --all-features

fmt:
	cargo fmt --all

bench-instructions:
	RUSTFLAGS="${RUSTFLAGS:--C target-cpu=generic}" cargo bench --locked -p df-derive --features bench-instruction-counts --bench instruction_counts

set shell := ["bash", "-c"]

lint :
	cargo clippy --all-features --all-targets -- \
		-W clippy::all -W clippy::cargo -W clippy::pedantic -W clippy::nursery -D warnings

test:
	cargo test --workspace --all-features

fmt:
	cargo fmt --all

bench:
	cargo bench
	@just bench-results

bench-results:
	@echo "Benchmark Mean per iteration (ms)"
	@for f in target/criterion/*/new/estimates.json; do \
		name=$(basename "$(dirname "$(dirname "$f")")"); \
		mean_ns=$(jq '.mean.point_estimate' "$f"); \
		mean_ms=$(awk "BEGIN {printf \"%.3f\", $mean_ns/1000000}"); \
		echo "$name $mean_ms ms"; \
	done | sort -k2 -n
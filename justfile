# orla local task runner. Install: `brew install just`.
# Run `just` for a list of recipes.

# Show available recipes when invoked without arguments.
default:
    @just --list

# Run the full CI pipeline locally (mirrors GitHub Actions). Uses
# test-cover so coverage.out is written; CI uploads it to codecov.
# modernize is intentionally not part of `check` — it's a periodic
# "what idioms could we adopt" review, not a gate.
check: build test-cover lint links

# Compile every package.
build:
    go build ./...

# Run the test suite with the race detector (storage tests need Docker).
test:
    go test -race -timeout 180s ./...

# Like `test` but writes coverage.out. Used by CI.
test-cover:
    go test -race -timeout 180s -coverprofile=coverage.out ./...

# Run golangci-lint v2.
lint:
    golangci-lint run --timeout 5m

# Check markdown links offline (no network). Catches broken local file
# refs like [x](docs/foo.md) when foo got renamed.
links:
    lychee --offline '**/*.md'

# Like `links` but also hits external URLs. Slow and rate-limited;
# run occasionally, don't put in CI.
links-online:
    lychee '**/*.md'

# Report gopls' modernize suggestions (no changes). Uses go run.
modernize:
    go run golang.org/x/tools/gopls/internal/analysis/modernize/cmd/modernize@latest -test ./...

# Apply modernize fixes in place. Review the diff before committing.
modernize-fix:
    go run golang.org/x/tools/gopls/internal/analysis/modernize/cmd/modernize@latest -fix -test ./...

# go fmt + go mod tidy.
fmt:
    go fmt ./...
    go mod tidy

# Regenerate sqlc-generated code under internal/storage/db.
sqlc:
    go run github.com/sqlc-dev/sqlc/cmd/sqlc@latest generate

# Build the daemon binary into bin/orla.
binary:
    @mkdir -p bin
    go build -o bin/orla ./cmd/orla

# Produce coverage.html for the default browser.
coverage:
    go test -coverprofile=coverage.out ./...
    go tool cover -html=coverage.out -o coverage.html
    @echo "open coverage.html"

# Remove build artifacts.
clean:
    rm -rf bin coverage.out coverage.html

# ── demo recipes ──────────────────────────────────────────────────────

# Register the four Bedrock backends + the four stages with orla.
# Requires the daemon to be running and .env to be sourced.
demo-setup:
    bash demo/scripts/setup.sh

# Run a baseline eval (manual stage→backend assignments from setup.sh).
# Pass N=<number> to run fewer questions (default 25).
demo-baseline N="25":
    cd demo && uv run python -m demo.eval --n {{ N }} --mode baseline

# Run an eval while tagging the run as "mapper" mode (the mapper must
# be running in another shell to actually reroute stages).
demo-mapper-eval N="25":
    cd demo && uv run python -m demo.eval --n {{ N }} --mode mapper

# Start the mapper. Runs forever (Ctrl-C to stop).
#   INTERVAL    poll interval seconds (default 15)
#   EPSILON     exploration probability (default 0.1; lower = less random)
#   COST_W      cost weight in reward (default 0.05; small because Bedrock
#               cost spread between backends is only ~5x)
#   LAT_W       latency weight in reward (default 0.05)
demo-mapper INTERVAL="15" EPSILON="0.1" COST_W="0.05" LAT_W="0.05":
    cd demo && uv run python -m demo.mapper \
        --interval {{ INTERVAL }} \
        --epsilon {{ EPSILON }} \
        --cost-weight {{ COST_W }} \
        --latency-weight {{ LAT_W }}

# Seed every (stage, backend) cell with N observations so the mapper
# can start in exploit mode. Run this once after demo-setup, before
# starting the mapper. ~$0.06 for the default 5 questions × 4 backends.
demo-calibrate N="5":
    cd demo && uv run python -m demo.eval.calibrate_main --per-round {{ N }}

# Open the Streamlit dashboard.
demo-dashboard:
    cd demo && uv run streamlit run src/demo/dashboard/app.py

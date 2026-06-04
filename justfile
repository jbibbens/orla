# orla local task runner. Install via https://just.systems.
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

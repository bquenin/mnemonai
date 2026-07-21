# Rust project checks

set positional-arguments
set shell := ["bash", "-euo", "pipefail", "-c"]

# List available commands
default:
    @just --list

# Run format, clippy-fix, build, and test in sequence
check: format clippy-fix build test

# Run check and fail if there are uncommitted changes (for CI)
check-ci: check
    #!/usr/bin/env bash
    set -euo pipefail
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "Error: check caused uncommitted changes"
        echo "Run 'just check' locally and commit the results"
        git diff --stat
        exit 1
    fi

# Format Rust files
format:
    cargo fmt --all

# Run clippy with all warnings
clippy:
    cargo clippy --all-targets -- -W clippy::all

# Auto-fix clippy warnings
clippy-fix:
    cargo clippy --fix --allow-dirty --all-targets -- -W clippy::all

# Build the project
build:
    cargo build --workspace

# Run tests
test:
    cargo test --workspace

# Install debug binary globally via symlink
install-dev:
    cargo build && ln -sf $(pwd)/target/debug/mnemonai ~/.cargo/bin/mnemonai

# Run the application
run *ARGS:
    cargo run -- "$@"

# Release a new patch version
release:
    @just _release patch

# Internal release helper
_release bump:
    @cargo-release {{bump}}

#!/bin/sh
# Builds the wasm module and writes the JS bindings to `$1`
# (default: the demo directory of the homepage repository).
set -eu
cd "$(dirname "$0")"
out="${1:-../../rmitsuboshi.github.io/miniboosts/pkg}"
repo="$(cd .. && pwd)"
# Panic locations embed source paths in the binary; rewrite local prefixes
# so the published file does not reveal the home directory layout.
# When several prefixes match, rustc applies the last one given.
RUSTFLAGS="--remap-path-prefix=$HOME=~ --remap-path-prefix=$repo=miniboosts" \
    cargo build --release --target wasm32-unknown-unknown
wasm=target/wasm32-unknown-unknown/release/miniboosts_web.wasm
if grep -q "$HOME" "$wasm"; then
    echo "error: $wasm still contains $HOME" >&2
    exit 1
fi
wasm-bindgen --target web --no-typescript --out-dir "$out" "$wasm"
ls -l "$out"

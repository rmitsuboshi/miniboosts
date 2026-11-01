#!/bin/sh
# Builds the wasm module and writes the JS bindings to `$1`
# (default: the demo directory of the homepage repository).
set -eu
cd "$(dirname "$0")"
out="${1:-../../rmitsuboshi.github.io/miniboosts/pkg}"
cargo build --release --target wasm32-unknown-unknown
wasm-bindgen --target web --no-typescript --out-dir "$out" \
    target/wasm32-unknown-unknown/release/miniboosts_web.wasm
ls -l "$out"

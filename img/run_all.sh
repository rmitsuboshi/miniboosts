#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || "${1:-}" == --help ]]; then
    echo "Usage: bash img/run_all.sh TRAIN.csv TEST.csv [OUTPUT_DIR] [runner options...]"
    echo "Default output: img/csv. Runner options follow OUTPUT_DIR; see --help on the Rust runner."
    if [[ "${1:-}" == --help ]]; then exit 0; else exit 2; fi
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
train=$1
test=$2
shift 2
# Validate before compiling or creating output files. Paths are relative to the caller.
for input in "$train" "$test"; do
    if [[ ! -f "$input" || ! -r "$input" ]]; then
        printf 'Error: input CSV not found or not readable: %s\n' "$input" >&2
        printf 'Use an existing path relative to your current directory, or an absolute path.\n' >&2
        exit 2
    fi
done
output="$root/img/csv"
if [[ $# -gt 0 ]]; then output=$1; shift; fi
mkdir -p "$output"
# Cargo resolves its target directory; this also respects CARGO_TARGET_DIR.
runner=(cargo run --quiet --release --locked --manifest-path "$root/Cargo.toml" -p miniboosts-example --)
algorithms="$("${runner[@]}" --list)"
{
    date -u '+UTC: %Y-%m-%dT%H:%M:%SZ'
    git -C "$root" rev-parse HEAD
    git -C "$root" status --short
    rustc -Vv
    uname -a
    printf 'Command: '
    printf '%q ' "$0" "$train" "$test" "$output" "$@"
    printf '\n'
} > "$output/environment.txt"
cp "$root/Cargo.lock" "$output/Cargo.lock"
python3 - "$train" "$test" > "$output/data-sha256.txt" <<'PYHASH'
import hashlib
import sys
from pathlib import Path
for name in sys.argv[1:]:
    path = Path(name).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    print(digest.hexdigest(), path)
PYHASH
# Sequential runs avoid CPU contention in timing comparisons. Stop on failure.
while IFS= read -r algorithm; do
    "${runner[@]}" "$algorithm" "$train" "$test" "$output/$algorithm.csv" "$@" \
        2>&1 | tee "$output/$algorithm.log"
done <<< "$algorithms"

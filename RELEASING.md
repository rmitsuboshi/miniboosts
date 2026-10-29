# Publishing MiniBoosts

The release is a single `miniboosts` package. Its version is set in
`[workspace.package]` (currently `0.5.0`). The only other workspace member,
`miniboosts-example`, has `publish = false`. There are no internal registry
dependencies to publish first.

## Verify

Run from the repository root:

```sh
cargo check --workspace --locked
cargo test --workspace --locked
cargo +1.85.0 test --workspace --locked
cargo fmt --all -- --check
cargo package --list --locked
cargo publish --dry-run --locked
```

During local preparation, add `--allow-dirty` to the packaging and dry-run
commands. Review and commit release changes before the actual upload. Check
that the package contains its sources, README, logging guide, and MIT LICENSE;
it must not contain `.venv`, experiment output, or local credentials.

The root API and prelude remain available. Source layout changes do not change
boosting update rules. Read `MIGRATION.md` for the breaking API changes since
0.4.0 and the numerical changes made during development.

## Upload

The existing crates.io owner is `rmitsuboshi` (checked on 2026-10-29). Use that
account or an authorized co-owner. A valid Cargo token needs permission to
publish `miniboosts`; local token presence and dry-run success do not establish
server-side permission. Authenticate interactively with `cargo login` if needed;
never put tokens in this repository.

After verifying that `0.5.0` is still unused, publish only the root package:

```sh
cargo publish --locked
```

Publishing is permanent and versions cannot be overwritten. After a successful
upload, check the crates.io entry and docs.rs build, update the README's
pre-release installation instructions, and tag the released commit.

MiniBoosts uses MIT. Clarabel 0.11.1 is Apache-2.0; dependency licenses remain
applicable when redistributing their code or binaries.

# Publishing MiniBoosts

The release is a single `miniboosts` package. Its version is set in
`[workspace.package]` (currently `0.5.2`). The only other workspace member,
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

After verifying that the new version (e.g. `0.5.2`) is still unused, publish only the root package:

```sh
cargo publish --locked
```

Publishing is permanent and versions cannot be overwritten. After a successful
upload, check the crates.io entry and docs.rs build, update the README's
pre-release installation instructions, and tag the released commit.

MiniBoosts uses MIT. Clarabel 0.11.1 is Apache-2.0; dependency licenses remain
applicable when redistributing their code or binaries.

## Python package

The Python package `miniboosts` (in `python/`) is published to PyPI by
`.github/workflows/release-python.yml`, which builds abi3 wheels for Linux
(glibc and musl), macOS, and Windows plus an sdist, tests the native wheels and
the sdist, and uploads with PyPI Trusted Publishing (no API token).

One-time setup (by the PyPI account owner):

1. On both <https://pypi.org> and <https://test.pypi.org>, add a pending
   publisher for project `miniboosts`: owner `rmitsuboshi`, repository
   `miniboosts`, workflow `release-python.yml`, and environment `pypi`
   (PyPI) or `testpypi` (TestPyPI).
2. In the GitHub repository settings, create the environments `pypi` and
   `testpypi`. Adding a required reviewer to `pypi` makes every upload wait
   for approval.

Each release:

1. Set the same version in `python/pyproject.toml`, `python/Cargo.toml`
   (including its `miniboosts` dependency), and `python/miniboosts/__init__.py`;
   the workflow fails if they differ.
2. Run `python -m pytest python/tests` after `maturin develop --release`.
3. Run the workflow manually (Actions > Python release > Run workflow) on the
   release commit. It publishes to TestPyPI; check the result in a fresh
   environment with
   `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ miniboosts`.
4. Tag the commit `py-v<version>` (e.g. `py-v0.5.2`) and push the tag. The
   workflow checks that the tag matches the package version and publishes to
   PyPI.

A version uploaded to PyPI or TestPyPI can never be uploaded again, even after
deleting it; bump the version to retry.

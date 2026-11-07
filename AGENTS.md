# MiniBoosts

This Rust workspace supports research and fair comparisons of boosting algorithms.
Prioritize mathematical correctness, reproducible experiments, and clear APIs.

## Layout
- `src/core`: shared traits, samples, and utilities.
- `src/boosters`, `src/weak_learners`: algorithm implementations.
- `src/hypotheses`, `src/optimization`, `src/logging`: aggregation, solvers, and measurement.
- `src/lib.rs`, `src/prelude.rs`: public exports.
- `tests`: integration tests and shared data.
- `img/example`: non-published experiment runner; the root library is the only public crate.

## Changes
- Keep boosters independent of concrete weak learners.
- For changes to update rules or stopping criteria, cite the source equation or
  algorithm. Distinguish the original method from approximations and numerical fixes.
- Preserve parameter meanings and document API or behavior changes.
- Check algorithm-specific invariants, boundary cases, and solver status.
  Do not treat non-finite values or solver failures as successful results.
- Keep changes scoped; avoid unrelated formatting and dependency updates.
- Report discrepancies between documentation and implementation.
- Keep APIs consistent across boosters: construct with `init(&sample)`,
  configure with builder methods, and reuse existing parameter names
  (e.g. `tolerance`, `nu`, `force_quit_at`) for the same roles.

## Documentation
Write rustdoc for docs.rs users. Each booster documents at least:
- an overview of the algorithm;
- its theoretical guarantee, if any, with assumptions;
- limitations;
- the original paper (authors, title, link).

Document every public method and function.

## Adding a booster or weak learner
An algorithm is added only when each of these is done, or ruled out with a
reason given to the user:
- the library: exports in `src/lib.rs` and `src/prelude.rs`, the list in
  `src/boosters/mod.rs` or `src/weak_learners/mod.rs`, and `MIGRATION.md`;
- the Python bindings in `python/`: native function, estimator, export,
  tests (passing on Python 3.10, the oldest version in CI), and
  `python/README.md`;
- `README.md`: the Algorithms table with a link to the paper, or the
  weak-learner paragraph;
- the web demo, if it can run the algorithm (two features, binary labels,
  a few hundred points, seconds in single-threaded wasm): `web/src/lib.rs`,
  and in the homepage repository (`../rmitsuboshi.github.io/miniboosts/`) the
  option, its parameters, and the Algorithms list in `index.html`, the
  wiring in `main.js`, the wasm built with `web/build.sh`, and a bumped `v=`
  in `index.html`. Check it in a browser.

## Experiments
- Match data splits, preprocessing, weak learners, and evaluation conditions.
- Record seeds, parameters, stopping rules, commit, dependencies, and hardware.
- Use release builds for timing; state what the measurement includes.
- Do not assume equal tolerance values imply equal accuracy across algorithms.
- Keep test data out of training and parameter selection.

## Verification
- Run tests for changed crates; run `cargo test --workspace` for shared changes.
- Use `cargo check --workspace` and `cargo fmt --all -- --check` as appropriate.
- Test mathematical changes on small deterministic examples with meaningful
  assertions and justified floating-point tolerances.
- Check affected public examples and documentation when APIs change.
- Report commands run, failures, and checks that could not be completed.

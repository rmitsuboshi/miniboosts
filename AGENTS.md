# MiniBoosts

This Rust workspace supports research and fair comparisons of boosting algorithms.
Prioritize mathematical correctness, reproducible experiments, and clear APIs.

## Layout
- `miniboosts-core`: shared traits, samples, and utilities.
- `boosters`, `weak-learners`: algorithm implementations.
- `hypotheses`, `optimization`, `logging`: aggregation, solvers, and measurement.
- `src`: public exports. Check `Cargo.toml` for active workspace members.

## Changes
- Keep boosters independent of concrete weak learners.
- For changes to update rules or stopping criteria, cite the source equation or
  algorithm. Distinguish the original method from approximations and numerical fixes.
- Preserve parameter meanings and document API or behavior changes.
- Check algorithm-specific invariants, boundary cases, and solver status.
  Do not treat non-finite values or solver failures as successful results.
- Keep changes scoped; avoid unrelated formatting and dependency updates.
- Report discrepancies between documentation and implementation.

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

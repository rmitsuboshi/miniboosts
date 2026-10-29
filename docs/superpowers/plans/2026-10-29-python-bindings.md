# Python bindings implementation plan

Goal: coexist with the Rust crate and expose algorithm-specific sklearn-style classifiers with fit(X, y), configurable weak learners, and strict confidence validation.

Accepted design: each algorithm is a classifier class, fit returns self, each fit starts afresh. Clone weak estimators each boosting round. Binary labels are encoded to -1/+1 with one mapping warning per fit unless already -1/+1, and restored by predict. Unfitted inference fails. Nonfinite or out-of-range weak confidence is an error with estimator/round/row/value context. Final ensemble scores are not clipped.

Architecture: standalone non-published Cargo package in python/ using PyO3 and maturin; root Rust users have no Python dependencies. Python handles estimator conventions and validation. Rust retains all eleven boosting algorithms and model aggregation, including graph separation. Add a checked dense Sample constructor for owned array conversion. A callback adapter caches one vector per weak hypothesis so Rust never calls Python per row. A native fitted model evaluates batches of validated weak-hypothesis scores with the original Rust aggregation.

- [x] Add failing Python contract tests and dense-constructor Rust tests.
- [x] Add checked dense constructor, native training/model/tree bridges, and explicit error propagation.
- [x] Implement Python classes, label encoding, fresh-fit semantics, estimator cloning and explicit sklearn prediction adapter.
- [x] Build/install a wheel, run Python tests, Rust workspace/MSRV tests, and distribution smoke checks.
- [x] Document supported inputs, bounded confidence contract, parallelism/serialization limits and install steps; independent code review.

Scope: dense real numeric X and binary classification. No GUI, automatic output scaling, implicit clipping, implicit model serialization, PyPI upload, or full sklearn estimator-check certification. External sklearn estimators can be used directly when they satisfy the contract, otherwise via explicit hard-label adapter. Sparse inputs, multiclass and custom probabilistic transforms remain outside this first release.

Verification: the installed macOS arm64 release wheel passes 45 Python tests on Python 3.12; stable and Rust 1.85 each pass 192 workspace tests (9 existing ignored). Root cargo check, both Rust formatting checks, native Rust 1.85 check, and git diff --check pass. A source distribution builds outside the repository. Root cargo package listing excludes python/ and .venv/. Python 3.10/3.12 CI added, but remote CI and other platforms have not been run locally. Independent review findings (boolean labels and terminal-probe iteration counting) corrected with regression tests. NumPy string-coercion/missing-label validation also covered. No commit or publication performed for this task.

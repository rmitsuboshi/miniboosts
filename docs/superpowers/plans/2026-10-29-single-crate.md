# Single-crate 0.5.0 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan.

**Goal:** Publish only miniboosts 0.5.0, under MIT, with no unpublished internal dependencies.

**Architecture:** Move active member libraries into modules under src. Preserve root exports and prelude; keep the example as a non-publishable workspace member. Do not change mathematical algorithms or external dependency versions.

**Tech Stack:** Rust 2024, MSRV 1.85, existing dependencies including Clarabel 0.11.1 (Apache-2.0).

**Spec:** User selected one public miniboosts crate instead of 17 public packages. This supersedes the earlier package-renaming draft.

## Review Focus

- Root and prelude imports continue compiling in downstream code.
- Internal crate-relative references retain their original meaning after moving.
- Documentation includes only files shipped in the archive.
- Existing deterministic tests survive, including ignored dataset tests.
- No internal registry dependencies, extra public packages, or .venv files are published.

## Task 1: Consolidate and verify

- [x] Move active sources into src/core, src/boosters, src/weak_learners, src/hypotheses, src/optimization, and src/logging; rewrite paths only.
- [x] Move integration tests to tests and consolidate identical German datasets under tests/data.
- [x] Configure root 0.5.0 MIT metadata and external dependencies; remove obsolete active member manifests. Retain disabled historical sources outside the package.
- [x] Update README, migration and release guides, and repository layout instructions.
- [x] Run workspace tests on stable and 1.85, cargo check, fmt, docs, and root cargo publish --dry-run (never upload).
- [x] Review the package contents and obtain an independent final review.

## Progress

The multi-package draft passed workspace publish dry-run; it is superseded by the user's single-package choice. Work continues in the current checkout to preserve the authorized in-progress changes. No upload is authorized or performed.

Final review: no actionable findings. All moved algorithm sources retain their numerical logic. Stable, Rust 1.85, and archive tests: 190 passed, 9 ignored each. Root publish dry-run succeeds without uploading.

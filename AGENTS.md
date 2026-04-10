# KittyCode Agent Guide

## Development Rules

- Practice TDD when behavior changes: add or update a failing test first whenever practical.
- Reuse existing functions and modules before adding new helpers or parallel code paths.
- Do not introduce redundant variables, redundant helper functions, or duplicated logic.
- Prefer small, explicit functions with clear inputs and outputs.
- Keep the `kittycode/` root lean: only `__init__.py`, `main.py`, and `cli.py` should remain as top-level files there.
- Preserve current user-visible behavior unless the task explicitly requires a behavior change.
- Update docs when code moves or user-visible behavior changes.
- Keep `kittycode/runtime/__init__.py` lightweight to avoid circular imports.

## Constraints

- Every code change must pass the full test suite: `python -m pytest -q`.
- Do not leave the project with broken imports, half-moved modules, or stale structure documentation.
- If a package move breaks legacy import paths still used by tests or callers, add a compatibility layer instead of forcing a broad rewrite.
- Keep the implementation simple and readable; avoid unnecessary architectural expansion.
- When changing the agent loop, tools, or CLI interrupt flow, verify both normal completion and interruption paths.

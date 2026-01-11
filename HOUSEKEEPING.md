# Housekeeping Protocol

1. Read the AGENTS.md file.
2. Look at the dependency network of the project, namely which script refers to which one.
3. Proceed doing different sanity checks and unit tests from root scripts to leaves.
4. Add that report to the AGENTS_LOG.md
5. Compile all errors and tests results into a report. And print that report in the Latest Report subsection below, overwriting previous reports.

# Current Project Housekeeping

## Dependency Network

**Status: BROKEN**
The following modules are referenced by tests but are missing from the codebase:
- `src.data.download`: Referenced by `tests/test_download_mock.py`, `tests/test_integration.py`.
- `src.data.process`: Referenced by `tests/test_integration.py`.
- `src.models.train_cf`: Referenced by `Makefile`.
- `src.models.train_bandit`: Referenced by `Makefile`.

Theoretical Structure (based on `AGENTS.md`):
- `src.data` -> `src.models`
- `tests` -> `src`

## Latest Report

**Execution Date:** [Current Date]

**Test Results:**
- `make test`: **FAILED**.
  - `ImportError: No module named 'src.data'` in 2/2 tests.
- `src.data.process`: **FAILED**. Module missing.
- `src.models.train_cf`: **FAILED**. Module missing.
- `src.models.train_bandit`: **FAILED**. Module missing.

**Summary:**
The project is currently in a critical state. The entire source code library (`src/data`, `src/models`) is missing from the file system, although `src/__init__.py` exists. The dependency graph is broken. Tests cannot run. The environment has been set up with `pip install -r requirements.txt`, but code execution is impossible.

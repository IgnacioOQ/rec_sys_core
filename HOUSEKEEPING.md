# Housekeeping Protocol

1. Read the AGENTS.md file.
2. Look at the dependency network of the project, namely which script refers to which one.
3. Proceed doing different sanity checks and unit tests from root scripts to leaves.
4. Add that report to the AGENTS_LOG.md
5. Compile all errors and tests results into a report. And print that report in the Latest Report subsection below, overwriting previous reports.

# Current Project Housekeeping

## Dependency Network

Based on updated import analysis:
- **Core Modules:**
    - `src.data.download`: Base and specific pipeline classes (MovieLens, Amazon).
    - `src.data.utils`: Helper functions for stats.
    - `src.data.process`: Orchestrator, depends on `download` and `utils`.
- **Advanced Modules (Models):**
    - `src.models.train_cf`: Depends on `scikit-surprise` and processed data.
    - `src.models.train_bandit`: Depends on `contextualbandits`, `sklearn`, and processed data.
- **Tests:**
    - `tests.test_download_mock`: Depends on `src.data.download`.
    - `tests.test_integration`: Depends on `src.data.process`.
- **Notebooks:**
    - `notebooks/exploration.ipynb` & `colab_exploration.ipynb`: Depend on all `src` modules.

## Latest Report

**Execution Date:** [Current Date]

**Test Results:**
- `make test`: **PASSED** (3 tests: 1 integration, 2 mocks).
- `src.data.process`: **PASSED**. Data files created in `data/interim/`.
- `src.models.train_cf`: **PASSED**. Model saved to `models/svd_model.pkl`.
    - *Note:* Required downgrading `numpy` to `<2` (specifically installed `1.26.4`) due to `scikit-surprise` incompatibility with NumPy 2.x.
- `src.models.train_bandit`: **PASSED**. Policy saved to `models/bandit_policy.pkl`.

**Summary:**
The project is structurally sound and functional. The dependency graph is clean with `src.data` acting as the foundation for `src.models`. All sanity checks passed after resolving a NumPy version conflict.

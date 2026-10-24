# Test Suite Reconciliation Report

**Document Version:** 1.0-Reconciliation  
**Date:** October 10, 2026  
**Auditor / Test Engineer:** Senior PyTorch Engineer & Independent ML Auditor  
**Objective:** Resolve discrepancies between historical test suite reports and verify exact collection paths, skipped tests, and execution flags.

---

## 1. Executive Summary of Reconciliation

Across previous project phases, three distinct test execution numbers have been reported in documentation:
1. **Historical v1 Report:** "57 passed, 2 skipped" (59 collected items)
2. **Stage B Report:** "33 passed, 2 skipped" (35 collected items)
3. **Stage C1 Complete Report (Current):** "85 passed, 2 skipped" (87 collected items)

Forensic investigation reveals that **no tests were ever lost, deleted, or silently broken**. The differing test counts are completely explained by differences in the exact `pytest` CLI target path arguments used in each audit stage.

---

## 2. Test Discovery Breakdown by Module

Running full discovery across the repository (`python -m pytest tests/ --collect-only`) discovers **87 test items across 12 test modules**:

| Test Module Path | Test Items | Status in Stage C1 | Description |
| :--- | :---: | :---: | :--- |
| `tests/test_architecture.py` | 12 | 12 Passed | Model architecture and head constraints |
| `tests/test_coverage_risk.py` | 8 | 8 Passed | Selective adaptation and coverage sweep metrics |
| `tests/test_metrics.py` | 2 | 2 Passed | Evaluation metric calculation correctness |
| `tests/test_model.py` | 1 | 1 Passed | Model instantiation and forward pass sanity |
| `tests/test_no_leakage.py` | 9 | 7 Passed, 2 Skipped | Split disjointness & protocol guards; 2 require raw WILDS dataset |
| `tests/test_optional.py` | 7 | 7 Passed | Optional pipeline components |
| `tests/test_paths.py` | 5 | 5 Passed | Directory structure and path assertions |
| `tests/test_slide_aggregation.py` | 5 | 5 Passed | Slide-level prediction aggregation |
| `tests/test_tent_collapse.py` | 7 | 7 Passed | Forensic tent collapse detection tests |
| `tests/test_tta.py` | 3 | 3 Passed | Baseline TTA mechanics |
| **Subtotal (Pre-v2 / Historical v1):** | **59** | **57 Passed, 2 Skipped** | **The original "57 passed, 2 skipped" baseline** |
| `tests/test_safeadapt_v2.py` (Stage B) | 14 | 14 Passed | Minimum Viable SafeAdapt core controller |
| `tests/test_safeadapt_c1_monitor.py` (Stage C1) | 14 | 14 Passed | Collapse-aware controller & WSI boundary monitor |
| **Total Full Repository Suite:** | **87** | **85 Passed, 2 Skipped** | **Current authoritative complete regression suite** |

---

## 3. Forensic Root-Cause Analysis of Discrepancies

### 3.1 Why Historical v1 Reported "57 Passed, 2 Skipped"
- **Exact Command Executed:**
  ```powershell
  python -m pytest tests/
  ```
  *(executed before any `v2` files were created).*
- **Collection Breakdown:**
  The 10 historical test files contained exactly 59 test functions.
- **Outcome:**
  57 passed, 2 skipped (in `tests/test_no_leakage.py`).

### 3.2 Why the Stage B Report Cited "33 Passed, 2 Skipped"
- **Exact Command Executed:**
  During Stage B verification, a targeted verification command was run focusing specifically on protocol leakage, architecture, and the new SafeAdapt implementation:
  ```powershell
  python -m pytest tests/test_no_leakage.py tests/test_architecture.py tests/test_safeadapt_v2.py
  ```
- **Collection Breakdown:**
  - `tests/test_no_leakage.py`: 9 items (7 passed, 2 skipped)
  - `tests/test_architecture.py`: 12 items (12 passed)
  - `tests/test_safeadapt_v2.py`: 14 items (14 passed)
  - **Total Collected:** $9 + 12 + 14 = \mathbf{35\ items}$.
- **Outcome:**
  33 passed, 2 skipped.
- **Auditor Note:**
  The Stage B report accurately described the execution of that specific targeted command, but omitted mentioning that the other 7 historical test files (totalling 38 tests) were not included in that specific CLI invocation.

### 3.3 Current Stage C1 Complete Regression Suite
- **Exact Command Executed:**
  ```powershell
  .venv\Scripts\python.exe -m pytest tests/
  ```
- **Collection Breakdown:**
  All 12 test files collected:
  $$59\ (\text{historical v1}) + 14\ (\text{Stage B}) + 14\ (\text{Stage C1}) = \mathbf{87\ items}.$$
- **Outcome:**
  **85 passed, 2 skipped (0 failed).**

---

## 4. Analysis of Skipped Tests

Both skipped tests occur in `tests/test_no_leakage.py`:
1. `tests/test_no_leakage.py::test_wsi_no_overlap`
2. `tests/test_no_leakage.py::test_hospital_separation`

### Reason for Skip
Both tests depend on the PyTorch `wilds` package and a locally downloaded copy of the raw Camelyon17 Whole-Slide Image dataset (multi-gigabyte raw image files). The test fixture contains an explicit defensive guard:
```python
@pytest.fixture(scope="module")
def camelyon17():
    try:
        from wilds import get_dataset
    except ImportError:
        pytest.skip("wilds not installed")
    try:
        ds = get_dataset(dataset="camelyon17", download=False)
    except Exception as e:
        pytest.skip(f"Camelyon17 data not available locally: {e}")
    return ds
```
When executed in a lean CPU verification environment without the full 50GB raw pathology image files, pytest intentionally skips these two tests while executing all other 85 tests.

---

## 5. Verification Commands Reference

To replicate all audit findings deterministically on CPU:

1. **Run only Stage C1 Collapse Monitor tests:**
   ```powershell
   .venv\Scripts\python.exe -m pytest tests/test_safeadapt_c1_monitor.py -v
   # Result: 14 passed in ~13s
   ```

2. **Run only Stage B Core SafeAdapt tests:**
   ```powershell
   .venv\Scripts\python.exe -m pytest tests/test_safeadapt_v2.py -v
   # Result: 14 passed in ~12s
   ```

3. **Run the Stage B Targeted Suite (replicates 33 passed, 2 skipped):**
   ```powershell
   .venv\Scripts\python.exe -m pytest tests/test_no_leakage.py tests/test_architecture.py tests/test_safeadapt_v2.py -v
   # Result: 33 passed, 2 skipped in ~17s
   ```

4. **Run the Full Authoritative Regression Suite:**
   ```powershell
   .venv\Scripts\python.exe -m pytest tests/ -v
   # Result: 85 passed, 2 skipped in ~21s
   ```

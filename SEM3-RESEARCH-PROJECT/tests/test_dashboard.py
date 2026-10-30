"""
tests/test_dashboard.py
=======================
Automated tests for Module 9 — Administrator/Researcher Dashboard.

Tests cover modules/dashboard_data.py (the pure-Python data layer).
The Streamlit page itself (pages/model_comparison.py) is not executed here
because Streamlit pages require a running server; the data layer is tested
independently.

Framework : Python built-in unittest
Run with  :
    py -m unittest tests.test_dashboard -v
    OR
    py -m unittest discover -s tests -v

Test philosophy
---------------
- Uses synthetic fixtures (in-memory JSON and numpy arrays).
- All fixtures are explicitly labelled as TEST DATA.
- No external network requests.
- No disk writes to the real models/ directory.
- Tests verify the data contract, not Streamlit rendering details.

Test index
----------
 1. load_dashboard_data — valid JSON produces DashboardData with valid=True
 2. load_dashboard_data — missing file produces valid=False with error message
 3. load_dashboard_data — invalid JSON produces valid=False
 4. load_dashboard_data — non-object JSON produces valid=False
 5. load_dashboard_data — model1 skipped produces valid=False
 6. load_dashboard_data — model2 skipped preserved, valid=True
 7. Specificity derived correctly: 1 − FPR
 8. metric_differences — M2 − M1 computed correctly (positive case)
 9. metric_differences — M2 − M1 computed correctly (negative case)
10. metric_differences — diff=None when either value is missing
11. MetricDiff.m2_is_better — True when M2 higher and higher_better=True
12. MetricDiff.m2_is_better — True when M2 lower and higher_better=False
13. MetricDiff.m2_is_better — False when M1 better (higher_better=True)
14. MetricDiff.m2_is_better — None when values are approximately equal
15. MetricDiff.m2_is_better — None when diff is None
16. build_export_csv — returns non-empty string for valid data
17. build_export_csv — returns empty string for invalid DashboardData
18. build_export_csv — contains expected metric keys
19. build_export_csv — does not contain sensitive data (no URLs, no paths)
20. build_export_csv — M2-skipped case produces "skipped" rows
21. describe_validity_issues — synthetic dataset flagged
22. describe_validity_issues — perfect scores flagged
23. describe_validity_issues — same split note recorded
24. describe_validity_issues — small dataset flagged
25. _parse_model_section — invalid float skipped with warning
26. _parse_model_section — infinite float rejected with warning
27. _parse_model_section — out-of-range float warned but preserved
28. Confusion matrix cross-check: mismatch with test_size adds warning
29. load_eval_arrays — missing file returns available=False with message
30. load_eval_arrays — valid npz file returns available=True with arrays
31. load_eval_arrays — corrupt npz returns available=False with message
32. load_eval_arrays — npz missing required key returns available=False
33. METRIC_DEFINITIONS — all entries have 4 fields and valid types
34. Positive-class label is phishing (label 1) in parsed metadata
35. Regression: existing test suites (url_validator, feature_extractor,
                authentication, security_analyzer, ml_predictor, risk_assessor)
                all imported without error
"""

from __future__ import annotations

import json
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from modules.dashboard_data import (
    load_dashboard_data,
    load_eval_arrays,
    build_export_csv,
    describe_validity_issues,
    metric_differences,
    METRIC_DEFINITIONS,
    DashboardData,
    MetricSet,
    MetricDiff,
    DatasetMeta,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_model_section(
    accuracy: float = 0.92,
    precision: float = 0.90,
    recall: float = 0.88,
    f1_score: float = 0.89,
    fpr: float = 0.05,
    fnr: float = 0.12,
    roc_auc: float = 0.95,
    tn: int = 95, fp: int = 5, fn: int = 12, tp: int = 88,
    test_size: int = 200,
    feature_count: int = 25,
    train_size: int = 800,
) -> dict:
    """Return a minimal model section dict — all test data."""
    return {
        "accuracy":   accuracy,
        "precision":  precision,
        "recall":     recall,
        "f1_score":   f1_score,
        "fpr":        fpr,
        "fnr":        fnr,
        "roc_auc":    roc_auc,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "test_size":        test_size,
        "test_phishing_count": tp + fn,
        "test_legit_count":    tn + fp,
        "feature_count":    feature_count,
        "train_size":       train_size,
        "training_duration_seconds": 1.2,
    }


def _make_metrics_doc(
    m1: dict | None = None,
    m2: dict | None = None,
    dataset: str = "test_dataset.csv",   # TEST DATA
    total_rows: int = 1000,
    phishing_count: int = 500,
    legitimate_count: int = 500,
    test_fraction: float = 0.2,
    random_seed: int = 42,
) -> dict:
    return {
        "metadata": {
            "dataset": dataset,
            "total_rows": total_rows,
            "phishing_count": phishing_count,
            "legitimate_count": legitimate_count,
            "label_column": "label",
            "label_encoding": {"phishing": 1, "legitimate": 0},
            "test_fraction": test_fraction,
            "random_seed": random_seed,
            "trained_at_utc": "2025-01-01T00:00:00Z",
        },
        "model1": m1 if m1 is not None else _make_model_section(),
        "model2": m2 if m2 is not None else _make_model_section(
            feature_count=39, fpr=0.04, fnr=0.10, accuracy=0.93,
        ),
    }


def _write_json_to_tmp(doc: dict) -> Path:
    """Write doc to a temp file and return its Path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(doc, f)
    f.close()
    return Path(f.name)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestLoadDashboardDataValid(unittest.TestCase):
    """Tests 1-6: Basic loading and valid/invalid cases."""

    def setUp(self):
        self.tmp = _write_json_to_tmp(_make_metrics_doc())

    def tearDown(self):
        os.unlink(self.tmp)

    def test_01_valid_json_produces_valid(self):
        dd = load_dashboard_data(self.tmp)
        self.assertTrue(dd.valid, f"Expected valid=True, got error: {dd.error}")

    def test_02_missing_file_produces_invalid(self):
        dd = load_dashboard_data(Path("/nonexistent/metrics.json"))
        self.assertFalse(dd.valid)
        self.assertIn("not found", dd.error.lower())

    def test_03_invalid_json_produces_invalid(self):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        f.write("{not valid json")
        f.close()
        try:
            dd = load_dashboard_data(Path(f.name))
            self.assertFalse(dd.valid)
        finally:
            os.unlink(f.name)

    def test_04_non_object_json_produces_invalid(self):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        f.write("[1, 2, 3]")
        f.close()
        try:
            dd = load_dashboard_data(Path(f.name))
            self.assertFalse(dd.valid)
        finally:
            os.unlink(f.name)

    def test_05_model1_skipped_produces_invalid(self):
        doc = _make_metrics_doc(m1={"skipped": True, "reason": "Not trained"})
        tmp = _write_json_to_tmp(doc)
        try:
            dd = load_dashboard_data(tmp)
            self.assertFalse(dd.valid)
        finally:
            os.unlink(tmp)

    def test_06_model2_skipped_valid_true(self):
        doc = _make_metrics_doc(m2={"skipped": True, "reason": "No security features"})
        tmp = _write_json_to_tmp(doc)
        try:
            dd = load_dashboard_data(tmp)
            self.assertTrue(dd.valid)
            self.assertTrue(dd.model2.skipped)
        finally:
            os.unlink(tmp)


class TestDerivedMetrics(unittest.TestCase):
    """Tests 7-15: Specificity, diffs, m2_is_better."""

    def setUp(self):
        doc = _make_metrics_doc()
        self.tmp = _write_json_to_tmp(doc)
        self.dd = load_dashboard_data(self.tmp)

    def tearDown(self):
        os.unlink(self.tmp)

    def test_07_specificity_derived_from_fpr(self):
        # specificity = 1 - fpr
        fpr = self.dd.model1.fpr
        spec = self.dd.model1.specificity
        self.assertIsNotNone(fpr)
        self.assertIsNotNone(spec)
        self.assertAlmostEqual(spec, 1.0 - fpr, places=6)

    def test_08_diff_positive_case(self):
        # M2 accuracy 0.93, M1 accuracy 0.92 → diff = +0.01
        d = next(x for x in self.dd.diffs if x.key == "accuracy")
        self.assertIsNotNone(d.diff)
        self.assertAlmostEqual(d.diff, 0.93 - 0.92, places=6)

    def test_09_diff_negative_case_fpr(self):
        # M2 fpr 0.04, M1 fpr 0.05 → diff = -0.01
        d = next(x for x in self.dd.diffs if x.key == "fpr")
        self.assertIsNotNone(d.diff)
        self.assertAlmostEqual(d.diff, 0.04 - 0.05, places=6)

    def test_10_diff_none_when_m2_missing(self):
        doc = _make_metrics_doc(m2={"skipped": True, "reason": "—"})
        tmp = _write_json_to_tmp(doc)
        try:
            dd = load_dashboard_data(tmp)
            for d in dd.diffs:
                self.assertIsNone(d.diff)
        finally:
            os.unlink(tmp)

    def test_11_m2_is_better_true_higher_better(self):
        d = MetricDiff(key="f1_score", label="F1", m1_value=0.80, m2_value=0.85,
                       diff=0.05, higher_better=True)
        self.assertTrue(d.m2_is_better)

    def test_12_m2_is_better_true_lower_better(self):
        # FPR: lower is better, M2 has lower FPR
        d = MetricDiff(key="fpr", label="FPR", m1_value=0.10, m2_value=0.06,
                       diff=-0.04, higher_better=False)
        self.assertTrue(d.m2_is_better)

    def test_13_m2_is_better_false_m1_better(self):
        d = MetricDiff(key="recall", label="Recall", m1_value=0.92, m2_value=0.88,
                       diff=-0.04, higher_better=True)
        self.assertFalse(d.m2_is_better)

    def test_14_m2_is_better_none_when_equal(self):
        d = MetricDiff(key="accuracy", label="Accuracy",
                       m1_value=0.90, m2_value=0.90 + 5e-11,
                       diff=5e-11, higher_better=True)
        # diff < 1e-9 → approximately equal
        self.assertIsNone(d.m2_is_better)

    def test_15_m2_is_better_none_when_diff_is_none(self):
        d = MetricDiff(key="roc_auc", label="ROC-AUC",
                       m1_value=None, m2_value=0.95,
                       diff=None, higher_better=True)
        self.assertIsNone(d.m2_is_better)


class TestBuildExportCsv(unittest.TestCase):
    """Tests 16-20: CSV export."""

    def setUp(self):
        self.tmp = _write_json_to_tmp(_make_metrics_doc())
        self.dd = load_dashboard_data(self.tmp)

    def tearDown(self):
        os.unlink(self.tmp)

    def test_16_valid_data_produces_non_empty_csv(self):
        csv_str = build_export_csv(self.dd)
        self.assertGreater(len(csv_str), 0)

    def test_17_invalid_data_produces_empty_csv(self):
        bad_dd = DashboardData(valid=False, error="No data")
        csv_str = build_export_csv(bad_dd)
        self.assertEqual(csv_str, "")

    def test_18_csv_contains_metric_keys(self):
        csv_str = build_export_csv(self.dd)
        for _, key, _, _ in METRIC_DEFINITIONS:
            self.assertIn(key, csv_str, f"Key '{key}' missing from CSV export")

    def test_19_csv_does_not_contain_sensitive_data(self):
        csv_str = build_export_csv(self.dd)
        # Should not contain any file path separators beyond the dataset filename
        self.assertNotIn("\\", csv_str.replace("\\\\", ""))   # no backslash paths
        self.assertNotIn("C:\\", csv_str)                      # no absolute Windows paths
        self.assertNotIn("/home/", csv_str)                    # no Unix home paths
        # Should not contain any URL strings
        self.assertNotIn("http://", csv_str)
        self.assertNotIn("https://", csv_str)

    def test_20_m2_skipped_produces_skipped_row(self):
        doc = _make_metrics_doc(m2={"skipped": True, "reason": "No security columns"})
        tmp = _write_json_to_tmp(doc)
        try:
            dd = load_dashboard_data(tmp)
            csv_str = build_export_csv(dd)
            self.assertIn("skipped", csv_str)
        finally:
            os.unlink(tmp)


class TestValidityIssues(unittest.TestCase):
    """Tests 21-24: describe_validity_issues."""

    def _load(self, doc: dict) -> DashboardData:
        tmp = _write_json_to_tmp(doc)
        try:
            return load_dashboard_data(tmp)
        finally:
            os.unlink(tmp)

    def test_21_synthetic_dataset_flagged(self):
        doc = _make_metrics_doc(dataset="synthetic_test_dataset.csv")
        dd = self._load(doc)
        notes = describe_validity_issues(dd)
        self.assertTrue(any("synthetic" in n.lower() for n in notes))

    def test_22_perfect_scores_flagged(self):
        perfect_m1 = _make_model_section(
            accuracy=1.0, precision=1.0, recall=1.0, f1_score=1.0
        )
        doc = _make_metrics_doc(m1=perfect_m1, dataset="real_dataset.csv",
                                total_rows=5000)
        dd = self._load(doc)
        notes = describe_validity_issues(dd)
        self.assertTrue(any("perfect" in n.lower() for n in notes))

    def test_23_same_split_note_recorded(self):
        doc = _make_metrics_doc()
        dd = self._load(doc)
        # Same test size → same_test_split=True
        notes = describe_validity_issues(dd)
        self.assertTrue(any(n for n in notes))  # at least one note exists

    def test_24_small_dataset_flagged(self):
        doc = _make_metrics_doc(total_rows=150, dataset="small_real.csv")
        dd = self._load(doc)
        notes = describe_validity_issues(dd)
        self.assertTrue(any("150" in n or "small" in n.lower() or "rows" in n.lower()
                            for n in notes))


class TestParseModelSectionRobustness(unittest.TestCase):
    """Tests 25-28: Robustness against bad metric values."""

    def _load(self, doc: dict) -> DashboardData:
        tmp = _write_json_to_tmp(doc)
        try:
            return load_dashboard_data(tmp)
        finally:
            os.unlink(tmp)

    def test_25_invalid_float_skipped_with_warning(self):
        m1 = _make_model_section()
        m1["accuracy"] = "not-a-number"
        doc = _make_metrics_doc(m1=m1)
        dd = self._load(doc)
        self.assertTrue(dd.valid)
        self.assertIsNone(dd.model1.accuracy)
        self.assertTrue(any("accuracy" in w for w in dd.warnings))

    def test_26_infinite_float_rejected_with_warning(self):
        import math
        m1 = _make_model_section()
        m1["precision"] = float("inf")
        doc = _make_metrics_doc(m1=m1)
        dd = self._load(doc)
        self.assertTrue(dd.valid)
        self.assertIsNone(dd.model1.precision)
        self.assertTrue(any("precision" in w.lower() for w in dd.warnings))

    def test_27_out_of_range_float_adds_warning(self):
        m1 = _make_model_section()
        m1["recall"] = 1.5   # out of [0, 1]
        doc = _make_metrics_doc(m1=m1)
        dd = self._load(doc)
        self.assertTrue(dd.valid)
        # Warns about out-of-range
        self.assertTrue(any("recall" in w.lower() for w in dd.warnings))

    def test_28_cm_mismatch_with_test_size_adds_warning(self):
        m1 = _make_model_section(tn=50, fp=10, fn=10, tp=50, test_size=200)
        # tn+fp+fn+tp = 120 ≠ test_size 200
        doc = _make_metrics_doc(m1=m1)
        dd = self._load(doc)
        self.assertTrue(dd.valid)
        # The mismatch should produce a warning
        self.assertTrue(any("200" in w or "confusion" in w.lower() for w in dd.warnings))


class TestLoadEvalArrays(unittest.TestCase):
    """Tests 29-32: EvalArrays loading."""

    def test_29_missing_file_returns_unavailable(self):
        ea = load_eval_arrays(Path("/nonexistent/eval_arrays.npz"))
        self.assertFalse(ea.available)
        self.assertGreater(len(ea.error), 0)

    def test_30_valid_npz_returns_available(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not available")

        m1_y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.int8)
        m1_y_prob = np.array([0.1, 0.9, 0.2, 0.85, 0.78, 0.05, 0.15, 0.92],
                              dtype=np.float32)

        f = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        f.close()
        try:
            np.savez_compressed(f.name, m1_y_true=m1_y_true, m1_y_prob=m1_y_prob)
            ea = load_eval_arrays(Path(f.name))
            self.assertTrue(ea.available, f"Expected available=True, got error: {ea.error}")
            self.assertIsNotNone(ea.m1_y_true)
            self.assertIsNotNone(ea.m1_y_prob)
            self.assertIsNone(ea.m2_y_true)   # not saved
        finally:
            os.unlink(f.name)

    def test_31_corrupt_file_returns_unavailable(self):
        f = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        f.write(b"this is not a valid npz file")
        f.close()
        try:
            ea = load_eval_arrays(Path(f.name))
            self.assertFalse(ea.available)
            self.assertGreater(len(ea.error), 0)
        finally:
            os.unlink(f.name)

    def test_32_missing_required_key_returns_unavailable(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not available")

        # Save only m2 arrays (missing m1_y_true)
        f = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        f.close()
        try:
            np.savez_compressed(
                f.name,
                m2_y_true=np.array([0, 1], dtype=np.int8),
                m2_y_prob=np.array([0.1, 0.9], dtype=np.float32),
            )
            ea = load_eval_arrays(Path(f.name))
            self.assertFalse(ea.available)
            self.assertIn("m1_y_true", ea.error)
        finally:
            os.unlink(f.name)


class TestMetricDefinitions(unittest.TestCase):
    """Test 33: METRIC_DEFINITIONS schema."""

    def test_33_all_definitions_have_correct_structure(self):
        self.assertGreater(len(METRIC_DEFINITIONS), 0)
        for entry in METRIC_DEFINITIONS:
            self.assertEqual(len(entry), 4,
                             f"Expected 4-tuple, got {len(entry)}: {entry}")
            label, key, higher_better, desc = entry
            self.assertIsInstance(label, str)
            self.assertIsInstance(key, str)
            self.assertIsInstance(higher_better, bool)
            self.assertIsInstance(desc, str)
            self.assertGreater(len(desc), 0)


class TestPositiveClassConvention(unittest.TestCase):
    """Test 34: Positive class = phishing = label 1."""

    def test_34_phishing_positive_class_in_metadata(self):
        doc = _make_metrics_doc()
        tmp = _write_json_to_tmp(doc)
        try:
            dd = load_dashboard_data(tmp)
            self.assertEqual(dd.metadata.phishing_label, 1)
            self.assertEqual(dd.metadata.legitimate_label, 0)
        finally:
            os.unlink(tmp)


class TestRegressionExistingTests(unittest.TestCase):
    """
    Test 35: Regression — existing modules can all be imported without error.
    This catches import-level breakage from the Module 9 changes.
    """

    def test_35a_url_validator_imports(self):
        from modules.url_validator import validate_url, ValidationResult
        self.assertTrue(callable(validate_url))

    def test_35b_feature_extractor_imports(self):
        from modules.feature_extractor import extract_features, numeric_feature_names
        self.assertTrue(callable(extract_features))

    def test_35c_authentication_imports(self):
        from modules.authentication import ROLE_ANALYST, ROLE_ADMIN
        self.assertNotEqual(ROLE_ANALYST, ROLE_ADMIN)

    def test_35d_security_analyzer_imports(self):
        from modules.security_analyzer import analyze_security
        self.assertTrue(callable(analyze_security))

    def test_35e_ml_predictor_imports(self):
        from modules.ml_predictor import predict_model1, predict_model2, MODEL1_FEATURES
        self.assertEqual(len(MODEL1_FEATURES), 25)

    def test_35f_risk_assessor_imports(self):
        from modules.risk_assessor import assess_risk, RiskCategory
        self.assertTrue(callable(assess_risk))

    def test_35g_dashboard_data_imports(self):
        from modules.dashboard_data import load_dashboard_data, build_export_csv
        self.assertTrue(callable(load_dashboard_data))
        self.assertTrue(callable(build_export_csv))


class TestAuthorizationContract(unittest.TestCase):
    """
    Tests for RBAC contract on data-loading functions.
    The dashboard_data module is pure Python with no Streamlit dependency,
    so it can be called by any caller.  Authorization is enforced at the
    page level (require_admin()) before these functions are ever called.

    These tests verify that the data-loading functions themselves do not
    bypass or weaken the authorization contract by exposing private data
    when called directly (e.g., they never return credentials or PII).
    """

    def setUp(self):
        self.tmp = _write_json_to_tmp(_make_metrics_doc())
        self.dd  = load_dashboard_data(self.tmp)

    def tearDown(self):
        os.unlink(self.tmp)

    def test_36_export_never_contains_passwords(self):
        csv_str = build_export_csv(self.dd)
        self.assertNotIn("password", csv_str.lower())
        self.assertNotIn("credential", csv_str.lower())
        self.assertNotIn("secret", csv_str.lower())

    def test_37_export_contains_only_declared_metric_keys(self):
        """CSV must only contain expected metric names, not hidden internal fields."""
        csv_str = build_export_csv(self.dd)
        # Internal session-state / authentication fields must never appear in CSV
        # Check for exact standalone occurrences (not substrings of legitimate words)
        import re
        for forbidden in ["_pw_hash", "wt_authenticated", "wt_email", "wt_role"]:
            # Match the key only when surrounded by non-word chars (not inside a longer word)
            pattern = r'(?<![A-Za-z0-9_])' + re.escape(forbidden) + r'(?![A-Za-z0-9_])'
            self.assertFalse(
                bool(re.search(pattern, csv_str)),
                f"Forbidden key '{forbidden}' found in CSV export",
            )

    def test_38_m2_skipped_reason_does_not_expose_paths(self):
        doc = _make_metrics_doc(
            m2={"skipped": True,
                "reason": "Security feature columns missing from dataset: ['sec_has_csp']"}
        )
        tmp = _write_json_to_tmp(doc)
        try:
            dd = load_dashboard_data(tmp)
            csv_str = build_export_csv(dd)
            # Column names are fine to expose; absolute paths are not
            self.assertNotIn("C:\\Users", csv_str)
            self.assertNotIn("/home/", csv_str)
        finally:
            os.unlink(tmp)

    def test_39_access_control_module_defines_require_admin(self):
        from modules.access_control import require_admin
        self.assertTrue(callable(require_admin))

    def test_40_access_control_module_defines_require_analyst_or_admin(self):
        from modules.access_control import require_analyst_or_admin
        self.assertTrue(callable(require_analyst_or_admin))




class TestNavigationWiring(unittest.TestCase):
    """
    Tests 41-47: Verify that the navigation source files contain the correct
    routing code for the Research Dashboard and that the RBAC role constants
    match what the pages check.

    These are static-analysis tests (grep the source files); they do not run
    Streamlit.  They catch the specific regression where model_comparison.py
    exists and is registered in app.py but is unreachable because no page
    contains a switch_page() call pointing to it.
    """

    def _read(self, rel_path: str) -> str:
        path = _PROJECT_ROOT / rel_path
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def test_41_app_py_registers_model_comparison(self):
        """app.py must register pages/model_comparison.py in st.navigation."""
        src = self._read("app.py")
        self.assertIn("model_comparison.py", src)
        self.assertIn("model_comparison_page", src)
        self.assertIn("st.navigation", src)

    def test_42_home_has_switch_to_research_for_admin(self):
        """pages/home.py must contain a switch_page to model_comparison for admin users."""
        src = self._read("pages/home.py")
        self.assertIn("model_comparison.py", src)
        # Must be conditional on ROLE_ADMIN, not unconditional
        self.assertIn("ROLE_ADMIN", src)

    def test_43_analyst_detection_has_switch_to_research_for_admin(self):
        """pages/analyst_detection.py must contain a switch_page to model_comparison for admin."""
        src = self._read("pages/analyst_detection.py")
        self.assertIn("model_comparison.py", src)
        self.assertIn("ROLE_ADMIN", src)

    def test_44_model_comparison_enforces_require_admin(self):
        """pages/model_comparison.py must call require_admin() before rendering."""
        src = self._read("pages/model_comparison.py")
        self.assertIn("require_admin()", src)

    def test_45_home_research_button_not_shown_to_analysts(self):
        """
        The Research Dashboard button in home.py must be inside an
        if _role == ROLE_ADMIN block (i.e., not reachable by analysts).
        """
        src = self._read("pages/home.py")
        # The switch_page to model_comparison.py must appear after a ROLE_ADMIN check
        admin_pos = src.find("ROLE_ADMIN")
        research_btn_pos = src.find("model_comparison.py")
        self.assertGreater(admin_pos, 0, "ROLE_ADMIN check missing from home.py")
        self.assertGreater(research_btn_pos, 0, "model_comparison.py link missing from home.py")
        # The admin check must appear before the research button link
        self.assertLess(
            admin_pos, research_btn_pos,
            "ROLE_ADMIN check must appear before the Research Dashboard button in home.py",
        )

    def test_46_model_comparison_page_exists(self):
        """pages/model_comparison.py must exist."""
        path = _PROJECT_ROOT / "pages" / "model_comparison.py"
        self.assertTrue(path.exists(), "pages/model_comparison.py does not exist")

    def test_47_dashboard_data_module_exists(self):
        """modules/dashboard_data.py must exist."""
        path = _PROJECT_ROOT / "modules" / "dashboard_data.py"
        self.assertTrue(path.exists(), "modules/dashboard_data.py does not exist")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()

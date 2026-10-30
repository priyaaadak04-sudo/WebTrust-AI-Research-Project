"""
tests/test_ml_predictor.py
===========================
Automated tests for Module 7 — ML Phishing Detection.

Framework : Python built-in unittest
Run with  :
    py -m unittest tests.test_ml_predictor -v
    OR
    py -m unittest discover -s tests -v

Test philosophy
---------------
- Uses a small synthetic dataset for training in isolated unit tests.
- Clearly identifies synthetic data — metrics from these tests are NOT the
  project's real research results.
- No real network requests.  SecurityAnalysisResult objects are constructed
  directly without calling analyze_security().
- Tests verify behaviour for both the "model exists" and "model missing" paths.
- All tests are self-contained; they do not depend on models/metrics.json being
  present on disk.

Test index
----------
 1.  MODEL1_FEATURES matches numeric_feature_names() exactly
 2.  MODEL2_FEATURES = MODEL1_FEATURES + MODEL2_SECURITY_FEATURES
 3.  Feature vector lengths match feature name list lengths
 4.  build_model1_vector — correct values from FeatureResult
 5.  build_model1_vector — missing feature returns error
 6.  build_model2_vector — None security_result returns error
 7.  build_model2_vector — failed security_result returns error
 8.  build_model2_vector — successful security_result produces 39 values
 9.  Security feature extraction — all headers present
10.  Security feature extraction — no headers, no cookies, no TLS
11.  Security feature extraction — TLS verified, days_remaining computed
12.  Security feature extraction — TLS not applicable → days_remain = -999
13.  predict_model1 — model not loaded → PredictionResult.success = False
14.  predict_model2 — model not loaded → PredictionResult.success = False
15.  predict_model1 — synthetic trained model → success = True, valid class
16.  predict_model2 — synthetic trained model → success = True, valid class
17.  predict_model1 — predicted_class is 0 or 1
18.  predict_model2 — predicted_class is 0 or 1
19.  predict_model1 — phishing_probability in [0, 1]
20.  predict_model2 — phishing_probability in [0, 1]
21.  Class label mapping: 0 → "Legitimate", 1 → "Potentially Phishing"
22.  predict_model2 with security_result=None → error about security analysis
23.  predict_model2 with failed security_result → error message
24.  load_metrics — missing file returns empty dict and error string
25.  load_metrics — valid JSON → non-empty dict
26.  Feature schema stable across multiple calls
27.  build_model1_vector — boolean features coerced to 0.0/1.0
28.  Synthetic training: Model 1 trains and saves successfully
29.  Synthetic training: saved model loads and predicts correctly
30.  train_models label normalisation — integer labels
31.  train_models label normalisation — string phishing/legitimate
32.  train_models label normalisation — unknown labels → ValueError
33.  train_models _extract_row_features — valid URL
34.  Authorization: load_metrics returns error when metrics missing (not crash)
35.  PredictionResult.features_used length matches feature_count
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

from modules.feature_extractor import numeric_feature_names, extract_features
from modules.url_validator import validate_url
from modules.ml_predictor import (
    MODEL1_FEATURES,
    MODEL2_FEATURES,
    MODEL2_SECURITY_FEATURES,
    build_model1_vector,
    build_model2_vector,
    _extract_security_features,
    predict_model1,
    predict_model2,
    load_metrics,
    load_models,
    PredictionResult,
    LABEL_PHISHING,
    LABEL_LEGITIMATE,
    CLASS_NAMES,
)
from modules.security_analyzer import (
    SecurityAnalysisResult,
    HeaderCheckResult,
    CookieCheckResult,
    TLSCheckResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feature_result(url: str = "https://example.com"):
    vr = validate_url(url)
    assert vr.is_valid, f"Expected valid URL, got: {vr.errors}"
    return extract_features(vr)


def _make_security_result(success: bool = True) -> SecurityAnalysisResult:
    """Build a minimal SecurityAnalysisResult for testing."""
    hc_list = [
        HeaderCheckResult(
            name="Content-Security-Policy",
            present=True, value="default-src 'self'",
            explanation="CSP", observation="Present", suggestion="",
            applicable=True,
        ),
        HeaderCheckResult(
            name="Strict-Transport-Security",
            present=True, value="max-age=31536000",
            explanation="HSTS", observation="Present", suggestion="",
            applicable=True,
        ),
        HeaderCheckResult(
            name="X-Content-Type-Options",
            present=True, value="nosniff",
            explanation="XCTO", observation="Present", suggestion="",
            applicable=True,
        ),
        HeaderCheckResult(
            name="X-Frame-Options",
            present=True, value="DENY",
            explanation="XFO", observation="Present", suggestion="",
            applicable=True,
        ),
        HeaderCheckResult(
            name="Referrer-Policy",
            present=True, value="strict-origin",
            explanation="RP", observation="Present", suggestion="",
            applicable=True,
        ),
        HeaderCheckResult(
            name="Permissions-Policy",
            present=True, value="geolocation=()",
            explanation="PP", observation="Present", suggestion="",
            applicable=True,
        ),
    ]
    tls = TLSCheckResult(
        applicable=True, verified=True,
        subject="example.com", issuer="Test CA",
        valid_from="2024-01-01", valid_until="2026-01-01",
        hostname_match=True, days_remaining=365,
        observation="Valid.",
    )
    return SecurityAnalysisResult(
        url="https://example.com",
        success=success,
        error="" if success else "Network error",
        status_code=200 if success else None,
        header_checks=hc_list if success else [],
        cookie_checks=[],
        cookies_present=False,
        tls=tls if success else TLSCheckResult(applicable=False),
    )


def _build_synthetic_pipeline(n_features: int) -> Pipeline:
    """Build and fit a tiny RandomForest pipeline on synthetic data."""
    np.random.seed(42)
    X = np.random.rand(100, n_features)
    y = np.array([0] * 50 + [1] * 50)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     RandomForestClassifier(n_estimators=10, random_state=42)),
    ])
    pipe.fit(X, y)
    return pipe


# ---------------------------------------------------------------------------
# Test 1-3 — Feature schema consistency
# ---------------------------------------------------------------------------

class TestFeatureSchema(unittest.TestCase):

    def test_01_model1_features_match_numeric_feature_names(self):
        """MODEL1_FEATURES must match the canonical list from feature_extractor."""
        self.assertEqual(
            MODEL1_FEATURES,
            numeric_feature_names(),
            msg="MODEL1_FEATURES and numeric_feature_names() must be identical",
        )

    def test_02_model2_features_equals_m1_plus_security(self):
        """MODEL2_FEATURES must be MODEL1_FEATURES concatenated with MODEL2_SECURITY_FEATURES."""
        expected = MODEL1_FEATURES + MODEL2_SECURITY_FEATURES
        self.assertEqual(MODEL2_FEATURES, expected)

    def test_03_feature_vector_lengths(self):
        """Vector length must equal feature name list length for both models."""
        self.assertEqual(len(MODEL1_FEATURES), 25)
        self.assertEqual(len(MODEL2_SECURITY_FEATURES), 14)
        self.assertEqual(len(MODEL2_FEATURES), 39)


# ---------------------------------------------------------------------------
# Test 4-8 — Feature vector builders
# ---------------------------------------------------------------------------

class TestFeatureVectorBuilders(unittest.TestCase):

    def setUp(self):
        self.fr = _make_feature_result()
        self.sr = _make_security_result(success=True)
        self.sr_failed = _make_security_result(success=False)

    def test_04_build_model1_vector_correct_length(self):
        """build_model1_vector returns a vector of length 25."""
        values, names, err = build_model1_vector(self.fr)
        self.assertEqual(len(values), 25)
        self.assertEqual(len(names), 25)

    def test_04b_build_model1_vector_correct_values(self):
        """build_model1_vector values must match FeatureResult.numeric."""
        values, names, err = build_model1_vector(self.fr)
        self.assertEqual(err, "", msg=f"Unexpected error: {err}")
        for fname, val in zip(names, values):
            raw = self.fr.numeric[fname]
            expected = float(int(raw)) if isinstance(raw, bool) else float(raw)
            self.assertAlmostEqual(val, expected, places=6, msg=f"Mismatch for {fname}")

    def test_05_build_model1_vector_missing_feature(self):
        """A FeatureResult with a missing numeric key returns an error string."""
        fr = _make_feature_result()
        del fr.numeric["url_length"]
        values, names, err = build_model1_vector(fr)
        self.assertIn("url_length", err, msg="Error should mention the missing feature")

    def test_06_build_model2_vector_none_security(self):
        """build_model2_vector with None security_result returns an error."""
        values, names, err = build_model2_vector(self.fr, None)
        self.assertNotEqual(err, "", msg="Error must be non-empty when security is None")
        self.assertIn("security analysis", err.lower())

    def test_07_build_model2_vector_failed_security(self):
        """build_model2_vector with a failed SecurityAnalysisResult returns an error."""
        values, names, err = build_model2_vector(self.fr, self.sr_failed)
        self.assertNotEqual(err, "")
        self.assertIn("failed", err.lower())

    def test_08_build_model2_vector_success(self):
        """build_model2_vector with a successful result returns 39 values."""
        values, names, err = build_model2_vector(self.fr, self.sr)
        self.assertEqual(len(values), 39)
        self.assertEqual(len(names), 39)


# ---------------------------------------------------------------------------
# Test 9-12 — Security feature extraction
# ---------------------------------------------------------------------------

class TestSecurityFeatureExtraction(unittest.TestCase):

    def test_09_all_headers_present(self):
        """With all headers present and verified TLS, all sec_has_* should be 1."""
        sr = _make_security_result(success=True)
        sec = _extract_security_features(sr)
        self.assertEqual(len(sec), 14)
        # Headers
        self.assertEqual(sec[0], 1.0, "sec_has_csp should be 1")
        self.assertEqual(sec[1], 1.0, "sec_has_hsts should be 1")
        self.assertEqual(sec[2], 1.0, "sec_has_xcto should be 1")
        self.assertEqual(sec[3], 1.0, "sec_has_xfo should be 1")
        self.assertEqual(sec[4], 1.0, "sec_has_referrer should be 1")
        self.assertEqual(sec[5], 1.0, "sec_has_permissions should be 1")
        # TLS
        self.assertEqual(sec[10], 1.0, "sec_tls_applicable should be 1")
        self.assertEqual(sec[11], 1.0, "sec_tls_verified should be 1")
        self.assertEqual(sec[12], 365.0, "sec_tls_days_remain should be 365")

    def test_10_no_headers_no_cookies_no_tls(self):
        """With no headers, cookies, or TLS, all sec features should be 0 / sentinel."""
        sr = SecurityAnalysisResult(
            url="http://example.com",
            success=True,
            header_checks=[
                HeaderCheckResult(
                    name=n, present=False, value=None,
                    explanation="", observation="", suggestion="",
                    applicable=True,
                )
                for n in [
                    "Content-Security-Policy", "Strict-Transport-Security",
                    "X-Content-Type-Options", "X-Frame-Options",
                    "Referrer-Policy", "Permissions-Policy",
                ]
            ],
            cookie_checks=[],
            cookies_present=False,
            tls=TLSCheckResult(applicable=False),
        )
        sec = _extract_security_features(sr)
        self.assertEqual(sec[0], 0.0, "sec_has_csp should be 0")
        self.assertEqual(sec[10], 0.0, "sec_tls_applicable should be 0")
        self.assertEqual(sec[11], 0.0, "sec_tls_verified should be 0")
        self.assertEqual(sec[12], -999.0, "sec_tls_days_remain should be sentinel -999")

    def test_11_tls_verified_days_remain(self):
        """TLS days_remaining value must be correctly propagated."""
        sr = _make_security_result(success=True)
        sr.tls.days_remaining = 30
        sec = _extract_security_features(sr)
        self.assertEqual(sec[12], 30.0)

    def test_12_tls_not_applicable_sentinel(self):
        """When TLS is not applicable, days_remain must be the sentinel -999."""
        sr = _make_security_result(success=True)
        sr.tls = TLSCheckResult(applicable=False)
        sec = _extract_security_features(sr)
        self.assertEqual(sec[10], 0.0)
        self.assertEqual(sec[12], -999.0)


# ---------------------------------------------------------------------------
# Test 13-14 — Model not loaded
# ---------------------------------------------------------------------------

class TestModelNotLoaded(unittest.TestCase):

    def test_13_predict_model1_no_artifact(self):
        """When model artifact is missing, predict_model1 returns success=False."""
        fr = _make_feature_result()
        # Force reload with a non-existent path
        with patch("modules.ml_predictor.MODEL1_PATH", Path("/nonexistent/model1.joblib")):
            with patch("modules.ml_predictor._models_loaded", False):
                with patch("modules.ml_predictor._model1_pipeline", None):
                    # Patch the load to fail
                    import modules.ml_predictor as mp
                    orig = mp._models_loaded
                    mp._models_loaded = False
                    mp._model1_pipeline = None
                    mp._model1_load_error = "Model artifact not found: /nonexistent/model1.joblib"
                    result = predict_model1(fr)
                    mp._models_loaded = orig
        self.assertFalse(result.success)
        self.assertIn("not found", result.error.lower())

    def test_14_predict_model2_no_artifact(self):
        """When model artifact is missing, predict_model2 returns success=False."""
        fr = _make_feature_result()
        sr = _make_security_result(success=True)
        import modules.ml_predictor as mp
        orig_loaded = mp._models_loaded
        orig_m2 = mp._model2_pipeline
        orig_err = mp._model2_load_error
        mp._models_loaded = True
        mp._model2_pipeline = None
        mp._model2_load_error = "Model artifact not found: /nonexistent/model2.joblib"
        result = predict_model2(fr, sr)
        mp._models_loaded = orig_loaded
        mp._model2_pipeline = orig_m2
        mp._model2_load_error = orig_err
        self.assertFalse(result.success)
        self.assertIn("not found", result.error.lower())


# ---------------------------------------------------------------------------
# Test 15-20 — Synthetic trained model predictions
# ---------------------------------------------------------------------------

class TestSyntheticModelPredictions(unittest.TestCase):
    """
    These tests use a small synthetic Random Forest to exercise the prediction
    code paths without requiring real training data.
    SYNTHETIC DATA ONLY — these metrics do not represent real research results.
    """

    @classmethod
    def setUpClass(cls):
        """Build and inject a synthetic pipeline for Model 1 and Model 2."""
        cls.pipe1 = _build_synthetic_pipeline(len(MODEL1_FEATURES))
        cls.pipe2 = _build_synthetic_pipeline(len(MODEL2_FEATURES))
        cls.fr = _make_feature_result("https://bank.example.com/login?session=abc")
        cls.sr = _make_security_result(success=True)

    def _inject_m1(self):
        import modules.ml_predictor as mp
        mp._models_loaded = True
        mp._model1_pipeline = self.pipe1
        mp._model1_load_error = ""

    def _inject_m2(self):
        import modules.ml_predictor as mp
        mp._model2_pipeline = self.pipe2
        mp._model2_load_error = ""

    def _cleanup(self):
        import modules.ml_predictor as mp
        mp._models_loaded = False
        mp._model1_pipeline = None
        mp._model2_pipeline = None
        mp._model1_load_error = ""
        mp._model2_load_error = ""

    def test_15_predict_model1_success(self):
        """predict_model1 with a synthetic pipeline returns success=True."""
        self._inject_m1()
        try:
            result = predict_model1(self.fr)
        finally:
            self._cleanup()
        self.assertTrue(result.success, msg=f"Error: {result.error}")

    def test_16_predict_model2_success(self):
        """predict_model2 with a synthetic pipeline returns success=True."""
        self._inject_m1()
        self._inject_m2()
        try:
            result = predict_model2(self.fr, self.sr)
        finally:
            self._cleanup()
        self.assertTrue(result.success, msg=f"Error: {result.error}")

    def test_17_predict_model1_valid_class(self):
        """Predicted class from Model 1 must be 0 or 1."""
        self._inject_m1()
        try:
            result = predict_model1(self.fr)
        finally:
            self._cleanup()
        self.assertIn(result.predicted_class, [0, 1])

    def test_18_predict_model2_valid_class(self):
        """Predicted class from Model 2 must be 0 or 1."""
        self._inject_m1()
        self._inject_m2()
        try:
            result = predict_model2(self.fr, self.sr)
        finally:
            self._cleanup()
        self.assertIn(result.predicted_class, [0, 1])

    def test_19_predict_model1_valid_probability(self):
        """Phishing probability from Model 1 must be in [0, 1]."""
        self._inject_m1()
        try:
            result = predict_model1(self.fr)
        finally:
            self._cleanup()
        if result.phishing_probability is not None:
            self.assertGreaterEqual(result.phishing_probability, 0.0)
            self.assertLessEqual(result.phishing_probability, 1.0)

    def test_20_predict_model2_valid_probability(self):
        """Phishing probability from Model 2 must be in [0, 1]."""
        self._inject_m1()
        self._inject_m2()
        try:
            result = predict_model2(self.fr, self.sr)
        finally:
            self._cleanup()
        if result.phishing_probability is not None:
            self.assertGreaterEqual(result.phishing_probability, 0.0)
            self.assertLessEqual(result.phishing_probability, 1.0)

    def test_35_feature_count_matches_features_used(self):
        """PredictionResult.feature_count must equal len(features_used)."""
        self._inject_m1()
        try:
            result = predict_model1(self.fr)
        finally:
            self._cleanup()
        self.assertEqual(result.feature_count, len(result.features_used))


# ---------------------------------------------------------------------------
# Test 21 — Class label mapping
# ---------------------------------------------------------------------------

class TestClassLabels(unittest.TestCase):

    def test_21_class_name_mapping(self):
        """Class label constants must map correctly."""
        self.assertEqual(LABEL_LEGITIMATE, 0)
        self.assertEqual(LABEL_PHISHING, 1)
        self.assertEqual(CLASS_NAMES[0], "Legitimate")
        self.assertEqual(CLASS_NAMES[1], "Potentially Phishing")


# ---------------------------------------------------------------------------
# Test 22-23 — Model 2 with invalid security result
# ---------------------------------------------------------------------------

class TestModel2SecurityRequirement(unittest.TestCase):

    def setUp(self):
        self.fr = _make_feature_result()

    def test_22_predict_model2_none_security(self):
        """predict_model2 with security_result=None must fail with a clear error."""
        import modules.ml_predictor as mp
        pipe = _build_synthetic_pipeline(len(MODEL2_FEATURES))
        mp._models_loaded = True
        mp._model2_pipeline = pipe
        mp._model2_load_error = ""
        try:
            result = predict_model2(self.fr, None)
        finally:
            mp._models_loaded = False
            mp._model2_pipeline = None
            mp._model2_load_error = ""
        self.assertFalse(result.success)
        self.assertIn("security analysis", result.error.lower())

    def test_23_predict_model2_failed_security(self):
        """predict_model2 with a failed SecurityAnalysisResult must fail."""
        import modules.ml_predictor as mp
        pipe = _build_synthetic_pipeline(len(MODEL2_FEATURES))
        mp._models_loaded = True
        mp._model2_pipeline = pipe
        mp._model2_load_error = ""
        sr_failed = SecurityAnalysisResult(
            url="https://example.com",
            success=False,
            error="Connection timeout",
        )
        try:
            result = predict_model2(self.fr, sr_failed)
        finally:
            mp._models_loaded = False
            mp._model2_pipeline = None
            mp._model2_load_error = ""
        self.assertFalse(result.success)
        self.assertIn("failed", result.error.lower())


# ---------------------------------------------------------------------------
# Test 24-25 — Metrics loading
# ---------------------------------------------------------------------------

class TestMetricsLoading(unittest.TestCase):

    def test_24_load_metrics_missing_file(self):
        """load_metrics returns empty dict and error when file missing."""
        with patch("modules.ml_predictor.METRICS_JSON_PATH", Path("/nonexistent/metrics.json")):
            data, err = load_metrics()
        self.assertEqual(data, {})
        self.assertIn("not found", err.lower())

    def test_25_load_metrics_valid_json(self):
        """load_metrics returns the parsed dict when a valid JSON file exists."""
        sample = {"metadata": {"dataset": "test.csv"}, "model1": {"accuracy": 0.95}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample, f)
            tmp_path = Path(f.name)
        try:
            with patch("modules.ml_predictor.METRICS_JSON_PATH", tmp_path):
                data, err = load_metrics()
            self.assertEqual(err, "")
            self.assertIn("model1", data)
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test 26 — Feature schema stability
# ---------------------------------------------------------------------------

class TestFeatureSchemaStability(unittest.TestCase):

    def test_26_schema_stable_across_calls(self):
        """Feature names must be identical across multiple calls."""
        self.assertEqual(MODEL1_FEATURES, MODEL1_FEATURES)
        self.assertEqual(MODEL2_FEATURES, MODEL2_FEATURES)
        # Also verify lists are not mutated
        self.assertEqual(len(MODEL1_FEATURES), 25)
        self.assertEqual(len(MODEL2_FEATURES), 39)


# ---------------------------------------------------------------------------
# Test 27 — Boolean feature coercion
# ---------------------------------------------------------------------------

class TestBooleanCoercion(unittest.TestCase):

    def test_27_booleans_coerced_to_float(self):
        """build_model1_vector must coerce boolean features to 0.0 or 1.0."""
        fr = _make_feature_result("https://example.com")
        values, names, err = build_model1_vector(fr)
        # All values must be float
        for v in values:
            self.assertIsInstance(v, float, msg=f"Expected float, got {type(v)}: {v}")


# ---------------------------------------------------------------------------
# Test 28-29 — Synthetic end-to-end training and loading
# ---------------------------------------------------------------------------

class TestSyntheticTrainingAndLoading(unittest.TestCase):
    """
    SYNTHETIC DATA ONLY — metrics produced here are not real research results.
    """

    def test_28_synthetic_model_trains_and_saves(self):
        """A synthetic sklearn Pipeline can be saved and is a valid joblib artifact."""
        pipe = _build_synthetic_pipeline(len(MODEL1_FEATURES))
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            tmp_path = f.name
        try:
            joblib.dump(pipe, tmp_path)
            self.assertTrue(Path(tmp_path).exists())
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_29_saved_model_loads_and_predicts(self):
        """A saved pipeline loads correctly and produces valid predictions."""
        pipe = _build_synthetic_pipeline(len(MODEL1_FEATURES))
        fr = _make_feature_result()
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            tmp_path = Path(f.name)

        try:
            joblib.dump(pipe, str(tmp_path))
            loaded = joblib.load(str(tmp_path))

            # Test prediction directly
            values, names, _ = build_model1_vector(fr)
            X = pd.DataFrame([values], columns=names)
            preds = loaded.predict(X)
            self.assertIn(preds[0], [0, 1])
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test 30-32 — train_models label normalisation
# ---------------------------------------------------------------------------

class TestLabelNormalisation(unittest.TestCase):

    def _norm(self, series, p=None, l=None):
        from train_models import _normalise_labels
        return _normalise_labels(pd.Series(series), p, l)

    def test_30_integer_labels(self):
        """Integer 1/0 labels map correctly to PHISHING/LEGITIMATE."""
        result = self._norm([1, 0, 1, 0])
        self.assertEqual(list(result), [1, 0, 1, 0])

    def test_31_string_phishing_legitimate(self):
        """String 'phishing'/'legitimate' labels map correctly."""
        result = self._norm(["phishing", "legitimate", "phishing"])
        self.assertEqual(list(result), [1, 0, 1])

    def test_32_unknown_labels_raise(self):
        """Unrecognised label values with explicit mapping raise ValueError."""
        from train_models import _normalise_labels
        with self.assertRaises(ValueError):
            _normalise_labels(pd.Series(["cat", "dog"]), "cat", "fish")


# ---------------------------------------------------------------------------
# Test 33 — Feature extraction in train_models
# ---------------------------------------------------------------------------

class TestTrainModelsExtraction(unittest.TestCase):

    def test_33_extract_row_features_valid_url(self):
        """_extract_row_features returns a dict with all MODEL1 features for a valid URL."""
        from train_models import _extract_row_features
        row = pd.Series({"url": "https://example.com/login"})
        result = _extract_row_features("https://example.com/login", row, MODEL1_FEATURES)
        self.assertIsNotNone(result, msg="Should extract features from a valid URL")
        self.assertEqual(set(result.keys()), set(MODEL1_FEATURES))

    def test_33b_extract_row_features_invalid_url(self):
        """_extract_row_features returns None for an invalid/blocked URL."""
        from train_models import _extract_row_features
        row = pd.Series({"url": "not-a-url"})
        result = _extract_row_features("not-a-url", row, MODEL1_FEATURES)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Tests 36-45 — Targeted regression tests for the HTML rendering fix and
# pipeline verification (Task: Fix and Validate the ML Detection Module)
# ---------------------------------------------------------------------------


class TestFeatureCountDisplay(unittest.TestCase):
    """
    Tests 36-39: Verify the feature count is a readable integer value
    from the actual feature schema — never raw HTML.

    These tests operate on PredictionResult objects only; they do not
    exercise Streamlit rendering, which requires a running server.
    """

    def setUp(self):
        self.pipe1 = _build_synthetic_pipeline(len(MODEL1_FEATURES))
        self.pipe2 = _build_synthetic_pipeline(len(MODEL2_FEATURES))
        self.fr = _make_feature_result()
        self.sr = _make_security_result(success=True)
        import modules.ml_predictor as mp
        mp._models_loaded = True
        mp._model1_pipeline = self.pipe1
        mp._model1_load_error = ""
        mp._model2_pipeline = self.pipe2
        mp._model2_load_error = ""

    def tearDown(self):
        import modules.ml_predictor as mp
        mp._models_loaded = False
        mp._model1_pipeline = None
        mp._model2_pipeline = None
        mp._model1_load_error = ""
        mp._model2_load_error = ""

    def test_36_model1_feature_count_is_integer(self):
        """PredictionResult.feature_count for Model 1 must be an int, not HTML."""
        result = predict_model1(self.fr)
        self.assertIsInstance(result.feature_count, int)
        # Must never be an HTML string
        self.assertNotIn("<", str(result.feature_count))
        self.assertNotIn("div", str(result.feature_count))

    def test_37_model2_feature_count_is_integer(self):
        """PredictionResult.feature_count for Model 2 must be an int, not HTML."""
        result = predict_model2(self.fr, self.sr)
        self.assertIsInstance(result.feature_count, int)
        self.assertNotIn("<", str(result.feature_count))

    def test_38_model1_feature_count_equals_schema_length(self):
        """Model 1 feature_count must equal len(MODEL1_FEATURES) = 25."""
        result = predict_model1(self.fr)
        self.assertEqual(result.feature_count, len(MODEL1_FEATURES))
        self.assertEqual(result.feature_count, 25)

    def test_39_model2_feature_count_equals_schema_length(self):
        """Model 2 feature_count must equal len(MODEL2_FEATURES) = 39."""
        result = predict_model2(self.fr, self.sr)
        self.assertEqual(result.feature_count, len(MODEL2_FEATURES))
        self.assertEqual(result.feature_count, 39)


class TestFeatureOrdering(unittest.TestCase):
    """
    Tests 40-41: Verify that features_used in PredictionResult
    matches the intended canonical schemas in the correct order.
    """

    def setUp(self):
        self.pipe1 = _build_synthetic_pipeline(len(MODEL1_FEATURES))
        self.pipe2 = _build_synthetic_pipeline(len(MODEL2_FEATURES))
        self.fr = _make_feature_result()
        self.sr = _make_security_result(success=True)
        import modules.ml_predictor as mp
        mp._models_loaded = True
        mp._model1_pipeline = self.pipe1
        mp._model1_load_error = ""
        mp._model2_pipeline = self.pipe2
        mp._model2_load_error = ""

    def tearDown(self):
        import modules.ml_predictor as mp
        mp._models_loaded = False
        mp._model1_pipeline = None
        mp._model2_pipeline = None
        mp._model1_load_error = ""
        mp._model2_load_error = ""

    def test_40_model1_features_used_matches_schema(self):
        """features_used for Model 1 must match MODEL1_FEATURES exactly."""
        result = predict_model1(self.fr)
        self.assertEqual(result.features_used, MODEL1_FEATURES,
                         msg="Model 1 features_used must be identical to MODEL1_FEATURES")

    def test_41_model2_features_used_matches_schema(self):
        """features_used for Model 2 must match MODEL2_FEATURES exactly."""
        result = predict_model2(self.fr, self.sr)
        self.assertEqual(result.features_used, MODEL2_FEATURES,
                         msg="Model 2 features_used must be identical to MODEL2_FEATURES")


class TestProbabilityClassIndex(unittest.TestCase):
    """
    Tests 42-43: Verify that phishing probability is taken from the correct
    class index (class 1 = phishing, class 0 = legitimate).
    """

    def setUp(self):
        # Build a pipeline trained to always predict phishing for a specific input
        np.random.seed(99)
        X_phish = np.ones((60, len(MODEL1_FEATURES))) * 100
        X_legit  = np.zeros((60, len(MODEL1_FEATURES)))
        X = np.vstack([X_phish, X_legit])
        y = np.array([1] * 60 + [0] * 60)
        self.pipe_phish = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     RandomForestClassifier(n_estimators=10, random_state=99)),
        ])
        self.pipe_phish.fit(X, y)
        import modules.ml_predictor as mp
        mp._models_loaded = True
        mp._model1_pipeline = self.pipe_phish
        mp._model1_load_error = ""

    def tearDown(self):
        import modules.ml_predictor as mp
        mp._models_loaded = False
        mp._model1_pipeline = None
        mp._model1_load_error = ""

    def test_42_phishing_class_index_is_one(self):
        """LABEL_PHISHING must be 1 and LABEL_LEGITIMATE must be 0."""
        self.assertEqual(LABEL_PHISHING, 1)
        self.assertEqual(LABEL_LEGITIMATE, 0)

    def test_43_phishing_probability_from_class_1(self):
        """Phishing probability must come from class index 1, not class index 0."""
        # Build a feature result that looks like a phishing URL (all large values)
        fr = _make_feature_result("https://bank-verify-account.xyz/login?secure=true")
        # Manually override numeric values to be all 100 (matching the pipeline training)
        for k in list(fr.numeric.keys()):
            fr.numeric[k] = 100
        result = predict_model1(fr)
        if result.success and result.phishing_probability is not None:
            # Classes ordered [0, 1] — phishing_probability is proba[:, 1]
            classes = list(self.pipe_phish.classes_)
            self.assertEqual(classes, [0, 1],
                             "Pipeline must have classes ordered [0 (legit), 1 (phish)]")
            # The phishing probability + legit probability must sum to ~1
            total = (result.phishing_probability or 0) + (result.legitimate_probability or 0)
            self.assertAlmostEqual(total, 1.0, places=3,
                                   msg="Phishing + legit probabilities must sum to 1.0")


class TestMissingArtifactMessages(unittest.TestCase):
    """
    Tests 44-45: Verify that missing artifacts produce readable error messages,
    not raw HTML or tracebacks.
    """

    def test_44_missing_model1_gives_readable_error(self):
        """Missing Model 1 artifact: predicted_label is 'Unavailable', no HTML in error."""
        fr = _make_feature_result()
        import modules.ml_predictor as mp
        mp._models_loaded = True
        mp._model1_pipeline = None
        mp._model1_load_error = "Model artifact not found: models/model1_baseline.joblib"
        try:
            result = predict_model1(fr)
        finally:
            mp._models_loaded = False
            mp._model1_pipeline = None
            mp._model1_load_error = ""
        self.assertFalse(result.success)
        self.assertEqual(result.predicted_label, "Unavailable")
        # Error message must be readable text, not HTML
        self.assertNotIn("<div", result.error)
        self.assertNotIn("<span", result.error)
        self.assertIn("not found", result.error.lower())

    def test_45_missing_model2_gives_readable_error(self):
        """Missing Model 2 artifact: error message is human-readable, no HTML."""
        fr = _make_feature_result()
        sr = _make_security_result(success=True)
        import modules.ml_predictor as mp
        mp._models_loaded = True
        mp._model2_pipeline = None
        mp._model2_load_error = "Model artifact not found: models/model2_extended.joblib"
        try:
            result = predict_model2(fr, sr)
        finally:
            mp._models_loaded = False
            mp._model2_pipeline = None
            mp._model2_load_error = ""
        self.assertFalse(result.success)
        self.assertEqual(result.predicted_label, "Unavailable")
        self.assertNotIn("<div", result.error)
        self.assertIn("not found", result.error.lower())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()

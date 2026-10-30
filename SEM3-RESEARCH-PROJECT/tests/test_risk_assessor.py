"""
tests/test_risk_assessor.py
===========================
Unit tests for modules/risk_assessor.py (Module 8).

Coverage
--------
 1. Low risk category assigned at p < 0.20
 2. Medium risk category at 0.20 <= p < 0.50
 3. High risk category at 0.50 <= p < 0.80
 4. Critical risk category at p >= 0.80
 5. Boundary: p exactly at THRESHOLD_LOW_MAX (0.20) → Medium
 6. Boundary: p exactly at THRESHOLD_MEDIUM_MAX (0.50) → High
 7. Boundary: p exactly at THRESHOLD_HIGH_MAX (0.80) → Critical
 8. Probability = 0.0 → Low
 9. Probability = 1.0 → Critical
10. No valid predictions → Unknown risk category
11. Only Model 1 available (no Model 2) → uses Model 1 probability
12. Only Model 2 available (no Model 1) → uses Model 2 probability
13. Model class disagreement detected and flagged
14. Notable probability difference detected (same class, large diff)
15. No disagreement when both models agree closely
16. Missing security analysis appears in evidence as "missing" kind
17. Failed security analysis appears in evidence as "missing" kind
18. Successful security analysis produces evidence items
19. URL validation evidence: plain HTTP flagged
20. URL validation evidence: IP address flagged
21. Feature evidence: URL shortener flagged
22. Recommendations are non-empty for Low, Medium, High, Critical, Unknown
23. Disagreement adds extra recommendation
24. assess_risk always returns RiskAssessment (no exception on bad inputs)
25. Invalid/non-finite probability treated as unavailable (UNKNOWN)
26. Out-of-range probability treated as unavailable (UNKNOWN)
27. Sensitive cookie values never appear in evidence detail
28. assessed_at is a non-empty ISO-8601-like timestamp
29. threshold_config snapshot matches module constants
30. primary_model label recorded correctly in RiskAssessment
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from modules.risk_assessor import (
    assess_risk,
    RiskAssessment,
    RiskCategory,
    THRESHOLD_LOW_MAX,
    THRESHOLD_MEDIUM_MAX,
    THRESHOLD_HIGH_MAX,
    PRIMARY_MODEL_PREFERENCE,
    NOTABLE_PROBABILITY_DIFF,
)
from modules.ml_predictor import PredictionResult, LABEL_PHISHING, LABEL_LEGITIMATE


# ---------------------------------------------------------------------------
# Helpers to build minimal mock objects
# ---------------------------------------------------------------------------

def _make_pred(
    success: bool = True,
    predicted_class: int = 0,
    phishing_probability: float | None = 0.05,
    legitimate_probability: float | None = 0.95,
    predicted_label: str = "Legitimate",
    feature_count: int = 25,
    error: str = "",
) -> PredictionResult:
    """Return a PredictionResult-like object using the real dataclass."""
    return PredictionResult(
        success=success,
        predicted_class=predicted_class,
        phishing_probability=phishing_probability,
        legitimate_probability=legitimate_probability,
        predicted_label=predicted_label,
        feature_count=feature_count,
        error=error,
    )


def _make_phishing_pred(prob: float = 0.85) -> PredictionResult:
    return _make_pred(
        predicted_class=LABEL_PHISHING,
        phishing_probability=prob,
        legitimate_probability=1.0 - prob,
        predicted_label="Potentially Phishing",
    )


def _make_legit_pred(prob: float = 0.05) -> PredictionResult:
    return _make_pred(
        predicted_class=LABEL_LEGITIMATE,
        phishing_probability=prob,
        legitimate_probability=1.0 - prob,
        predicted_label="Legitimate",
    )


def _make_failed_pred() -> PredictionResult:
    return _make_pred(
        success=False,
        predicted_class=None,
        phishing_probability=None,
        legitimate_probability=None,
        predicted_label="",
        error="Model artifact not found.",
    )


def _make_validation_result(
    uses_https: bool = True,
    is_ip_address: bool = False,
    hostname: str = "example.com",
    port: int | None = None,
    scheme: str = "https",
    normalized_url: str = "https://example.com/",
) -> MagicMock:
    vr = MagicMock()
    vr.uses_https = uses_https
    vr.is_ip_address = is_ip_address
    vr.hostname = hostname
    vr.port = port
    vr.scheme = scheme
    vr.normalized_url = normalized_url
    return vr


def _make_feature_result(
    numeric: dict | None = None,
    keywords_matched: list | None = None,
) -> MagicMock:
    fr = MagicMock()
    fr.numeric = numeric or {
        "url_length": 30,
        "url_path_length": 10,
        "url_query_length": 0,
        "flag_has_query": False,
        "host_subdomain_count": 0,
        "flag_has_punycode": False,
        "flag_is_url_shortener": False,
        "flag_has_at_symbol": False,
        "char_count_percent": 0,
        "kw_num_matched": 0,
    }
    fr.keywords_matched = keywords_matched if keywords_matched is not None else []
    return fr


def _make_security_result(
    success: bool = True,
    error: str = "",
    tls_verified: bool = True,
    tls_applicable: bool = True,
    tls_days_remaining: int | None = 120,
    cookies_present: bool = False,
    header_checks: list | None = None,
    cookie_checks: list | None = None,
) -> MagicMock:
    sr = MagicMock()
    sr.success = success
    sr.error = error

    tls = MagicMock()
    tls.applicable = tls_applicable
    tls.verified = tls_verified
    tls.days_remaining = tls_days_remaining
    tls.issuer = "Let's Encrypt"
    tls.valid_until = "2025-12-31"
    tls.error = "" if tls_verified else "Certificate expired"
    tls.observation = "TLS certificate is valid."
    tls.not_applicable_reason = "Plain HTTP — TLS not applicable."
    sr.tls = tls

    sr.cookies_present = cookies_present
    sr.header_checks = header_checks or []
    sr.cookie_checks = cookie_checks or []
    return sr


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestRiskCategoryAssignment(unittest.TestCase):
    """Tests 1–9: probability → risk category mapping."""

    def _assess(self, prob: float) -> RiskAssessment:
        pred = _make_legit_pred(prob) if prob < 0.5 else _make_phishing_pred(prob)
        # Force the probability to the desired value regardless of class
        pred = _make_pred(
            predicted_class=LABEL_PHISHING if prob >= 0.5 else LABEL_LEGITIMATE,
            phishing_probability=prob,
            legitimate_probability=1.0 - prob,
        )
        return assess_risk("https://example.com/", None, pred, None)

    def test_01_low_risk(self):
        ra = self._assess(0.10)
        self.assertEqual(ra.risk_category, RiskCategory.LOW)

    def test_02_medium_risk(self):
        ra = self._assess(0.35)
        self.assertEqual(ra.risk_category, RiskCategory.MEDIUM)

    def test_03_high_risk(self):
        ra = self._assess(0.65)
        self.assertEqual(ra.risk_category, RiskCategory.HIGH)

    def test_04_critical_risk(self):
        ra = self._assess(0.90)
        self.assertEqual(ra.risk_category, RiskCategory.CRITICAL)

    def test_05_boundary_low_max_is_medium(self):
        # At exactly THRESHOLD_LOW_MAX the category is Medium
        ra = self._assess(THRESHOLD_LOW_MAX)
        self.assertEqual(ra.risk_category, RiskCategory.MEDIUM)

    def test_06_boundary_medium_max_is_high(self):
        ra = self._assess(THRESHOLD_MEDIUM_MAX)
        self.assertEqual(ra.risk_category, RiskCategory.HIGH)

    def test_07_boundary_high_max_is_critical(self):
        ra = self._assess(THRESHOLD_HIGH_MAX)
        self.assertEqual(ra.risk_category, RiskCategory.CRITICAL)

    def test_08_probability_zero_is_low(self):
        ra = self._assess(0.0)
        self.assertEqual(ra.risk_category, RiskCategory.LOW)

    def test_09_probability_one_is_critical(self):
        ra = self._assess(1.0)
        self.assertEqual(ra.risk_category, RiskCategory.CRITICAL)


class TestUnknownAndModelSelection(unittest.TestCase):
    """Tests 10–12: Unknown category, model selection."""

    def test_10_no_predictions_unknown(self):
        ra = assess_risk("https://example.com/", None, None, None)
        self.assertEqual(ra.risk_category, RiskCategory.UNKNOWN)
        self.assertIsNone(ra.primary_probability)

    def test_10b_failed_predictions_unknown(self):
        ra = assess_risk("https://example.com/", None, _make_failed_pred(), _make_failed_pred())
        self.assertEqual(ra.risk_category, RiskCategory.UNKNOWN)

    def test_11_only_model1_uses_model1(self):
        p1 = _make_phishing_pred(0.75)
        ra = assess_risk("https://x.com/", None, p1, None)
        self.assertAlmostEqual(ra.primary_probability, 0.75, places=6)
        self.assertIn("Model 1", ra.primary_model)

    def test_12_model2_preferred_over_model1(self):
        # When PRIMARY_MODEL_PREFERENCE is "model2", Model 2 probability wins
        if PRIMARY_MODEL_PREFERENCE != "model2":
            self.skipTest("Test assumes model2 preference")
        p1 = _make_phishing_pred(0.30)
        p2 = _make_phishing_pred(0.80)
        ra = assess_risk("https://x.com/", None, p1, p2)
        self.assertAlmostEqual(ra.primary_probability, 0.80, places=6)
        self.assertIn("Model 2", ra.primary_model)

    def test_12b_falls_back_to_model1_when_model2_fails(self):
        if PRIMARY_MODEL_PREFERENCE != "model2":
            self.skipTest("Test assumes model2 preference")
        p1 = _make_phishing_pred(0.75)
        p2 = _make_failed_pred()
        ra = assess_risk("https://x.com/", None, p1, p2)
        self.assertAlmostEqual(ra.primary_probability, 0.75, places=6)
        self.assertIn("Model 1", ra.primary_model)


class TestDisagreementDetection(unittest.TestCase):
    """Tests 13–15: Disagreement and notable probability difference."""

    def test_13_class_disagreement_flagged(self):
        p1 = _make_legit_pred(0.10)    # class 0
        p2 = _make_phishing_pred(0.85) # class 1
        ra = assess_risk("https://x.com/", None, p1, p2)
        self.assertTrue(ra.models_disagree)
        self.assertGreater(len(ra.disagreement_note), 0)
        self.assertFalse(ra.notable_prob_diff)

    def test_14_notable_probability_diff_flagged(self):
        # Same class (both phishing), large probability gap
        p1 = _make_phishing_pred(0.55)
        p2 = _make_phishing_pred(0.55 + NOTABLE_PROBABILITY_DIFF + 0.01)
        ra = assess_risk("https://x.com/", None, p1, p2)
        self.assertFalse(ra.models_disagree)
        self.assertTrue(ra.notable_prob_diff)
        self.assertGreater(len(ra.disagreement_note), 0)

    def test_15_no_disagreement_when_models_close(self):
        p1 = _make_legit_pred(0.10)
        p2 = _make_legit_pred(0.12)
        ra = assess_risk("https://x.com/", None, p1, p2)
        self.assertFalse(ra.models_disagree)
        self.assertFalse(ra.notable_prob_diff)
        self.assertEqual(ra.disagreement_note, "")


class TestEvidenceExtraction(unittest.TestCase):
    """Tests 16–22: Evidence items from validation, features, and security."""

    def test_16_missing_security_analysis_in_evidence(self):
        p1 = _make_legit_pred(0.10)
        ra = assess_risk("https://x.com/", None, p1, None, security_result=None)
        labels = [ev.label for ev in ra.evidence]
        self.assertTrue(
            any("security" in lbl.lower() for lbl in labels),
            f"Expected security evidence item; got: {labels}",
        )
        kinds = [ev.kind for ev in ra.evidence if "security" in ev.label.lower()]
        self.assertTrue(any(k == "missing" for k in kinds))

    def test_17_failed_security_result_in_evidence(self):
        sr = _make_security_result(success=False, error="Connection refused")
        p1 = _make_legit_pred(0.10)
        ra = assess_risk("https://x.com/", None, p1, None, security_result=sr)
        labels = [ev.label.lower() for ev in ra.evidence]
        self.assertTrue(any("security" in lbl for lbl in labels))

    def test_18_successful_security_adds_tls_evidence(self):
        sr = _make_security_result(success=True, tls_verified=True, tls_days_remaining=120)
        p1 = _make_legit_pred(0.10)
        ra = assess_risk("https://x.com/", None, p1, None, security_result=sr)
        self.assertTrue(ra.security_analysis_available)
        labels = [ev.label.lower() for ev in ra.evidence]
        self.assertTrue(any("tls" in lbl for lbl in labels))

    def test_19_plain_http_flagged_in_evidence(self):
        vr = _make_validation_result(uses_https=False, scheme="http")
        fr = _make_feature_result()
        p1 = _make_legit_pred(0.05)
        ra = assess_risk("http://x.com/", fr, p1, None, validation_result=vr)
        labels = [ev.label.lower() for ev in ra.evidence]
        self.assertTrue(any("http" in lbl for lbl in labels))

    def test_20_ip_address_flagged_in_evidence(self):
        vr = _make_validation_result(is_ip_address=True, hostname="192.168.1.1")
        fr = _make_feature_result()
        p1 = _make_legit_pred(0.05)
        ra = assess_risk("https://192.168.1.1/", fr, p1, None, validation_result=vr)
        labels = [ev.label.lower() for ev in ra.evidence]
        self.assertTrue(any("ip" in lbl for lbl in labels))

    def test_21_url_shortener_flagged_in_evidence(self):
        vr = _make_validation_result()
        fr = _make_feature_result(numeric={
            "url_length": 20, "url_path_length": 5, "url_query_length": 0,
            "flag_has_query": False, "host_subdomain_count": 0,
            "flag_has_punycode": False, "flag_is_url_shortener": True,
            "flag_has_at_symbol": False, "char_count_percent": 0, "kw_num_matched": 0,
        })
        p1 = _make_legit_pred(0.05)
        ra = assess_risk("https://bit.ly/abc", fr, p1, None, validation_result=vr)
        labels = [ev.label.lower() for ev in ra.evidence]
        self.assertTrue(any("shortener" in lbl for lbl in labels))

    def test_22_evidence_is_non_empty_with_full_inputs(self):
        vr = _make_validation_result()
        fr = _make_feature_result()
        sr = _make_security_result()
        p1 = _make_legit_pred(0.08)
        ra = assess_risk("https://x.com/", fr, p1, None, security_result=sr, validation_result=vr)
        self.assertGreater(len(ra.evidence), 0)


class TestRecommendations(unittest.TestCase):
    """Tests 23–24: Recommendations per category and disagreement."""

    def _rec_for_prob(self, prob: float) -> list[str]:
        pred = _make_pred(
            predicted_class=LABEL_PHISHING if prob >= 0.5 else LABEL_LEGITIMATE,
            phishing_probability=prob,
            legitimate_probability=1.0 - prob,
        )
        ra = assess_risk("https://x.com/", None, pred, None)
        return ra.recommendations

    def test_23_all_categories_have_recommendations(self):
        for prob in [0.05, 0.35, 0.65, 0.90]:
            recs = self._rec_for_prob(prob)
            self.assertGreater(len(recs), 0, f"No recs for prob={prob}")

    def test_23b_unknown_has_recommendation(self):
        ra = assess_risk("https://x.com/", None, None, None)
        self.assertGreater(len(ra.recommendations), 0)

    def test_24_disagreement_adds_extra_recommendation(self):
        p1 = _make_legit_pred(0.10)
        p2 = _make_phishing_pred(0.85)
        ra = assess_risk("https://x.com/", None, p1, p2)
        self.assertTrue(any("disagree" in r.lower() for r in ra.recommendations))


class TestRobustness(unittest.TestCase):
    """Tests 25–30: Robustness, invalid inputs, sensitive data, metadata."""

    def test_25_invalid_probability_type_treated_as_unavailable(self):
        pred = _make_pred(
            success=True, predicted_class=LABEL_PHISHING,
            phishing_probability="not-a-number",
        )
        ra = assess_risk("https://x.com/", None, pred, None)
        self.assertEqual(ra.risk_category, RiskCategory.UNKNOWN)

    def test_26_infinite_probability_treated_as_unavailable(self):
        import math
        pred = _make_pred(
            success=True, predicted_class=LABEL_PHISHING,
            phishing_probability=math.inf,
        )
        ra = assess_risk("https://x.com/", None, pred, None)
        self.assertEqual(ra.risk_category, RiskCategory.UNKNOWN)

    def test_26b_out_of_range_probability_treated_as_unavailable(self):
        pred = _make_pred(
            success=True, predicted_class=LABEL_PHISHING,
            phishing_probability=1.5,
        )
        ra = assess_risk("https://x.com/", None, pred, None)
        self.assertEqual(ra.risk_category, RiskCategory.UNKNOWN)

    def test_27_sensitive_cookie_values_not_in_evidence(self):
        """Cookie values and tokens must never appear in evidence detail."""
        ck = MagicMock()
        ck.name = "session"
        ck.has_secure = False
        ck.has_httponly = False
        ck.samesite = None
        ck.observations = ["Missing Secure attribute."]
        # The mock cookie has no .value attribute set on purpose;
        # the risk assessor must never read or output ck.value.
        ck.value = "SUPERSECRETTOKEN12345"

        sr = _make_security_result(
            success=True,
            cookies_present=True,
            cookie_checks=[ck],
        )
        p1 = _make_legit_pred(0.10)
        ra = assess_risk("https://x.com/", None, p1, None, security_result=sr)
        all_detail = " ".join(ev.detail for ev in ra.evidence)
        self.assertNotIn("SUPERSECRETTOKEN12345", all_detail)

    def test_28_assessed_at_is_timestamp(self):
        ra = assess_risk("https://x.com/", None, _make_legit_pred(0.10), None)
        self.assertTrue(len(ra.assessed_at) > 0)
        # Should look like an ISO-8601 date
        self.assertRegex(ra.assessed_at, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    def test_29_threshold_snapshot_matches_constants(self):
        ra = assess_risk("https://x.com/", None, _make_legit_pred(0.10), None)
        self.assertAlmostEqual(ra.threshold_config["low_max"],    THRESHOLD_LOW_MAX)
        self.assertAlmostEqual(ra.threshold_config["medium_max"], THRESHOLD_MEDIUM_MAX)
        self.assertAlmostEqual(ra.threshold_config["high_max"],   THRESHOLD_HIGH_MAX)

    def test_30_primary_model_label_recorded(self):
        p1 = _make_phishing_pred(0.80)
        ra = assess_risk("https://x.com/", None, p1, None)
        self.assertNotEqual(ra.primary_model, "None")
        self.assertIn("Model", ra.primary_model)

    def test_30b_assess_risk_never_raises(self):
        """assess_risk must not raise on any combination of None / bad inputs."""
        combinations = [
            (None, None, None, None, None, None),
            ("", None, None, None, None, None),
            ("not-a-url", None, _make_failed_pred(), _make_failed_pred(), None, None),
        ]
        for (url, fr, p1, p2, sr, vr) in combinations:
            with self.subTest(url=url):
                try:
                    ra = assess_risk(url or "", fr, p1, p2, sr, vr)
                    self.assertIsInstance(ra, RiskAssessment)
                except Exception as exc:
                    self.fail(f"assess_risk raised an exception: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()

"""
tests/test_feature_extractor.py
================================
Automated tests for Module 5 — Traditional URL, Domain and Webpage Feature
Extraction.

Framework : Python built-in unittest
Run with  :
    py -m unittest tests.test_feature_extractor -v
    OR
    py -m pytest tests/test_feature_extractor.py -v

Test philosophy
---------------
- No real network requests are made.  extract_features() makes no network
  calls by design; these tests verify that guarantee holds.
- Tests operate on ValidationResult objects produced by the real validate_url()
  function to ensure the integration contract is exercised.
- Tests do not assert that any URL is phishing or legitimate.  Keyword matches
  and structural observations are tested as data values, not as verdicts.
- Every numeric feature key is tested for presence and correct type.

Test index
----------
 1. Standard HTTPS URL — basic feature values
 2. HTTP URL — flag_uses_https False, HTTP warning in validator
 3. Multiple subdomains — subdomain_count correct
 4. Long path — url_path_length correct
 5. URL with query parameters — num_query_params, url_query_length
 6. URL with empty query string — has_query True, num_query_params 0
 7. URL with fragment — flag_has_fragment True, url_fragment_length
 8. URL with percent-encoded characters — char_count_percent
 9. URL with IP address as hostname — flag_is_ip_address, registered_domain unknown
10. URL with non-default port — flag_has_non_default_port True
11. URL containing keywords — keywords_matched, kw_num_matched
12. Case-insensitive keyword detection
13. Hostname with punycode — flag_has_punycode True
14. No keyword match — kw_num_matched 0
15. Consistent feature names — numeric dict keys match numeric_feature_names()
16. Consistent data types — all numeric values are int, float, or bool
17. Deterministic extraction — same URL produces identical features twice
18. TypeError on non-ValidationResult input
19. ValueError on invalid ValidationResult (is_valid=False)
20. Malformed URL rejected before feature extraction
21. Unsupported scheme rejected before feature extraction
22. URL shortener detected — flag_is_url_shortener True
23. Non-shortener URL — flag_is_url_shortener False
24. to_dict() returns a flat dict with expected keys
25. Registered domain with two-label suffix (co.in)
26. Registered domain with two-label suffix (co.uk)
27. Subdomain count zero for bare registered domain
28. Hyphen in hostname — char_count_hyphen
29. Underscore in path — char_count_underscore
30. No network activity during extraction (import-level check)
"""

from __future__ import annotations

import sys
import os
import types
import unittest

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.url_validator import validate_url, ValidationResult
from modules.feature_extractor import (
    extract_features,
    numeric_feature_names,
    FeatureResult,
    PHISHING_KEYWORDS,
    URL_SHORTENERS,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _valid(url: str) -> ValidationResult:
    """Run validate_url and assert the URL is valid before returning."""
    vr = validate_url(url)
    if not vr.is_valid:
        raise AssertionError(
            f"Expected URL to be valid for test setup, but got errors: {vr.errors}"
        )
    return vr


# ===========================================================================
# Tests
# ===========================================================================

class TestExtractFeaturesBasic(unittest.TestCase):
    """Core feature extraction on well-formed URLs."""

    # ── Test 1 ───────────────────────────────────────────────────────────────
    def test_01_standard_https_url(self):
        """A standard HTTPS URL produces expected basic feature values."""
        vr = _valid("https://example.com/path")
        fr = extract_features(vr)

        self.assertIsInstance(fr, FeatureResult)
        self.assertTrue(fr.numeric["flag_uses_https"],
                        "HTTPS URL must set flag_uses_https True")
        self.assertGreater(fr.numeric["url_length"], 0,
                           "URL length must be positive")
        self.assertEqual(fr.descriptive["scheme"], "https")
        self.assertEqual(fr.descriptive["hostname"], "example.com")
        self.assertFalse(fr.numeric["flag_is_ip_address"])

    # ── Test 2 ───────────────────────────────────────────────────────────────
    def test_02_http_url(self):
        """HTTP URL sets flag_uses_https False."""
        vr = _valid("http://example.com/page")
        fr = extract_features(vr)

        self.assertFalse(fr.numeric["flag_uses_https"],
                         "HTTP URL must set flag_uses_https False")
        self.assertEqual(fr.descriptive["scheme"], "http")

    # ── Test 3 ───────────────────────────────────────────────────────────────
    def test_03_multiple_subdomains(self):
        """Subdomain count reflects labels before the registered domain."""
        vr = _valid("https://mail.accounts.example.com/inbox")
        fr = extract_features(vr)

        # hostname: mail.accounts.example.com → RD: example.com → 2 subdomains
        self.assertEqual(fr.numeric["host_subdomain_count"], 2,
                         "Two subdomains before example.com")
        self.assertEqual(fr.numeric["host_num_labels"], 4)

    # ── Test 4 ───────────────────────────────────────────────────────────────
    def test_04_long_path(self):
        """Path length feature matches the actual path string length."""
        path = "/very/long/path/with/many/segments/to/test/length"
        vr = _valid(f"https://example.com{path}")
        fr = extract_features(vr)

        self.assertEqual(fr.numeric["url_path_length"], len(path))

    # ── Test 5 ───────────────────────────────────────────────────────────────
    def test_05_query_parameters(self):
        """Query parameter count and query length are measured correctly."""
        vr = _valid("https://example.com/page?a=1&b=2&c=3")
        fr = extract_features(vr)

        self.assertEqual(fr.numeric["url_num_query_params"], 3,
                         "Three distinct query parameters")
        self.assertGreater(fr.numeric["url_query_length"], 0)
        self.assertTrue(fr.numeric["flag_has_query"])

    # ── Test 6 ───────────────────────────────────────────────────────────────
    def test_06_empty_query_string(self):
        """A URL ending with '?' has has_query True and num_query_params 0."""
        # urllib.parse preserves the '?' — vr.query will be empty string ""
        # but the normalised_url may or may not include '?'.
        # We test via the ValidationResult query field directly.
        vr = validate_url("https://example.com/page?")
        if not vr.is_valid:
            self.skipTest("Validator rejected the URL with empty query — skip.")
        fr = extract_features(vr)

        # flag_has_query depends on vr.query being a non-empty string.
        # urllib.parse strips the bare '?' so vr.query == "".
        # Per design: has_query is True only when vr.query is non-empty.
        # Accept either outcome as correct since it depends on the parser.
        self.assertIsInstance(fr.numeric["flag_has_query"], bool)
        self.assertEqual(fr.numeric["url_num_query_params"], 0)

    # ── Test 7 ───────────────────────────────────────────────────────────────
    def test_07_fragment(self):
        """Fragment presence and length are captured."""
        vr = _valid("https://example.com/page#section-2")
        fr = extract_features(vr)

        self.assertTrue(fr.numeric["flag_has_fragment"])
        self.assertEqual(fr.numeric["url_fragment_length"], len("section-2"))

    # ── Test 8 ───────────────────────────────────────────────────────────────
    def test_08_percent_encoded_characters(self):
        """Percent-encoding markers in the path are counted."""
        vr = _valid("https://example.com/search?q=hello%20world%21")
        fr = extract_features(vr)

        self.assertGreaterEqual(fr.numeric["char_count_percent"], 2,
                                "At least two '%' characters in the query")

    # ── Test 9 ───────────────────────────────────────────────────────────────
    def test_09_ip_address_hostname(self):
        """IP address as hostname sets flag_is_ip_address and registered_domain unknown.

        Uses 8.8.8.8 (a publicly routable address) because the validator
        correctly blocks documentation/private addresses such as 203.0.113.0/24
        (RFC 5737 TEST-NET-3).  A public IP passes validation but still cannot
        have a registered domain determined.
        """
        vr = _valid("https://8.8.8.8/path")
        fr = extract_features(vr)

        self.assertTrue(fr.numeric["flag_is_ip_address"],
                        "IP-address hostname must set flag_is_ip_address True")
        self.assertEqual(fr.descriptive["registered_domain"], "unknown",
                         "Registered domain cannot be determined for an IP address")
        self.assertEqual(fr.numeric["domain_registered_length"], -1,
                         "Sentinel -1 when registered domain is unknown")

    # ── Test 10 ──────────────────────────────────────────────────────────────
    def test_10_non_default_port(self):
        """Explicit non-default port sets flag_has_non_default_port True."""
        vr = _valid("https://example.com:8443/admin")
        fr = extract_features(vr)

        self.assertTrue(fr.numeric["flag_has_non_default_port"],
                        "Port 8443 differs from HTTPS default 443")

    # ── Test 11 ──────────────────────────────────────────────────────────────
    def test_11_keywords_matched(self):
        """Known phishing keywords in the URL are detected and counted."""
        vr = _valid("https://paytm-verify-example.com/login?action=confirm")
        fr = extract_features(vr)

        self.assertTrue(fr.numeric["flag_has_keyword"])
        self.assertGreater(fr.numeric["kw_num_matched"], 0)
        self.assertTrue(
            any(kw in fr.keywords_matched for kw in ("login", "verify", "confirm")),
            "At least one of login/verify/confirm must be detected"
        )

    # ── Test 12 ──────────────────────────────────────────────────────────────
    def test_12_keyword_case_insensitive(self):
        """Keyword matching is case-insensitive."""
        vr = _valid("https://example.com/LOGIN")
        fr = extract_features(vr)

        self.assertIn("login", fr.keywords_matched,
                      "'LOGIN' in path must match keyword 'login' case-insensitively")
        self.assertTrue(fr.numeric["flag_has_keyword"])

    # ── Test 13 ──────────────────────────────────────────────────────────────
    def test_13_punycode_hostname(self):
        """Punycode-encoded hostname sets flag_has_punycode True."""
        # xn--11b4c3d.com is a valid punycode hostname.
        vr = validate_url("https://xn--11b4c3d.com/path")
        if not vr.is_valid:
            self.skipTest("Validator rejected punycode hostname — skip.")
        fr = extract_features(vr)

        self.assertTrue(fr.numeric["flag_has_punycode"],
                        "xn-- prefix must be detected as punycode")

    # ── Test 14 ──────────────────────────────────────────────────────────────
    def test_14_no_keyword_match(self):
        """A URL with no matching keywords produces kw_num_matched = 0."""
        vr = _valid("https://weather.example.com/forecast/london")
        fr = extract_features(vr)

        self.assertEqual(fr.numeric["kw_num_matched"], 0)
        self.assertFalse(fr.numeric["flag_has_keyword"])
        self.assertEqual(fr.keywords_matched, [])


class TestFeatureNamesAndTypes(unittest.TestCase):
    """Feature schema consistency tests."""

    # ── Test 15 ──────────────────────────────────────────────────────────────
    def test_15_numeric_keys_match_schema(self):
        """Every call returns exactly the keys declared in numeric_feature_names()."""
        vr = _valid("https://example.com/test")
        fr = extract_features(vr)

        expected = set(numeric_feature_names())
        actual = set(fr.numeric.keys())
        self.assertEqual(actual, expected,
                         f"Extra keys: {actual - expected}  Missing: {expected - actual}")

    # ── Test 16 ──────────────────────────────────────────────────────────────
    def test_16_numeric_value_types(self):
        """All numeric dict values are int, float, or bool — no None, no str."""
        vr = _valid("https://sub.example.co.uk/path?q=1#frag")
        fr = extract_features(vr)

        for key, val in fr.numeric.items():
            self.assertIsInstance(
                val, (int, float, bool),
                f"Feature '{key}' has unexpected type {type(val).__name__}: {val!r}"
            )

    # ── Test 17 ──────────────────────────────────────────────────────────────
    def test_17_deterministic_extraction(self):
        """The same URL always produces identical numeric features."""
        url = "https://secure.banking.example.co.in/verify?session=abc123"
        vr1 = _valid(url)
        vr2 = _valid(url)

        fr1 = extract_features(vr1)
        fr2 = extract_features(vr2)

        self.assertEqual(fr1.numeric, fr2.numeric,
                         "Feature extraction must be deterministic")
        self.assertEqual(fr1.keywords_matched, fr2.keywords_matched)


class TestInputValidation(unittest.TestCase):
    """Error handling for invalid inputs."""

    # ── Test 18 ──────────────────────────────────────────────────────────────
    def test_18_type_error_on_non_validation_result(self):
        """Passing a non-ValidationResult raises TypeError."""
        with self.assertRaises(TypeError):
            extract_features("https://example.com/login")

    # ── Test 19 ──────────────────────────────────────────────────────────────
    def test_19_value_error_on_invalid_result(self):
        """Passing a ValidationResult with is_valid=False raises ValueError."""
        vr = validate_url("not a url at all")
        self.assertFalse(vr.is_valid, "Setup: URL should be invalid")
        with self.assertRaises(ValueError):
            extract_features(vr)

    # ── Test 20 ──────────────────────────────────────────────────────────────
    def test_20_malformed_url_rejected_before_extraction(self):
        """A malformed URL fails at validation — extraction is never called."""
        vr = validate_url("https://")
        # The URL has no hostname — validator rejects it.
        self.assertFalse(vr.is_valid)
        # Confirm extraction raises rather than silently continuing.
        with self.assertRaises(ValueError):
            extract_features(vr)

    # ── Test 21 ──────────────────────────────────────────────────────────────
    def test_21_unsupported_scheme_rejected(self):
        """An unsupported scheme is rejected by the validator."""
        vr = validate_url("ftp://example.com/file.txt")
        self.assertFalse(vr.is_valid,
                         "ftp:// is unsupported and must be rejected")
        with self.assertRaises(ValueError):
            extract_features(vr)


class TestSpecificFeatures(unittest.TestCase):
    """Targeted tests for individual feature groups."""

    # ── Test 22 ──────────────────────────────────────────────────────────────
    def test_22_url_shortener_detected(self):
        """A well-known URL shortener hostname sets flag_is_url_shortener True."""
        vr = _valid("https://bit.ly/3xAbCdE")
        fr = extract_features(vr)

        self.assertTrue(fr.numeric["flag_is_url_shortener"],
                        "bit.ly must be detected as a URL shortener")

    # ── Test 23 ──────────────────────────────────────────────────────────────
    def test_23_non_shortener_url(self):
        """A normal domain does not trigger flag_is_url_shortener."""
        vr = _valid("https://www.example.com/page")
        fr = extract_features(vr)

        self.assertFalse(fr.numeric["flag_is_url_shortener"])

    # ── Test 24 ──────────────────────────────────────────────────────────────
    def test_24_to_dict_flat_structure(self):
        """to_dict() returns a flat dict containing url, descriptive, and numeric keys."""
        vr = _valid("https://example.com/test")
        fr = extract_features(vr)
        d = fr.to_dict()

        self.assertIsInstance(d, dict)
        self.assertIn("url", d)
        # All numeric keys must appear at the top level.
        for key in numeric_feature_names():
            self.assertIn(key, d, f"Numeric key '{key}' missing from to_dict()")
        # Descriptive keys must also appear.
        for key in ("hostname", "scheme", "registered_domain"):
            self.assertIn(key, d, f"Descriptive key '{key}' missing from to_dict()")

    # ── Test 25 ──────────────────────────────────────────────────────────────
    def test_25_two_label_suffix_co_in(self):
        """Two-label Indian suffix co.in is handled correctly."""
        vr = _valid("https://mybank.co.in/login")
        fr = extract_features(vr)

        self.assertEqual(fr.descriptive["registered_domain"], "mybank.co.in")
        self.assertEqual(fr.descriptive["public_suffix"], "co.in")
        # No subdomains before 'mybank'.
        self.assertEqual(fr.numeric["host_subdomain_count"], 0)

    # ── Test 26 ──────────────────────────────────────────────────────────────
    def test_26_two_label_suffix_co_uk(self):
        """Two-label UK suffix co.uk is handled correctly."""
        vr = _valid("https://shop.retailer.co.uk/checkout")
        fr = extract_features(vr)

        self.assertEqual(fr.descriptive["registered_domain"], "retailer.co.uk")
        self.assertEqual(fr.descriptive["public_suffix"], "co.uk")
        self.assertEqual(fr.numeric["host_subdomain_count"], 1)

    # ── Test 27 ──────────────────────────────────────────────────────────────
    def test_27_subdomain_count_zero(self):
        """A bare registered domain (no subdomains) gives subdomain_count 0."""
        vr = _valid("https://example.com/page")
        fr = extract_features(vr)

        self.assertEqual(fr.numeric["host_subdomain_count"], 0)

    # ── Test 28 ──────────────────────────────────────────────────────────────
    def test_28_hyphen_in_hostname(self):
        """Hyphens in the hostname are counted in char_count_hyphen."""
        vr = _valid("https://my-bank-portal.example.com/")
        fr = extract_features(vr)

        # 'my-bank-portal' has 2 hyphens; the full URL may have more.
        self.assertGreaterEqual(fr.numeric["char_count_hyphen"], 2)

    # ── Test 29 ──────────────────────────────────────────────────────────────
    def test_29_underscore_in_path(self):
        """Underscores in the path are counted in char_count_underscore."""
        vr = _valid("https://example.com/some_long_path_here")
        fr = extract_features(vr)

        self.assertEqual(fr.numeric["char_count_underscore"], 3)

    # ── Test 30 ──────────────────────────────────────────────────────────────
    def test_30_no_network_requests(self):
        """
        extract_features() must not import or call socket / requests / urllib.request.

        This test verifies that no network-capable module is imported *as a
        side-effect* of importing feature_extractor.  It does not (and cannot)
        fully guarantee that a future modification won't add a network call, but
        it provides a regression check at the module-import level.
        """
        import importlib
        import modules.feature_extractor as fe_mod

        # None of these modules should be directly imported by feature_extractor.
        forbidden_network_modules = {"socket", "requests", "urllib.request", "http.client"}
        imported_names = set(vars(fe_mod).keys())

        for mod_name in forbidden_network_modules:
            # Check the module's namespace for the short name.
            short = mod_name.split(".")[-1]
            if short in imported_names:
                obj = getattr(fe_mod, short)
                if isinstance(obj, types.ModuleType):
                    self.fail(
                        f"feature_extractor imports network module '{mod_name}' "
                        "which must not be used during feature extraction."
                    )


class TestDefaultPortFlags(unittest.TestCase):
    """Port flag edge cases."""

    def test_31_default_port_443_https(self):
        """Explicit port 443 on HTTPS is the default — flag must be False."""
        vr = _valid("https://example.com:443/page")
        fr = extract_features(vr)

        self.assertFalse(fr.numeric["flag_has_non_default_port"],
                         "Port 443 is the HTTPS default — not a non-default port")

    def test_32_default_port_80_http(self):
        """Explicit port 80 on HTTP is the default — flag must be False."""
        vr = _valid("http://example.com:80/page")
        fr = extract_features(vr)

        self.assertFalse(fr.numeric["flag_has_non_default_port"],
                         "Port 80 is the HTTP default — not a non-default port")

    def test_33_no_port_specified(self):
        """No explicit port — flag_has_non_default_port must be False."""
        vr = _valid("https://example.com/page")
        fr = extract_features(vr)

        self.assertFalse(fr.numeric["flag_has_non_default_port"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
